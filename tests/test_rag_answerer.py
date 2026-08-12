"""Offline tests for grounded procurement-policy question answering."""

from typing import Any

import pytest

from rag.answerer import (
    AnswerGenerationError,
    GroundedAnswer,
    OpenAIAnswerGenerator,
    SourceReference,
    answer_question,
    build_context,
    print_retrieval_debug,
)
from rag.retriever import RetrievalDebugTrace, RetrievalResult


def make_result(
    chunk_id: str = "CA-PCC-1102-p0001-p0001-s001-c001",
    *,
    title: str = "California Public Contract Code § 1102",
    section: str = "1102",
    authority_level: str = "statute",
    page: int = 1,
) -> RetrievalResult:
    return RetrievalResult(
        distance=0.1,
        semantic_similarity=0.9,
        rerank_score=0.9,
        chunk_id=chunk_id,
        text="Emergency means a sudden, unexpected occurrence requiring action.",
        document_id="CA-PCC-1102",
        authority_level=authority_level,
        jurisdiction="California",
        section=section,
        page=page,
        page_end=page,
        title=title,
        agency="State of California",
    )


def model_source(chunk_id: str) -> SourceReference:
    return SourceReference(
        chunk_id=chunk_id,
        document_id="MODEL-VALUE",
        title="Model value",
        section="model-section",
        start_page=99,
        end_page=99,
        authority_level="model-value",
        jurisdiction="Model jurisdiction",
        agency="Model agency",
    )


class StubGenerator:
    def __init__(self, answer: GroundedAnswer | Exception) -> None:
        self.answer = answer
        self.calls: list[dict[str, str]] = []

    def generate(self, *, question: str, context: str) -> GroundedAnswer:
        self.calls.append({"question": question, "context": context})
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def fixed_retriever(results: list[RetrievalResult]) -> Any:
    def retrieve(*_args: Any, **_kwargs: Any) -> list[RetrievalResult]:
        return results

    return retrieve


def test_valid_grounded_response_preserves_trusted_source_metadata() -> None:
    result = make_result()
    generator = StubGenerator(
        GroundedAnswer(
            answer="Section 1102 defines an emergency as a sudden occurrence.",
            sources=[model_source(result.chunk_id)],
            insufficient_evidence=False,
            confidence=0.94,
        )
    )

    answer = answer_question(
        "What is an emergency?",
        collection=object(),
        embedder=object(),
        generator=generator,
        retriever=fixed_retriever([result]),
    )

    assert answer.insufficient_evidence is False
    assert answer.sources[0].document_id == "CA-PCC-1102"
    assert answer.sources[0].title == result.title
    assert answer.sources[0].section == "1102"
    assert answer.sources[0].start_page == 1
    assert answer.sources[0].authority_level == "statute"
    assert "SOURCE 1" in generator.calls[0]["context"]
    assert result.text in generator.calls[0]["context"]


def test_empty_question_is_rejected_before_retrieval() -> None:
    with pytest.raises(ValueError, match="question must not be blank"):
        answer_question("   ")


def test_no_retrieval_results_returns_insufficient_without_model_call() -> None:
    generator = StubGenerator(RuntimeError("must not be called"))

    answer = answer_question(
        "What approval threshold applies?",
        collection=object(),
        embedder=object(),
        generator=generator,
        retriever=fixed_retriever([]),
    )

    assert answer.insufficient_evidence is True
    assert answer.confidence == 0
    assert answer.sources == []
    assert "no relevant evidence" in answer.answer
    assert generator.calls == []


def test_model_can_return_an_insufficient_evidence_response() -> None:
    result = make_result()
    generator = StubGenerator(
        GroundedAnswer(
            answer="The source defines emergency but gives no approval threshold.",
            sources=[model_source(result.chunk_id)],
            insufficient_evidence=True,
            confidence=0.2,
        )
    )

    answer = answer_question(
        "What is the approval threshold?",
        collection=object(),
        embedder=object(),
        generator=generator,
        retriever=fixed_retriever([result]),
    )

    assert answer.insufficient_evidence is True
    assert answer.confidence == 0.2
    assert len(answer.sources) == 1


def test_multiple_sources_keep_retrieval_order_in_context() -> None:
    first = make_result("first", title="State statute")
    second = make_result(
        "second",
        title="Local policy",
        section="8.",
        authority_level="procurement_policy",
        page=8,
    )
    generator = StubGenerator(
        GroundedAnswer(
            answer="The statute defines emergency and the policy supplies procedure.",
            sources=[model_source("first"), model_source("second")],
            insufficient_evidence=False,
            confidence=0.85,
        )
    )

    answer = answer_question(
        "What rules apply?",
        collection=object(),
        embedder=object(),
        generator=generator,
        retriever=fixed_retriever([first, second]),
    )

    context = generator.calls[0]["context"]
    assert context.index("SOURCE 1") < context.index("SOURCE 2")
    assert context.index("State statute") < context.index("Local policy")
    assert [source.chunk_id for source in answer.sources] == ["first", "second"]


def test_openai_generator_uses_structured_low_temperature_response() -> None:
    parsed = GroundedAnswer(
        answer="Grounded answer.",
        sources=[model_source("chunk")],
        insufficient_evidence=False,
        confidence=0.8,
    )

    class FakeStructuredModel:
        def __init__(self) -> None:
            self.messages: list[tuple[str, str]] = []

        def invoke(self, messages: list[tuple[str, str]]) -> GroundedAnswer:
            self.messages = messages
            return parsed

    class FakeChatModel:
        def __init__(self) -> None:
            self.schema: type[GroundedAnswer] | None = None
            self.method: str | None = None
            self.structured_model = FakeStructuredModel()

        def with_structured_output(
            self,
            schema: type[GroundedAnswer],
            *,
            method: str,
        ) -> FakeStructuredModel:
            self.schema = schema
            self.method = method
            return self.structured_model

    chat_model = FakeChatModel()
    generator = OpenAIAnswerGenerator(
        chat_model=chat_model,
        model="test-model",
    )

    answer = generator.generate(question="Question?", context="SOURCE 1")

    assert answer == parsed
    assert generator.model == "test-model"
    assert generator.temperature == 0.1
    assert chat_model.schema is GroundedAnswer
    assert chat_model.method == "json_schema"
    messages = chat_model.structured_model.messages
    assert messages[0][0] == "system"
    assert messages[1] == (
        "human",
        "QUESTION:\nQuestion?\n\nRETRIEVED EVIDENCE:\nSOURCE 1",
    )
    instructions = " ".join(messages[0][1].split())
    assert "only from the retrieved evidence" in instructions


def test_model_api_failure_is_wrapped() -> None:
    generator = StubGenerator(RuntimeError("API unavailable"))

    with pytest.raises(
        AnswerGenerationError,
        match="grounded answer generation failed",
    ):
        answer_question(
            "What is an emergency?",
            collection=object(),
            embedder=object(),
            generator=generator,
            retriever=fixed_retriever([make_result()]),
        )


def test_context_formatter_includes_page_span_and_metadata() -> None:
    result = make_result()
    result.page_end = 2

    context = build_context([result])

    assert "Document ID: CA-PCC-1102" in context
    assert "Authority level: statute" in context
    assert "Jurisdiction: California" in context
    assert "Agency: State of California" in context
    assert "Pages: 1-2" in context


def test_retrieval_debug_output_shows_each_ranking_stage(capsys) -> None:
    raw = make_result("raw", section="2.24.060")
    reranked = make_result("reranked", section="2.24.080")
    reranked.rerank_score = 0.82
    final = make_result("final", section="2.24.090")
    final.rerank_score = 0.81
    trace = RetrievalDebugTrace(
        raw_semantic_results=[raw],
        authority_reranked_results=[reranked],
        diversified_results=[final],
        final_results=[final],
        context_strategy="diversified",
    )

    print_retrieval_debug(trace)

    output = capsys.readouterr().out
    assert "RAW SEMANTIC RESULTS" in output
    assert "AUTHORITY-RERANKED RESULTS" in output
    assert "FINAL DIVERSIFIED/MMR SELECTION" in output
    assert "FINAL CHUNKS PASSED TO THE LLM" in output
    assert "section=2.24.060" in output
    assert "chunk_id=reranked" in output
    assert "distance=0.100000" in output
    assert "authority=statute" in output
    assert "rerank_score=0.820000" in output
