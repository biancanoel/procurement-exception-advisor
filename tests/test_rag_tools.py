"""Tests for the thin LangChain procurement retrieval tool."""

from typing import Any

from rag.retriever import RetrievalResult
from rag.tools import search_procurement_rules


def make_result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        distance=1.0 - score,
        semantic_similarity=score,
        rerank_score=score,
        diversity_score=score,
        max_selected_overlap=0.0,
        chunk_id=chunk_id,
        text=f"Rule text for {chunk_id}",
        document_id="SM-MUNICIPAL-CODE-2.24",
        authority_level="local_law",
        jurisdiction="California",
        section=chunk_id,
        page=1,
        page_end=1,
        title="Santa Monica Municipal Code Chapter 2.24",
        agency="City of Santa Monica",
    )


def test_search_procurement_rules_invokes_existing_retriever(monkeypatch) -> None:
    expected = [
        make_result("2.24.090", 0.91),
        make_result("2.24.060", 0.87),
        make_result("2.24.080", 0.84),
    ]
    collection = object()
    embedder = object()
    captured: dict[str, Any] = {}

    monkeypatch.setattr("rag.tools.get_collection", lambda: collection)
    monkeypatch.setattr("rag.tools.OpenAIEmbeddingProvider", lambda: embedder)

    def fake_retriever(
        query: str,
        *,
        collection: Any,
        embedder: Any,
        top_k: int,
    ) -> list[RetrievalResult]:
        captured.update(
            query=query,
            collection=collection,
            embedder=embedder,
            top_k=top_k,
        )
        return expected

    monkeypatch.setattr("rag.tools.retrieve_diversified_chunks", fake_retriever)

    results = search_procurement_rules.invoke(
        {
            "query": "Who may make emergency purchases?",
            "top_k": 3,
        }
    )

    assert captured == {
        "query": "Who may make emergency purchases?",
        "collection": collection,
        "embedder": embedder,
        "top_k": 3,
    }
    assert results is expected
    assert all(isinstance(result, RetrievalResult) for result in results)
    assert [result.chunk_id for result in results] == [
        "2.24.090",
        "2.24.060",
        "2.24.080",
    ]


def test_search_procurement_rules_uses_default_top_k(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setattr("rag.tools.get_collection", object)
    monkeypatch.setattr("rag.tools.OpenAIEmbeddingProvider", object)

    def fake_retriever(
        query: str,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        captured["query"] = query
        captured["top_k"] = kwargs["top_k"]
        return []

    monkeypatch.setattr("rag.tools.retrieve_diversified_chunks", fake_retriever)

    results = search_procurement_rules.invoke({"query": "emergency rules"})

    assert results == []
    assert captured == {"query": "emergency rules", "top_k": 5}
