"""Tests for the thin LangChain procurement retrieval tool."""

from typing import Any

import pytest
from langchain_core.tools import ToolException

from models.cases import EmergencyCaseInput
from rag.retriever import RetrievalResult
from rag.tools import get_case_facts, search_procurement_rules
from data.case_loader import CaseFileNotFoundError


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


def make_case() -> EmergencyCaseInput:
    return EmergencyCaseInput.model_validate(
        {
            "schema_version": "1.0",
            "case_id": "EM-001",
            "title": "Emergency Sewer Main Repair",
            "workflow_type": "emergency_procurement",
            "jurisdiction": {
                "state": "California",
                "agency": "Pilot City",
            },
            "department": "Public Works",
            "estimated_amount_usd": 184000,
            "proposed_vendor": "Inland Utility Contractors",
            "request_text": "A ruptured sewer main requires immediate repair.",
            "available_documents": [
                {
                    "document_id": "EM001-D01",
                    "title": "Public Works Incident Report",
                    "summary": "Documents the rupture and current conditions.",
                }
            ],
        }
    )


def test_get_case_facts_calls_existing_loader_and_preserves_case(
    monkeypatch,
) -> None:
    expected = make_case()
    before = expected.model_dump()
    requested_ids: list[str] = []

    def fake_load_case(case_id: str) -> EmergencyCaseInput:
        requested_ids.append(case_id)
        return expected

    monkeypatch.setattr("rag.tools.load_case", fake_load_case)

    result = get_case_facts.invoke({"case_id": "em-001"})

    assert requested_ids == ["em-001"]
    assert result is expected
    assert result.case_id == "EM-001"
    assert result.title == "Emergency Sewer Main Repair"
    assert result.jurisdiction.state == "California"
    assert result.jurisdiction.agency == "Pilot City"
    assert result.department == "Public Works"
    assert result.estimated_amount_usd == 184000
    assert result.proposed_vendor == "Inland Utility Contractors"
    assert result.available_documents[0].summary.startswith("Documents")
    assert expected.model_dump() == before


def test_get_case_facts_handles_unknown_case_safely(monkeypatch) -> None:
    def missing_case(_case_id: str) -> EmergencyCaseInput:
        raise CaseFileNotFoundError("Case EM-999 was not found")

    monkeypatch.setattr("rag.tools.load_case", missing_case)

    with pytest.raises(ToolException, match="EM-999.*not found"):
        get_case_facts.invoke({"case_id": "EM-999"})
