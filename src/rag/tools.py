"""LangChain tool wrappers for existing procurement capabilities."""

from typing import Any

from langchain_core.tools import ToolException, tool

from models.cases import EmergencyCaseInput
from data.case_loader import CaseFileNotFoundError, load_case

from rag.retriever import (
    OpenAIEmbeddingProvider,
    RetrievalResult,
    get_collection,
    retrieve_diversified_chunks,
)
from integrations.usaspending_mcp import call_search_awards


@tool
def search_procurement_rules(
    query: str,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Search procurement rules using existing authority-aware retrieval and diversified chunks (MMR)."""

    return retrieve_diversified_chunks(
        query,
        collection=get_collection(),
        embedder=OpenAIEmbeddingProvider(),
        top_k=top_k,
    )


@tool
def get_case_facts(case_id: str) -> EmergencyCaseInput:
    """Load the known, unevaluated facts for an emergency procurement case."""

    try:
        return load_case(case_id)
    except (CaseFileNotFoundError, ValueError) as error:
        raise ToolException(str(error)) from error


@tool
def search_government_awards(
    keywords: list[str] | None = None,
    recipient_name: str | None = None,
    awarding_agency: str | None = None,
    time_period_start: str | None = None,
    time_period_end: str | None = None,
    naics_codes: list[str | int] | None = None,
    psc_codes: list[str | int] | None = None,
    award_amount_min: float | None = None,
    award_amount_max: float | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search federal contract awards for market research.

    Use this to identify vendors that received similar government awards and
    inspect comparable descriptions and amounts. Results are market
    intelligence only; they do not establish that an award is available for
    cooperative or piggyback use.
    """

    arguments: dict[str, Any] = {
        "award_type": "contracts",
        "limit": limit,
    }
    optional_arguments = {
        "keywords": keywords,
        "recipient_name": recipient_name,
        "awarding_agency": awarding_agency,
        "time_period_start": time_period_start,
        "time_period_end": time_period_end,
        "naics_codes": naics_codes,
        "psc_codes": psc_codes,
        "award_amount_min": award_amount_min,
        "award_amount_max": award_amount_max,
    }
    arguments.update(
        {
            name: value
            for name, value in optional_arguments.items()
            if value is not None
        }
    )

    try:
        return call_search_awards(arguments)
    except Exception as error:
        raise ToolException(
            f"USAspending MCP search_awards failed: {error}"
        ) from error
