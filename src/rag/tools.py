"""LangChain tool wrappers for existing procurement capabilities."""

from langchain_core.tools import ToolException, tool

from models.cases import EmergencyCaseInput
from data.case_loader import CaseFileNotFoundError, load_case

from rag.retriever import (
    OpenAIEmbeddingProvider,
    RetrievalResult,
    get_collection,
    retrieve_diversified_chunks,
)


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
