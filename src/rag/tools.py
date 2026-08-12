"""LangChain tool wrappers for existing procurement RAG capabilities."""

from langchain_core.tools import tool

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
