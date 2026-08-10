"""OpenAI embedding and local Chroma retrieval for procurement policies."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag.ingest import (
    DEFAULT_MUNICIPAL_CODE_DOCX_PATH,
    DEFAULT_ORDINANCE_TEXT_PATH,
    DEFAULT_PDF_PATH,
    ingest_docx,
    ingest_page_marked_text,
    ingest_pdf,
    santa_monica_metadata,
    santa_monica_municipal_code_metadata,
    santa_monica_ordinance_metadata,
)
from rag.metadata import DocumentChunk


DEFAULT_COLLECTION_NAME = "procurement_policies"
DEFAULT_CHROMA_PATH = Path(".chroma")
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CANDIDATE_MULTIPLIER = 4
DEFAULT_SEMANTIC_WEIGHT = 0.9
DEFAULT_AUTHORITY_WEIGHT = 0.1

AUTHORITY_PRIORITY = {
    "statute": 5,
    "local_law": 4,
    "procurement_policy": 3,
    "administrative_instruction": 2,
    "local_executive_order": 1,
}

class EmbeddingProvider(Protocol):
    """Minimal interface shared by OpenAI and offline test embedders."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector for each input string."""


class OpenAIEmbeddingProvider:
    """Create text embeddings through the OpenAI embeddings API."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        client: Any | None = None,
    ) -> None:
        if client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY must be set to generate embeddings"
                )
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self._client = client
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )
        return [
            item.embedding
            for item in sorted(response.data, key=lambda item: item.index)
        ]


class RetrievalResult(BaseModel):
    """A procurement-policy chunk returned by semantic retrieval."""

    model_config = ConfigDict(extra="forbid")

    distance: float = Field(ge=0)
    semantic_similarity: float
    rerank_score: float = 0.0
    chunk_id: str
    text: str
    authority_level: str
    section: str | None = None
    page: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    title: str
    agency: str


def get_collection(
    *,
    db_path: str | Path = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    rebuild: bool = False,
) -> Any:
    """Open a persistent local Chroma collection."""

    import chromadb
    from chromadb.errors import NotFoundError

    client = chromadb.PersistentClient(path=str(db_path))
    if rebuild:
        try:
            client.delete_collection(collection_name)
        except NotFoundError:
            pass
    return client.get_or_create_collection(
        name=collection_name,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )


def _chroma_metadata(chunk: DocumentChunk) -> dict[str, str | int | float | bool]:
    """Convert validated procurement metadata to Chroma scalar values."""

    values = chunk.metadata.model_dump(mode="json", exclude_none=True)
    return {
        key: value
        for key, value in values.items()
        if isinstance(value, (str, int, float, bool))
    }


def index_chunks(
    chunks: list[DocumentChunk],
    *,
    collection: Any,
    embedder: EmbeddingProvider,
) -> int:
    """Embed and store chunks that are not already present by chunk ID."""

    if not chunks:
        return 0

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    existing = collection.get(ids=list(chunk_by_id), include=[])
    existing_ids = set(existing.get("ids", []))
    new_chunks = [
        chunk
        for chunk_id, chunk in chunk_by_id.items()
        if chunk_id not in existing_ids
    ]
    if not new_chunks:
        return 0

    embeddings = embedder.embed([chunk.text for chunk in new_chunks])
    if len(embeddings) != len(new_chunks):
        raise ValueError("embedding provider returned the wrong vector count")

    collection.add(
        ids=[chunk.chunk_id for chunk in new_chunks],
        documents=[chunk.text for chunk in new_chunks],
        embeddings=embeddings,
        metadatas=[_chroma_metadata(chunk) for chunk in new_chunks],
    )
    return len(new_chunks)


def retrieve_chunks(
    query: str,
    *,
    collection: Any,
    embedder: EmbeddingProvider,
    top_k: int = 3,
    candidate_k: int | None = None,
    authority_priority: dict[str, int] = AUTHORITY_PRIORITY,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    authority_weight: float = DEFAULT_AUTHORITY_WEIGHT,
) -> list[RetrievalResult]:
    """Retrieve semantic candidates, then rerank by similarity and authority."""

    if not query.strip():
        raise ValueError("query must not be blank")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if semantic_weight < 0 or authority_weight < 0:
        raise ValueError("reranking weights must not be negative")
    if semantic_weight + authority_weight == 0:
        raise ValueError("at least one reranking weight must be positive")

    semantic_candidate_count = candidate_k or (
        top_k * DEFAULT_CANDIDATE_MULTIPLIER
    )
    if semantic_candidate_count < top_k:
        raise ValueError("candidate_k must be at least top_k")

    query_embedding = embedder.embed([query])[0]
    response = collection.query(
        query_embeddings=[query_embedding],
        n_results=semantic_candidate_count,
        include=["documents", "metadatas", "distances"],
    )
    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    candidates = [
        RetrievalResult(
            distance=distance,
            semantic_similarity=max(-1.0, min(1.0, 1.0 - distance)),
            chunk_id=chunk_id,
            text=text,
            authority_level=metadata["authority_level"],
            section=metadata.get("section"),
            page=metadata.get("page"),
            page_end=metadata.get("page_end", metadata.get("page")),
            title=metadata["title"],
            agency=metadata["agency"],
        )
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        )
    ]
    return rerank_results(
        candidates,
        top_k=top_k,
        authority_priority=authority_priority,
        semantic_weight=semantic_weight,
        authority_weight=authority_weight,
    )


def rerank_results(
    candidates: list[RetrievalResult],
    *,
    top_k: int,
    authority_priority: dict[str, int] = AUTHORITY_PRIORITY,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    authority_weight: float = DEFAULT_AUTHORITY_WEIGHT,
) -> list[RetrievalResult]:
    """Combine semantic similarity and authority priority into a final score."""

    max_priority = max(authority_priority.values(), default=1)
    total_weight = semantic_weight + authority_weight

    for candidate in candidates:
        authority_score = (
            authority_priority.get(candidate.authority_level, 0) / max_priority
        )
        candidate.rerank_score = (
            semantic_weight * candidate.semantic_similarity
            + authority_weight * authority_score
        ) / total_weight

    return sorted(
        candidates,
        key=lambda result: (
            -result.rerank_score,
            -result.semantic_similarity,
            result.chunk_id,
        ),
    )[:top_k]


def _print_results(results: list[RetrievalResult]) -> None:
    for rank, result in enumerate(results, start=1):
        pages = str(result.page)
        if result.page_end != result.page:
            pages = f"{result.page}-{result.page_end}"
        print(
            f"\n{rank}. score={result.rerank_score:.4f} | "
            f"distance={result.distance:.4f} | "
            f"{result.chunk_id} | section={result.section or '-'} | pages={pages}"
        )
        print(
            f"   {result.title} — {result.agency} "
            f"[{result.authority_level}]"
        )
        print(f"   {result.text}")


def index_main() -> None:
    """Ingest and index the Santa Monica procurement corpus in local Chroma."""

    parser = argparse.ArgumentParser(description=index_main.__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument(
        "--ordinance-text",
        type=Path,
        default=DEFAULT_ORDINANCE_TEXT_PATH,
    )
    parser.add_argument(
        "--municipal-code-docx",
        type=Path,
        default=DEFAULT_MUNICIPAL_CODE_DOCX_PATH,
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Replace the existing collection before indexing.",
    )
    args = parser.parse_args()

    _, emergency_chunks = ingest_pdf(
        args.pdf,
        santa_monica_metadata(args.pdf),
    )
    _, ordinance_chunks = ingest_page_marked_text(
        args.ordinance_text,
        santa_monica_ordinance_metadata(args.ordinance_text),
    )
    _, municipal_code_chunks = ingest_docx(
        args.municipal_code_docx,
        santa_monica_municipal_code_metadata(args.municipal_code_docx),
    )
    chunks = emergency_chunks + ordinance_chunks + municipal_code_chunks
    added = index_chunks(
        chunks,
        collection=get_collection(db_path=args.db, rebuild=args.rebuild),
        embedder=OpenAIEmbeddingProvider(),
    )
    print(
        f"Indexed {added} new chunks "
        f"({len(chunks)} total across 3 documents)."
    )


def query_main() -> None:
    """Query indexed procurement-policy chunks from the command line."""

    parser = argparse.ArgumentParser(description=query_main.__doc__)
    parser.add_argument("query", nargs="?")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=DEFAULT_SEMANTIC_WEIGHT,
    )
    parser.add_argument(
        "--authority-weight",
        type=float,
        default=DEFAULT_AUTHORITY_WEIGHT,
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_CHROMA_PATH)
    args = parser.parse_args()
    collection = get_collection(db_path=args.db)
    embedder = OpenAIEmbeddingProvider()

    if args.query:
        _print_results(
            retrieve_chunks(
                args.query,
                collection=collection,
                embedder=embedder,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                semantic_weight=args.semantic_weight,
                authority_weight=args.authority_weight,
            )
        )
        return

    print("Enter a procurement question (blank line to exit).")
    while query := input("> ").strip():
        _print_results(
            retrieve_chunks(
                query,
                collection=collection,
                embedder=embedder,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                semantic_weight=args.semantic_weight,
                authority_weight=args.authority_weight,
            )
        )


if __name__ == "__main__":
    query_main()
