"""Offline tests for Chroma indexing and semantic retrieval adapters."""

from types import SimpleNamespace
from typing import Any

from rag.metadata import DocumentChunk, ProcurementDocumentMetadata
from rag.retriever import (
    AUTHORITY_PRIORITY,
    OpenAIEmbeddingProvider,
    get_collection,
    index_chunks,
    rerank_results,
    retrieve_chunks,
)


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)), 1.0] for text in texts]


class FakeCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def get(self, *, ids: list[str], include: list[str]) -> dict[str, Any]:
        del include
        return {"ids": [chunk_id for chunk_id in ids if chunk_id in self.records]}

    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for chunk_id, document, embedding, metadata in zip(
            ids, documents, embeddings, metadatas, strict=True
        ):
            self.records[chunk_id] = {
                "document": document,
                "embedding": embedding,
                "metadata": metadata,
            }

    def query(self, **_kwargs: Any) -> dict[str, Any]:
        chunk_id, record = next(iter(self.records.items()))
        return {
            "ids": [[chunk_id]],
            "documents": [[record["document"]]],
            "metadatas": [[record["metadata"]]],
            "distances": [[0.125]],
        }


def make_chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="POLICY-p0009-p0009-s009-c001",
        text="8. The City Manager may waive competitive solicitation procedures.",
        metadata=ProcurementDocumentMetadata(
            document_id="POLICY",
            title="Emergency Procurement Policy",
            jurisdiction="California",
            agency="City of Santa Monica",
            document_type="emergency_declaration",
            authority_level="local_executive_order",
            exception_type="emergency",
            section="8.",
            page=9,
            page_end=9,
            source_path="policy.pdf",
        ),
    )


def test_openai_provider_batches_text_and_preserves_response_order() -> None:
    class FakeEmbeddingsAPI:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            assert kwargs["model"] == "text-embedding-3-small"
            assert kwargs["input"] == ["first", "second"]
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, embedding=[2.0]),
                    SimpleNamespace(index=0, embedding=[1.0]),
                ]
            )

    client = SimpleNamespace(embeddings=FakeEmbeddingsAPI())
    provider = OpenAIEmbeddingProvider(client=client)

    assert provider.embed(["first", "second"]) == [[1.0], [2.0]]


def test_indexing_stores_chunk_embedding_text_and_metadata() -> None:
    collection = FakeCollection()
    embedder = FakeEmbedder()

    added = index_chunks(
        [make_chunk()], collection=collection, embedder=embedder
    )

    assert added == 1
    record = collection.records[make_chunk().chunk_id]
    assert record["document"].startswith("8. The City Manager")
    assert record["embedding"]
    assert record["metadata"]["section"] == "8."
    assert record["metadata"]["page"] == 9


def test_reindexing_same_chunk_does_not_duplicate_or_reembed() -> None:
    collection = FakeCollection()
    embedder = FakeEmbedder()
    chunk = make_chunk()

    assert index_chunks([chunk], collection=collection, embedder=embedder) == 1
    assert index_chunks([chunk], collection=collection, embedder=embedder) == 0

    assert len(collection.records) == 1
    assert len(embedder.calls) == 1


def test_retrieval_returns_required_policy_fields() -> None:
    collection = FakeCollection()
    embedder = FakeEmbedder()
    index_chunks([make_chunk()], collection=collection, embedder=embedder)

    results = retrieve_chunks(
        "Can bidding be waived?",
        collection=collection,
        embedder=embedder,
        top_k=3,
    )

    assert len(results) == 1
    assert results[0].distance == 0.125
    assert results[0].semantic_similarity == 0.875
    assert results[0].rerank_score > 0
    assert results[0].authority_level == "local_executive_order"
    assert results[0].section == "8."
    assert results[0].page == 9
    assert results[0].page_end == 9
    assert results[0].title == "Emergency Procurement Policy"
    assert results[0].agency == "City of Santa Monica"


def test_retrieval_reranks_larger_candidate_set_by_authority() -> None:
    class CandidateCollection:
        requested_count: int | None = None

        def query(self, **kwargs: Any) -> dict[str, Any]:
            self.requested_count = kwargs["n_results"]
            base_metadata = {
                "page": 1,
                "page_end": 1,
                "title": "Policy source",
                "agency": "Test Agency",
            }
            return {
                "ids": [["executive", "policy", "law"]],
                "documents": [["Executive", "Policy", "Law"]],
                "metadatas": [[
                    {
                        **base_metadata,
                        "authority_level": "local_executive_order",
                    },
                    {**base_metadata, "authority_level": "procurement_policy"},
                    {**base_metadata, "authority_level": "local_law"},
                ]],
                "distances": [[0.05, 0.08, 0.10]],
            }

    collection = CandidateCollection()
    results = retrieve_chunks(
        "emergency authority",
        collection=collection,
        embedder=FakeEmbedder(),
        top_k=3,
    )

    assert collection.requested_count == 12
    assert [result.chunk_id for result in results] == [
        "law",
        "policy",
        "executive",
    ]


def test_reranking_priority_mapping_is_configurable() -> None:
    candidates = [
        make_result("law", "local_law", similarity=0.90),
        make_result("executive", "local_executive_order", similarity=0.89),
    ]

    results = rerank_results(
        candidates,
        top_k=2,
        authority_priority={**AUTHORITY_PRIORITY, "local_executive_order": 10},
    )

    assert results[0].chunk_id == "executive"


def test_rebuild_replaces_existing_chroma_collection(tmp_path) -> None:
    collection = get_collection(db_path=tmp_path)
    collection.add(
        ids=["old"],
        documents=["Old document"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"authority_level": "local_law"}],
    )

    rebuilt = get_collection(db_path=tmp_path, rebuild=True)

    assert rebuilt.count() == 0


def make_result(
    chunk_id: str,
    authority_level: str,
    *,
    similarity: float,
) -> Any:
    from rag.retriever import RetrievalResult

    return RetrievalResult(
        distance=1.0 - similarity,
        semantic_similarity=similarity,
        chunk_id=chunk_id,
        text="Text",
        authority_level=authority_level,
        title="Title",
        agency="Agency",
    )
