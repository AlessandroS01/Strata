"""Unit tests for strata.indexing.qdrant_store (Embedded Qdrant VectorStore adapter)."""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from strata.core.config import Settings
from strata.core.models import Chunk, RetrievedChunk
from strata.core.protocols import VectorStore
from strata.indexing.qdrant_store import QdrantVectorStore

TEST_DIM = 4


@pytest.fixture
def test_settings() -> Settings:
    """Provide isolated Settings configured with 4-dimensional vectors."""
    return Settings(
        embedding_dimension=TEST_DIM,
        qdrant_collection_name="test_strata_chunks",
        distance_metric="Cosine",
    )


@pytest.fixture
def memory_store(test_settings: Settings) -> Iterator[QdrantVectorStore]:
    """Provide an in-memory QdrantVectorStore fixture ensuring proper connection closure."""
    store = QdrantVectorStore(settings=test_settings, location=":memory:")
    yield store
    store.close()


def make_chunk(
    chunk_id: str = "chunk-1",
    doc_id: str = "doc-1",
    index: int = 0,
    parent_section_id: str | None = None,
    chunk_hash: str | None = None,
    text: str = "Sample chunk text",
    heading_path: str = "Doc > Intro",
    token_count: int = 25,
    tags: list[str] | None = None,
) -> Chunk:
    """Helper to construct a valid Chunk domain model."""
    return Chunk(
        id=chunk_id,
        document_id=doc_id,
        chunk_hash=chunk_hash or f"hash-{chunk_id}",
        text=text,
        heading_path=heading_path,
        index=index,
        parent_section_id=parent_section_id,
        token_count=token_count,
        tags=tags if tags is not None else ["python", "search"],
    )


# -----------------------------------------------------------------------------
# Protocol & Initialization Tests
# -----------------------------------------------------------------------------


def test_implements_vector_store_protocol(memory_store: QdrantVectorStore) -> None:
    """Verify QdrantVectorStore satisfies the runtime VectorStore Protocol."""
    assert isinstance(memory_store, VectorStore)
    assert VectorStore in QdrantVectorStore.__mro__


def test_init_with_preconfigured_client(test_settings: Settings) -> None:
    """Verify QdrantVectorStore accepts an existing QdrantClient."""
    client = QdrantClient(location=":memory:")
    with QdrantVectorStore(settings=test_settings, client=client) as store:
        assert store.client is client
        assert store.count() == 0


def test_init_with_default_settings_and_disk_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify initialization creates on-disk directories when client and location are None."""
    qdrant_dir = tmp_path / "custom_qdrant_data"
    custom_settings = Settings(
        qdrant_path=qdrant_dir,
        embedding_dimension=TEST_DIM,
        qdrant_collection_name="disk_chunks",
    )
    monkeypatch.setattr("strata.indexing.qdrant_store.get_settings", lambda: custom_settings)

    with QdrantVectorStore() as store:
        assert store.settings.qdrant_path == qdrant_dir
        assert store.collection_name == "disk_chunks"
        assert qdrant_dir.exists()
        assert store.count() == 0


def test_init_with_custom_collection_name(test_settings: Settings) -> None:
    """Verify explicit collection_name overrides settings default."""
    custom_name = "override_collection"
    with QdrantVectorStore(
        settings=test_settings, location=":memory:", collection_name=custom_name
    ) as store:
        assert store.collection_name == custom_name
        assert store.client.collection_exists(custom_name)


def test_ensure_collection_skips_creation_if_already_exists(test_settings: Settings) -> None:
    """Verify _ensure_collection does not recreate or fail when collection already exists."""
    client = QdrantClient(location=":memory:")
    try:
        # Initialize first store (creates collection)
        store1 = QdrantVectorStore(settings=test_settings, client=client)
        assert client.collection_exists(store1.collection_name)
        # Calling _ensure_collection directly when collection already exists
        store1._ensure_collection()

        # Re-wrap the same active client with a second store instance
        store2 = QdrantVectorStore(settings=test_settings, client=client)
        assert store2.client.collection_exists(store2.collection_name)
    finally:
        client.close()


# -----------------------------------------------------------------------------
# Distance Metric & ID Normalization Tests
# -----------------------------------------------------------------------------


def test_resolve_distance_metric_variants() -> None:
    """Verify valid distance metrics resolve correctly with whitespace/case handling."""
    for metric_name, expected in [
        ("Cosine", models.Distance.COSINE),
        ("  cosine  ", models.Distance.COSINE),
        ("dot", models.Distance.DOT),
        ("EUCLID", models.Distance.EUCLID),
    ]:
        s = Settings(distance_metric=metric_name, embedding_dimension=TEST_DIM)
        with QdrantVectorStore(settings=s, location=":memory:") as store:
            assert store._resolve_distance_metric() == expected


def test_resolve_distance_metric_unknown_fallback_to_cosine(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify unknown distance metric logs a warning and defaults to COSINE."""
    s = Settings(distance_metric="UNKNOWN_METRIC", embedding_dimension=TEST_DIM)
    with caplog.at_level("WARNING"), QdrantVectorStore(settings=s, location=":memory:") as store:
        assert store._resolve_distance_metric() == models.Distance.COSINE
    assert "Unknown distance metric 'UNKNOWN_METRIC', falling back to COSINE" in caplog.text


def test_to_point_id_valid_uuid() -> None:
    """Verify valid UUID strings are preserved as point IDs."""
    raw_uuid = "12345678-1234-5678-1234-567812345678"
    point_id = QdrantVectorStore._to_point_id(raw_uuid)
    assert point_id == raw_uuid


def test_to_point_id_non_uuid_string() -> None:
    """Verify arbitrary non-UUID string IDs are converted deterministically to UUID5."""
    chunk_id = "chunk_section_42"
    point_id = QdrantVectorStore._to_point_id(chunk_id)
    expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
    assert point_id == expected
    assert QdrantVectorStore._to_point_id(chunk_id) == expected


# -----------------------------------------------------------------------------
# Upsert Tests
# -----------------------------------------------------------------------------


def test_upsert_chunks_embeddings_and_vectors_kwarg(memory_store: QdrantVectorStore) -> None:
    """Verify upsert_chunks functions with positional embeddings and keyword vectors."""
    c1 = make_chunk(chunk_id="c1", parent_section_id="sec-1")
    v1 = [0.1, 0.2, 0.3, 0.4]

    # Upsert with embeddings parameter
    memory_store.upsert_chunks([c1], embeddings=[v1])
    assert memory_store.count() == 1

    # Upsert with vectors kwarg
    c2 = make_chunk(chunk_id="c2", parent_section_id=None)
    v2 = [0.5, 0.6, 0.7, 0.8]
    memory_store.upsert_chunks([c2], vectors=[v2])
    assert memory_store.count() == 2


def test_upsert_chunks_missing_vectors_raises(memory_store: QdrantVectorStore) -> None:
    """Verify ValueError is raised if neither embeddings nor vectors are provided."""
    c1 = make_chunk()
    with pytest.raises(ValueError, match="Either 'embeddings' or 'vectors' must be provided"):
        memory_store.upsert_chunks([c1])


def test_upsert_chunks_length_mismatch_raises(memory_store: QdrantVectorStore) -> None:
    """Verify ValueError is raised when chunks and vectors count differ."""
    c1 = make_chunk(chunk_id="c1")
    c2 = make_chunk(chunk_id="c2")
    with pytest.raises(ValueError, match="Mismatch between number of chunks"):
        memory_store.upsert_chunks([c1, c2], embeddings=[[0.1, 0.2, 0.3, 0.4]])


def test_upsert_chunks_empty_list_returns_immediately(memory_store: QdrantVectorStore) -> None:
    """Verify upserting an empty chunk list returns early without operations."""
    memory_store.upsert_chunks([], embeddings=[])
    assert memory_store.count() == 0


def test_upsert_chunks_dimension_mismatch_raises(memory_store: QdrantVectorStore) -> None:
    """Verify ValueError is raised if a vector dimension differs from settings."""
    c1 = make_chunk(chunk_id="c1")
    wrong_dim_vector = [0.1, 0.2]
    with pytest.raises(ValueError, match="Vector dimension 2 does not match expected dimension 4"):
        memory_store.upsert_chunks([c1], embeddings=[wrong_dim_vector])


# -----------------------------------------------------------------------------
# Search Tests
# -----------------------------------------------------------------------------


def test_search_empty_query_or_nonpositive_limit(memory_store: QdrantVectorStore) -> None:
    """Verify search returns empty list for empty vector or non-positive limits."""
    query = [0.1, 0.2, 0.3, 0.4]
    assert memory_store.search(query_vector=[], limit=10) == []
    assert memory_store.search(query_vector=query, limit=0) == []
    assert memory_store.search(query_vector=query, limit=-5) == []


def test_search_dimension_mismatch_raises(memory_store: QdrantVectorStore) -> None:
    """Verify ValueError is raised when query vector dimension does not match settings."""
    wrong_query = [0.1, 0.2]
    with pytest.raises(
        ValueError, match="Query vector dimension 2 does not match expected dimension 4"
    ):
        memory_store.search(query_vector=wrong_query, limit=5)


def test_search_ranking_and_chunk_hydration(memory_store: QdrantVectorStore) -> None:
    """Verify search ranks results by cosine similarity and hydrates RetrievedChunk models."""
    c1 = make_chunk(
        chunk_id="c1",
        doc_id="doc-1",
        chunk_hash="hash-1",
        heading_path="Doc > Sec1",
        index=0,
        token_count=10,
        tags=["python", "async"],
        parent_section_id="sec-1",
    )
    c2 = make_chunk(
        chunk_id="c2",
        doc_id="doc-2",
        chunk_hash="hash-2",
        heading_path="Doc > Sec2",
        index=1,
        token_count=20,
        tags=["rust", "systems"],
        parent_section_id=None,
    )

    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0, 0.0]
    memory_store.upsert_chunks([c1, c2], embeddings=[v1, v2])

    results = memory_store.search(query_vector=[1.0, 0.0, 0.0, 0.0], limit=2)
    assert len(results) == 2

    top_hit = results[0]
    assert isinstance(top_hit, RetrievedChunk)
    assert top_hit.chunk.id == "c1"
    assert top_hit.chunk.document_id == "doc-1"
    assert top_hit.chunk.heading_path == "Doc > Sec1"
    assert top_hit.chunk.chunk_hash == "hash-1"
    assert top_hit.chunk.index == 0
    assert top_hit.chunk.token_count == 10
    assert top_hit.chunk.tags == ["python", "async"]
    assert top_hit.chunk.parent_section_id == "sec-1"
    assert top_hit.chunk.text == ""
    assert top_hit.dense_rank == 1
    assert pytest.approx(top_hit.score, rel=1e-3) == 1.0

    second_hit = results[1]
    assert second_hit.chunk.id == "c2"
    assert second_hit.chunk.parent_section_id is None
    assert second_hit.dense_rank == 2


def test_search_with_tags_filter(memory_store: QdrantVectorStore) -> None:
    """Verify search filtering by tags parameter."""
    c1 = make_chunk(chunk_id="c1", tags=["database", "sql"])
    c2 = make_chunk(chunk_id="c2", tags=["nosql", "graph"])
    memory_store.upsert_chunks(
        [c1, c2],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
    )

    hits = memory_store.search(
        query_vector=[1.0, 0.0, 0.0, 0.0],
        limit=5,
        tags=["database"],
    )
    assert len(hits) == 1
    assert hits[0].chunk.id == "c1"


def test_search_payload_filter_tags_list(memory_store: QdrantVectorStore) -> None:
    """Verify payload_filter with 'tags' key as a list."""
    c1 = make_chunk(chunk_id="c1", tags=["sql"])
    c2 = make_chunk(chunk_id="c2", tags=["graph"])
    memory_store.upsert_chunks([c1, c2], embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    results = memory_store.search(
        query_vector=[1.0, 0.0, 0.0, 0.0],
        limit=5,
        payload_filter={"tags": ["sql"]},
    )
    assert len(results) == 1
    assert results[0].chunk.id == "c1"


def test_search_payload_filter_tags_scalar(memory_store: QdrantVectorStore) -> None:
    """Verify payload_filter with 'tags' key as a scalar string."""
    c1 = make_chunk(chunk_id="c1", tags=["sql"])
    c2 = make_chunk(chunk_id="c2", tags=["graph"])
    memory_store.upsert_chunks([c1, c2], embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    results = memory_store.search(
        query_vector=[1.0, 0.0, 0.0, 0.0],
        limit=5,
        payload_filter={"tags": "sql"},
    )
    assert len(results) == 1
    assert results[0].chunk.id == "c1"


def test_search_payload_filter_field_list_and_tuple(memory_store: QdrantVectorStore) -> None:
    """Verify payload_filter with non-tags key as list and tuple."""
    c1 = make_chunk(chunk_id="c1", doc_id="doc-A")
    c2 = make_chunk(chunk_id="c2", doc_id="doc-B")
    c3 = make_chunk(chunk_id="c3", doc_id="doc-C")
    memory_store.upsert_chunks(
        [c1, c2, c3],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )

    # Filter with list
    res_list = memory_store.search(
        query_vector=[1.0, 0.0, 0.0, 0.0],
        limit=5,
        payload_filter={"document_id": ["doc-A", "doc-B"]},
    )
    assert {h.chunk.id for h in res_list} == {"c1", "c2"}

    # Filter with tuple
    res_tuple = memory_store.search(
        query_vector=[1.0, 0.0, 0.0, 0.0],
        limit=5,
        payload_filter={"document_id": ("doc-A",)},
    )
    assert len(res_tuple) == 1
    assert res_tuple[0].chunk.id == "c1"


def test_search_payload_filter_field_scalar(memory_store: QdrantVectorStore) -> None:
    """Verify payload_filter with non-tags key as scalar."""
    c1 = make_chunk(chunk_id="c1", doc_id="doc-A")
    c2 = make_chunk(chunk_id="c2", doc_id="doc-B")
    memory_store.upsert_chunks([c1, c2], embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    results = memory_store.search(
        query_vector=[1.0, 0.0, 0.0, 0.0],
        limit=5,
        payload_filter={"document_id": "doc-A"},
    )
    assert len(results) == 1
    assert results[0].chunk.id == "c1"


def test_search_point_without_chunk_id_payload_fallback(memory_store: QdrantVectorStore) -> None:
    """Verify search handles point without payload or missing chunk_id by using point ID."""
    point_uuid = str(uuid.uuid4())
    memory_store.client.upsert(
        collection_name=memory_store.collection_name,
        points=[
            models.PointStruct(
                id=point_uuid,
                vector=[1.0, 0.0, 0.0, 0.0],
                payload=None,
            )
        ],
    )
    results = memory_store.search(query_vector=[1.0, 0.0, 0.0, 0.0], limit=1)
    assert len(results) == 1
    assert results[0].chunk.id == point_uuid
    assert results[0].chunk.document_id == ""


def test_search_tuples(memory_store: QdrantVectorStore) -> None:
    """Verify search_tuples returns ranked (chunk_id, score) pairs."""
    c1 = make_chunk(chunk_id="c1", tags=["search"])
    c2 = make_chunk(chunk_id="c2", tags=["other"])
    memory_store.upsert_chunks([c1, c2], embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    tuples = memory_store.search_tuples(
        query_vector=[1.0, 0.0, 0.0, 0.0],
        limit=5,
        tags=["search"],
    )
    assert len(tuples) == 1
    chunk_id, score = tuples[0]
    assert chunk_id == "c1"
    assert pytest.approx(score, rel=1e-3) == 1.0


# -----------------------------------------------------------------------------
# Deletion Tests
# -----------------------------------------------------------------------------


def test_delete_by_ids(memory_store: QdrantVectorStore) -> None:
    """Verify deleting specific chunks by chunk IDs."""
    c1 = make_chunk(chunk_id="c1")
    c2 = make_chunk(chunk_id="c2")
    c3 = make_chunk(chunk_id="c3")
    memory_store.upsert_chunks(
        [c1, c2, c3],
        embeddings=[[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]],
    )
    assert memory_store.count() == 3

    # Delete subset
    memory_store.delete_by_ids(["c1", "c2"])
    assert memory_store.count() == 1
    assert memory_store.get_all_point_ids() == {"c3"}

    # Deleting empty list returns early
    memory_store.delete_by_ids([])
    assert memory_store.count() == 1


def test_delete_by_document_id(memory_store: QdrantVectorStore) -> None:
    """Verify deleting all chunks belonging to a document."""
    c1 = make_chunk(chunk_id="c1", doc_id="doc-A")
    c2 = make_chunk(chunk_id="c2", doc_id="doc-A")
    c3 = make_chunk(chunk_id="c3", doc_id="doc-B")
    memory_store.upsert_chunks(
        [c1, c2, c3],
        embeddings=[[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]],
    )
    assert memory_store.count() == 3

    memory_store.delete_by_document_id("doc-A")
    assert memory_store.count() == 1
    assert memory_store.get_all_point_ids() == {"c3"}


def test_delete_by_chunk_hashes(memory_store: QdrantVectorStore) -> None:
    """Verify deleting points matching specific chunk hashes."""
    c1 = make_chunk(chunk_id="c1", chunk_hash="hash-1")
    c2 = make_chunk(chunk_id="c2", chunk_hash="hash-2")
    c3 = make_chunk(chunk_id="c3", chunk_hash="hash-3")
    memory_store.upsert_chunks(
        [c1, c2, c3],
        embeddings=[[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]],
    )
    assert memory_store.count() == 3

    memory_store.delete_by_chunk_hashes(["hash-1", "hash-3"])
    assert memory_store.count() == 1
    assert memory_store.get_all_point_ids() == {"c2"}

    # Deleting empty list returns early
    memory_store.delete_by_chunk_hashes([])
    assert memory_store.count() == 1


# -----------------------------------------------------------------------------
# Point IDs Scroll & Count Tests
# -----------------------------------------------------------------------------


def test_get_all_point_ids(memory_store: QdrantVectorStore) -> None:
    """Verify retrieving set of all point IDs."""
    chunks = [make_chunk(chunk_id=f"c{i}") for i in range(5)]
    vectors = [[0.1 * i, 0.2, 0.3, 0.4] for i in range(5)]
    memory_store.upsert_chunks(chunks, embeddings=vectors)

    all_ids = memory_store.get_all_point_ids()
    assert all_ids == {"c0", "c1", "c2", "c3", "c4"}


def test_get_all_point_ids_fallback_without_chunk_id(memory_store: QdrantVectorStore) -> None:
    """Verify get_all_point_ids falls back to record.id when chunk_id is not in payload."""
    raw_point_id = str(uuid.uuid4())
    memory_store.client.upsert(
        collection_name=memory_store.collection_name,
        points=[
            models.PointStruct(
                id=raw_point_id,
                vector=[0.1, 0.2, 0.3, 0.4],
                payload={"other_field": 123},
            )
        ],
    )
    ids = memory_store.get_all_point_ids()
    assert raw_point_id in ids


def test_get_all_point_ids_pagination(memory_store: QdrantVectorStore) -> None:
    """Verify scrolling across multiple batches when collection has > 100 points."""
    num_points = 105
    chunks = [make_chunk(chunk_id=f"page_c_{i}") for i in range(num_points)]
    vectors = [[0.1, 0.2, 0.3, 0.4] for _ in range(num_points)]
    memory_store.upsert_chunks(chunks, embeddings=vectors)

    assert memory_store.count() == num_points
    all_ids = memory_store.get_all_point_ids()
    assert len(all_ids) == num_points
    assert all_ids == {f"page_c_{i}" for i in range(num_points)}


def test_count(memory_store: QdrantVectorStore) -> None:
    """Verify count reflects vector insertions and deletions."""
    assert memory_store.count() == 0
    c1 = make_chunk(chunk_id="c1")
    memory_store.upsert_chunks([c1], embeddings=[[0.1, 0.2, 0.3, 0.4]])
    assert memory_store.count() == 1
    memory_store.delete_by_ids(["c1"])
    assert memory_store.count() == 0


# -----------------------------------------------------------------------------
# Connection Management & Context Manager Tests
# -----------------------------------------------------------------------------


def test_close_and_context_manager(test_settings: Settings) -> None:
    """Verify context manager enters and exits cleanly, invoking close()."""
    with QdrantVectorStore(settings=test_settings, location=":memory:") as store:
        assert isinstance(store, QdrantVectorStore)
        assert store.count() == 0

    # Idempotent close
    store.close()


def test_close_safe_without_client_or_close_method(test_settings: Settings) -> None:
    """Verify close() handles client lacking close attribute or deleted client."""
    store = QdrantVectorStore(settings=test_settings, location=":memory:")
    real_client = store.client

    # Replace with mock object lacking close method
    store.client = object()  # type: ignore[assignment]
    store.close()  # Should not raise

    # Delete client attribute completely
    del store.client
    store.close()  # Should not raise

    # Safely close the original client
    real_client.close()
