"""Unit tests for strata.indexing.bm25_index (Sparse lexical BM25 index adapter)."""

import math
from pathlib import Path

import pytest

from strata.core.config import Settings
from strata.core.models import Chunk, RetrievedChunk
from strata.core.protocols import SparseIndex
from strata.indexing.bm25_index import BM25Index


def make_chunk(
    chunk_id: str = "chunk-1",
    doc_id: str = "doc-1",
    index: int = 0,
    text: str = "Sample chunk text",
    heading_path: str = "Doc > Section",
    chunk_hash: str | None = None,
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
        parent_section_id=None,
        token_count=len(text.split()),
        tags=tags if tags is not None else ["search", "test"],
    )


class TestBM25IndexProtocol:
    """Validate protocol compliance and inheritance."""

    def test_inherits_sparse_index_protocol(self) -> None:
        """BM25Index must inherit directly from SparseIndex and pass runtime check."""
        index = BM25Index()
        assert SparseIndex in BM25Index.__mro__
        assert isinstance(index, SparseIndex)

    def test_protocol_methods_present(self) -> None:
        """Verify protocol methods are callable."""
        index = BM25Index()
        assert callable(getattr(index, "index_chunks", None))
        assert callable(getattr(index, "search", None))
        assert callable(getattr(index, "clear", None))
        assert hasattr(index, "count")


class TestBM25Tokenizer:
    """Validate text tokenization and regex behavior."""

    def test_tokenize_normalization_and_identifiers(self) -> None:
        """Verify lowercasing, word boundary matching, and identifier preservation."""
        index = BM25Index()
        text = "FastAPI handles HTTP_GET requests for user-profile at /api/v1."
        tokens = index.tokenize(text)

        expected = [
            "fastapi",
            "handles",
            "http_get",
            "requests",
            "for",
            "user-profile",
            "at",
            "api",
            "v1",
        ]
        assert tokens == expected

    def test_tokenize_empty_and_whitespace(self) -> None:
        """Empty string or whitespace returns empty token list safely."""
        index = BM25Index()
        assert index.tokenize("") == []
        assert index.tokenize("   \t\n  ") == []

    def test_tokenize_punctuation_only(self) -> None:
        """Punctuation without words or identifiers returns empty token list safely."""
        index = BM25Index()
        assert index.tokenize("!@#$%^&*()+={}[]:;<>?,./~`") == []

    def test_static_default_tokenize(self) -> None:
        """Verify default_tokenize static helper method."""
        tokens = BM25Index.default_tokenize("Hello WORLD")
        assert tokens == ["hello", "world"]


class TestBM25IndexingAndSearch:
    """Validate indexing and retrieval capabilities."""

    @pytest.fixture
    def sample_chunks(self) -> list[Chunk]:
        """Provide a diverse set of sample chunks for lexical retrieval."""
        return [
            make_chunk(
                chunk_id="c-py",
                doc_id="doc-lang",
                text="Python is a high-level interpreted programming language with dynamic semantics.",
                tags=["python", "programming"],
            ),
            make_chunk(
                chunk_id="c-rust",
                doc_id="doc-lang",
                text="Rust empowers everyone to build reliable and efficient systems software.",
                tags=["rust", "systems"],
            ),
            make_chunk(
                chunk_id="c-web",
                doc_id="doc-web",
                text="FastAPI is a modern, fast web framework for building APIs with Python standard typing.",
                tags=["python", "web"],
            ),
            make_chunk(
                chunk_id="c-db",
                doc_id="doc-db",
                text="SQLite is a C-language library that implements a small, fast, self-contained SQL engine.",
                tags=["sql", "storage"],
            ),
        ]

    def test_index_and_exact_search(self, sample_chunks: list[Chunk]) -> None:
        """Test indexing dummy chunks and searching with exact matching terms."""
        index = BM25Index()
        index.index_chunks(sample_chunks)

        assert index.count == 4
        assert len(index) == 4
        assert "c-py" in index

        results = index.search("python", limit=5)
        assert len(results) == 2

        # Both Python chunks should match
        retrieved_ids = [r.chunk.id for r in results]
        assert "c-py" in retrieved_ids
        assert "c-web" in retrieved_ids

        for rank, res in enumerate(results, start=1):
            assert isinstance(res, RetrievedChunk)
            assert res.score > 0.0
            assert res.sparse_score == res.score
            assert res.sparse_rank == rank
            assert res.chunk.text != ""

    def test_term_frequency_ranking(self) -> None:
        """Verify ranking: chunks containing multiple query term occurrences score higher."""
        chunk_high = make_chunk(
            chunk_id="c-high",
            text="Python Python Python is the best language for data science and machine learning.",
        )
        chunk_single = make_chunk(
            chunk_id="c-single",
            text="Python is a dynamic programming language.",
        )
        chunk_other1 = make_chunk(
            chunk_id="c-other1",
            text="Rust is designed for performance and reliability in systems engineering.",
        )
        chunk_other2 = make_chunk(
            chunk_id="c-other2",
            text="Go is an open-source programming language supported by Google.",
        )

        index = BM25Index()
        index.index_chunks([chunk_high, chunk_single, chunk_other1, chunk_other2])

        results = index.search("Python", limit=5)
        assert len(results) == 2
        assert results[0].chunk.id == "c-high"
        assert results[1].chunk.id == "c-single"
        assert results[0].score > results[1].score

    def test_search_nonexistent_term_returns_empty(self, sample_chunks: list[Chunk]) -> None:
        """Search with nonexistent terms must return an empty result list."""
        index = BM25Index()
        index.index_chunks(sample_chunks)

        results = index.search("supercalifragilistic", limit=10)
        assert results == []

        results_multi = index.search("nonexistent_term another_missing_term", limit=10)
        assert results_multi == []

    def test_search_limit(self, sample_chunks: list[Chunk]) -> None:
        """Search limit parameter strictly restricts returned results count."""
        index = BM25Index()
        index.index_chunks(sample_chunks)

        results = index.search("programming", limit=1)
        assert len(results) == 1

    def test_search_with_tag_filter(self, sample_chunks: list[Chunk]) -> None:
        """Search respects payload_filter tag filtering when provided."""
        index = BM25Index()
        index.index_chunks(sample_chunks)

        # "python" query matches c-py and c-web; filtering for "web" should only return c-web
        results = index.search("python", limit=5, payload_filter={"tags": ["web"]})
        assert len(results) == 1
        assert results[0].chunk.id == "c-web"


class TestBM25Deletion:
    """Validate differential deletion of chunks."""

    @pytest.fixture
    def populated_index(self) -> BM25Index:
        """Provide index pre-populated with test chunks."""
        chunks = [
            make_chunk(chunk_id="c-1", doc_id="d-1", chunk_hash="hash-1", text="Alpha query term"),
            make_chunk(chunk_id="c-2", doc_id="d-1", chunk_hash="hash-2", text="Beta query term"),
            make_chunk(chunk_id="c-3", doc_id="d-2", chunk_hash="hash-3", text="Gamma query term"),
        ]
        return BM25Index(chunks=chunks)

    def test_delete_chunks_by_id(self, populated_index: BM25Index) -> None:
        """Deleting chunk by ID removes it from count, internal maps, and search."""
        assert populated_index.count == 3
        populated_index.delete_chunks(["c-1"])

        assert populated_index.count == 2
        assert "c-1" not in populated_index
        assert populated_index.get_chunk("c-1") is None
        assert populated_index.search("Alpha", limit=5) == []

        # Remaining chunks are still searchable
        results = populated_index.search("Beta", limit=5)
        assert len(results) == 1
        assert results[0].chunk.id == "c-2"

    def test_delete_by_ids_alias(self, populated_index: BM25Index) -> None:
        """Verify delete_by_ids and delete_chunk aliases."""
        populated_index.delete_chunk("c-2")
        assert populated_index.count == 2
        assert populated_index.search("Beta", limit=5) == []

        populated_index.delete_by_ids(["c-3"])
        assert populated_index.count == 1
        assert populated_index.search("Gamma", limit=5) == []

    def test_delete_by_document_id(self, populated_index: BM25Index) -> None:
        """Delete all chunks belonging to a document ID."""
        populated_index.delete_by_document_id("d-1")
        assert populated_index.count == 1
        assert populated_index.get_chunk("c-1") is None
        assert populated_index.get_chunk("c-2") is None
        assert populated_index.get_chunk("c-3") is not None

    def test_delete_by_chunk_hashes(self, populated_index: BM25Index) -> None:
        """Delete chunks matching cryptographic SHA-256 hashes."""
        populated_index.delete_by_chunk_hashes(["hash-2", "hash-3"])
        assert populated_index.count == 1
        assert populated_index.get_chunk("c-1") is not None
        assert populated_index.get_chunk("c-2") is None
        assert populated_index.get_chunk("c-3") is None

    def test_delete_nonexistent_id_is_noop(self, populated_index: BM25Index) -> None:
        """Deleting a non-existent ID does not mutate index or throw error."""
        populated_index.delete_chunks(["nonexistent-id"])
        assert populated_index.count == 3


class TestBM25Serialization:
    """Validate saving and restoring index state."""

    @pytest.fixture
    def test_chunks(self) -> list[Chunk]:
        """Diverse chunks for serialization verification."""
        return [
            make_chunk(
                chunk_id="c-doc1",
                doc_id="doc-1",
                text="Air-gapped offline personal knowledge engine with local hybrid RAG.",
                tags=["offline", "strata"],
            ),
            make_chunk(
                chunk_id="c-doc2",
                doc_id="doc-2",
                text="BM25 sparse lexical ranking provides high precision for identifiers and codes.",
                tags=["bm25", "search"],
            ),
            make_chunk(
                chunk_id="c-doc3",
                doc_id="doc-3",
                text="Qdrant dense vector index handles semantic conceptual similarity search.",
                tags=["qdrant", "vector"],
            ),
        ]

    def test_save_and_load_json(self, tmp_path: Path, test_chunks: list[Chunk]) -> None:
        """Saving to a tmp_path file and loading back reproduces identical search rankings and scores."""
        index = BM25Index(chunks=test_chunks)
        save_file = tmp_path / "bm25_index.json"

        index.save(save_file)
        assert save_file.exists()

        # Load back into a new instance using classmethod
        loaded_index = BM25Index.load(save_file)

        assert loaded_index.count == index.count
        assert loaded_index.idx_to_chunk_id == index.idx_to_chunk_id
        assert loaded_index.chunk_id_to_idx == index.chunk_id_to_idx

        # Compare search results across multiple queries
        test_queries = ["offline knowledge", "bm25 lexical", "vector similarity", "nonexistent"]
        for query in test_queries:
            orig_results = index.search(query, limit=5)
            loaded_results = loaded_index.search(query, limit=5)

            assert len(orig_results) == len(loaded_results)
            for orig, loaded in zip(orig_results, loaded_results, strict=True):
                assert orig.chunk.id == loaded.chunk.id
                assert orig.sparse_rank == loaded.sparse_rank
                assert math.isclose(orig.score, loaded.score, rel_tol=1e-5)

    def test_load_from_path_instance_method(self, tmp_path: Path, test_chunks: list[Chunk]) -> None:
        """Verify loading into an existing instance via load_from_path."""
        index = BM25Index(chunks=test_chunks)
        save_file = tmp_path / "index_dump.json"
        index.save(save_file)

        target = BM25Index()
        assert target.count == 0
        target.load_from_path(save_file)

        assert target.count == 3
        res = target.search("offline", limit=5)
        assert len(res) == 1
        assert res[0].chunk.id == "c-doc1"

    def test_save_and_load_pickle(self, tmp_path: Path, test_chunks: list[Chunk]) -> None:
        """Verify binary pickle format save and load."""
        index = BM25Index(chunks=test_chunks)
        save_file = tmp_path / "index_dump.pkl"

        index.save(save_file)
        loaded = BM25Index.load(save_file)

        assert loaded.count == index.count
        res = loaded.search("bm25", limit=5)
        assert len(res) == 1
        assert res[0].chunk.id == "c-doc2"

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Loading a non-existent file path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            BM25Index.load(tmp_path / "nonexistent.json")


class TestBM25EdgeCases:
    """Validate boundary conditions and edge cases."""

    def test_empty_index_search(self) -> None:
        """Search on an unpopulated index returns empty list."""
        index = BM25Index()
        assert index.count == 0
        assert index.bm25 is None
        assert index.corpus == []
        assert index.search("anything", limit=10) == []
        assert index.search("", limit=10) == []

    def test_empty_query_on_populated_index(self) -> None:
        """Empty query on populated index returns empty list."""
        index = BM25Index()
        index.index_chunks([make_chunk(text="Some valid content")])

        assert index.search("", limit=5) == []
        assert index.search("    ", limit=5) == []
        assert index.search("!@#$%", limit=5) == []

    def test_limit_zero_or_negative(self) -> None:
        """Non-positive search limit returns empty list."""
        index = BM25Index()
        index.index_chunks([make_chunk(text="Some valid content")])

        assert index.search("content", limit=0) == []
        assert index.search("content", limit=-1) == []

    def test_empty_chunk_text_handling(self) -> None:
        """Chunks with empty text do not raise ZeroDivisionError and return empty search."""
        chunk_empty1 = make_chunk(chunk_id="c-e1", text="")
        chunk_empty2 = make_chunk(chunk_id="c-e2", text="   ")
        chunk_empty3 = make_chunk(chunk_id="c-e3", text="!@#$%")

        index = BM25Index()
        index.index_chunks([chunk_empty1, chunk_empty2, chunk_empty3])

        assert index.count == 3
        # No tokens exist in any document, BM25Okapi safely bypassed
        assert index.bm25 is None
        assert index.search("anything", limit=5) == []

    def test_clear_index(self) -> None:
        """Clearing the index resets all internal mappings and BM25 model."""
        index = BM25Index()
        index.index_chunks([make_chunk(text="Sample text")])
        assert index.count == 1
        assert index.bm25 is not None

        index.clear()
        assert index.count == 0
        assert index.bm25 is None
        assert index.corpus == []
        assert index.idx_to_chunk_id == []
        assert index.chunk_id_to_idx == {}
        assert index.search("Sample", limit=5) == []

    def test_reindex_existing_chunk_updates_in_place(self) -> None:
        """Re-indexing a chunk with an existing ID updates it without duplication."""
        c1 = make_chunk(chunk_id="c-1", text="Original version text about bananas.")
        index = BM25Index(chunks=[c1])
        assert index.count == 1
        assert len(index.search("bananas", limit=5)) == 1

        c1_updated = make_chunk(chunk_id="c-1", text="Updated version text about strawberries.")
        index.index_chunks([c1_updated])

        assert index.count == 1
        assert len(index.search("strawberries", limit=5)) == 1
        assert len(index.search("bananas", limit=5)) == 0

    def test_get_chunk_and_get_all_chunks(self) -> None:
        """Verify get_chunk and get_all_chunks helpers."""
        c1 = make_chunk(chunk_id="c-1", text="First chunk")
        c2 = make_chunk(chunk_id="c-2", text="Second chunk")
        index = BM25Index(chunks=[c1, c2])

        assert index.get_chunk("c-1") == c1
        assert index.get_chunk("nonexistent") is None

        all_chunks = index.get_all_chunks()
        assert len(all_chunks) == 2
        assert {c.id for c in all_chunks} == {"c-1", "c-2"}

    def test_custom_parameters_initialization(self) -> None:
        """Verify initializing with custom k1, b, epsilon, and settings."""
        settings = Settings(bm25_k1=1.8, bm25_b=0.8)
        index = BM25Index(settings=settings, k1=2.0, b=0.5, epsilon=0.1)

        assert index.k1 == 2.0
        assert index.b == 0.5
        assert index.epsilon == 0.1
