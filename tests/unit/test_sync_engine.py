"""Unit tests for strata.indexing.sync_engine (Differential Sync Engine)."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from strata.core.config import Settings
from strata.core.exceptions import SyncConflictError
from strata.core.models import Chunk, Document, DocumentOutline, RetrievedChunk, Section
from strata.core.protocols import DocumentStore, Embedder, SparseIndex, VectorStore
from strata.indexing.bm25_index import BM25Index
from strata.indexing.sync_engine import IndexSyncEngine
from strata.ingestion.chunker import MarkdownChunker
from strata.ingestion.reader import LocalFileReader
from strata.storage.sqlite_store import SQLiteStore


def make_chunk(
    chunk_id: str = "chunk-1",
    doc_id: str = "doc-1",
    index: int = 0,
    text: str = "Sample chunk text",
    heading_path: str = "Doc > Intro",
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


class MockDocumentStore(DocumentStore):
    """In-memory DocumentStore mock tracking chunks and documents."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}

    def add_document(self, document: Document) -> None:
        self.documents[document.id] = document

    def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    def get_document_by_path(self, file_path: str) -> Document | None:
        for doc in self.documents.values():
            if doc.file_path == file_path:
                return doc
        return None

    def list_documents(self, tags: list[str] | None = None) -> list[Document]:
        return list(self.documents.values())

    def update_document(self, document: Document) -> None:
        self.documents[document.id] = document

    def delete_document(self, document_id: str) -> None:
        self.documents.pop(document_id, None)
        self.chunks = {cid: c for cid, c in self.chunks.items() if c.document_id != document_id}

    def add_sections(self, sections: list[Section]) -> None:
        pass

    def get_section(self, section_id: str) -> Section | None:
        return None

    def get_sections_by_document_id(self, document_id: str) -> list[Section]:
        return []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self.chunks[c.id] = c

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def get_chunks_by_document_id(self, document_id: str) -> list[Chunk]:
        return [c for c in self.chunks.values() if c.document_id == document_id]

    def get_all_chunks(self) -> list[Chunk]:
        return list(self.chunks.values())

    def get_active_chunk_hashes(self, document_id: str) -> list[str]:
        return [c.chunk_hash for c in self.chunks.values() if c.document_id == document_id]

    def delete_chunks_by_hashes(self, chunk_hashes: list[str]) -> None:
        hashes = set(chunk_hashes)
        self.chunks = {cid: c for cid, c in self.chunks.items() if c.chunk_hash not in hashes}

    def delete_chunk_by_id(self, chunk_id: str) -> None:
        """Helper to simulate chunk deletion."""
        self.chunks.pop(chunk_id, None)

    def get_document_outline(self, document_id: str) -> DocumentOutline:
        return DocumentOutline(document_id=document_id, document_title="", items=[])

    def get_adjacent_chunks(self, chunk_id: str, window: int = 1) -> list[Chunk]:
        return []

    def count_documents(self) -> int:
        return len(self.documents)

    def count_chunks(self) -> int:
        return len(self.chunks)


class MockVectorStore(VectorStore):
    """In-memory VectorStore mock recording upserts and point IDs."""

    def __init__(self) -> None:
        self.points: dict[str, tuple[Chunk, list[float]]] = {}
        self.upsert_batches: list[list[Chunk]] = []
        self.deleted_ids: list[str] = []

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.upsert_batches.append(chunks)
        for chunk, vec in zip(chunks, embeddings, strict=True):
            self.points[chunk.id] = (chunk, vec)

    def get_all_point_ids(self) -> set[str]:
        return set(self.points.keys())

    def delete_by_ids(self, chunk_ids: list[str]) -> None:
        self.deleted_ids.extend(chunk_ids)
        for cid in chunk_ids:
            self.points.pop(cid, None)

    def delete_by_document_id(self, document_id: str) -> None:
        to_del = [cid for cid, (c, _) in self.points.items() if c.document_id == document_id]
        self.delete_by_ids(to_del)

    def delete_by_chunk_hashes(self, chunk_hashes: list[str]) -> None:
        hashes = set(chunk_hashes)
        to_del = [cid for cid, (c, _) in self.points.items() if c.chunk_hash in hashes]
        self.delete_by_ids(to_del)

    def search(
        self,
        query_vector: list[float],
        limit: int,
        payload_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        return []

    def count(self) -> int:
        return len(self.points)


class MockSparseIndex(SparseIndex):
    """In-memory SparseIndex mock recording indexed chunks and deletions."""

    def __init__(self) -> None:
        self.indexed_chunks: dict[str, Chunk] = {}
        self.deleted_ids: list[str] = []
        self.saved_paths: list[Path] = []

    def index_chunks(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self.indexed_chunks[c.id] = c

    def delete_by_ids(self, chunk_ids: list[str]) -> None:
        self.deleted_ids.extend(chunk_ids)
        for cid in chunk_ids:
            self.indexed_chunks.pop(cid, None)

    def search(self, query: str, limit: int) -> list[RetrievedChunk]:
        return []

    def clear(self) -> None:
        self.indexed_chunks.clear()

    @property
    def count(self) -> int:
        return len(self.indexed_chunks)

    def save(self, file_path: Path | str) -> None:
        self.saved_paths.append(Path(file_path))


class MockEmbedder(Embedder):
    """In-memory Embedder mock recording inference calls."""

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension
        self.embed_calls: list[list[str]] = []
        self.should_fail: bool = False

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.should_fail:
            raise RuntimeError("Simulated inference out of memory error")
        self.embed_calls.append(texts)
        return [[0.1] * self._dim for _ in texts]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * self._dim

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def total_texts_embedded(self) -> int:
        return sum(len(call) for call in self.embed_calls)


class TestIndexSyncEngine:
    """Test suite for IndexSyncEngine differential synchronization."""

    @pytest.fixture
    def setup_env(
        self,
        tmp_path: Path,
    ) -> tuple[MockDocumentStore, MockVectorStore, MockSparseIndex, MockEmbedder, IndexSyncEngine]:
        """Provide a clean in-memory harness for IndexSyncEngine."""
        doc_store = MockDocumentStore()
        vec_store = MockVectorStore()
        sparse_idx = MockSparseIndex()
        embedder = MockEmbedder()
        settings = Settings(bm25_index_path=tmp_path / "bm25.json")
        engine = IndexSyncEngine(
            sqlite_store=doc_store,
            vector_store=vec_store,
            sparse_index=sparse_idx,
            embedder=embedder,
            settings=settings,
        )
        return doc_store, vec_store, sparse_idx, embedder, engine

    def test_initial_full_sync(
        self,
        setup_env: tuple[
            MockDocumentStore, MockVectorStore, MockSparseIndex, MockEmbedder, IndexSyncEngine
        ],
    ) -> None:
        """Initial sync embeds and indexes all chunks from canonical storage."""
        doc_store, vec_store, sparse_idx, embedder, engine = setup_env

        # Populate SQLite with 3 chunks across 2 documents
        doc_store.add_document(
            Document(id="d1", title="Doc 1", file_path="d1.md", raw_content="text", doc_hash="h1")
        )
        doc_store.add_document(
            Document(id="d2", title="Doc 2", file_path="d2.md", raw_content="text", doc_hash="h2")
        )
        chunks = [
            make_chunk(chunk_id="c1", doc_id="d1", text="Chunk one text"),
            make_chunk(chunk_id="c2", doc_id="d1", text="Chunk two text"),
            make_chunk(chunk_id="c3", doc_id="d2", text="Chunk three text"),
        ]
        doc_store.add_chunks(chunks)

        report = engine.sync(batch_size=2)

        assert report.total_documents == 2
        assert report.total_chunks == 3
        assert report.new_chunks == 3
        assert report.unchanged_chunks == 0
        assert report.orphaned_chunks == 0
        assert report.elapsed_ms >= 0.0

        # Vector store & sparse index should have all 3 chunks
        assert vec_store.get_all_point_ids() == {"c1", "c2", "c3"}
        assert set(sparse_idx.indexed_chunks.keys()) == {"c1", "c2", "c3"}
        assert embedder.total_texts_embedded == 3

    def test_noop_sync_when_nothing_changed(
        self,
        setup_env: tuple[
            MockDocumentStore, MockVectorStore, MockSparseIndex, MockEmbedder, IndexSyncEngine
        ],
    ) -> None:
        """A second sync without changes generates 0 new embeddings."""
        doc_store, vec_store, _sparse_idx, embedder, engine = setup_env

        doc_store.add_document(
            Document(id="d1", title="Doc 1", file_path="d1.md", raw_content="text", doc_hash="h1")
        )
        doc_store.add_chunks([make_chunk(chunk_id="c1", doc_id="d1")])

        # First sync
        first_report = engine.sync()
        assert first_report.new_chunks == 1
        initial_embed_count = embedder.total_texts_embedded

        # Second sync
        second_report = engine.sync()
        assert second_report.total_chunks == 1
        assert second_report.new_chunks == 0
        assert second_report.unchanged_chunks == 1
        assert second_report.orphaned_chunks == 0

        # Zero new embeddings should have been computed
        assert embedder.total_texts_embedded == initial_embed_count
        assert len(vec_store.upsert_batches) == 1

    def test_differential_additions(
        self,
        setup_env: tuple[
            MockDocumentStore, MockVectorStore, MockSparseIndex, MockEmbedder, IndexSyncEngine
        ],
    ) -> None:
        """Adding one chunk only computes embeddings for that single addition."""
        doc_store, vec_store, sparse_idx, embedder, engine = setup_env

        doc_store.add_document(
            Document(id="d1", title="Doc 1", file_path="d1.md", raw_content="text", doc_hash="h1")
        )
        doc_store.add_chunks(
            [
                make_chunk(chunk_id="c1", doc_id="d1", text="Initial chunk 1"),
                make_chunk(chunk_id="c2", doc_id="d1", text="Initial chunk 2"),
            ]
        )

        engine.sync()
        assert embedder.total_texts_embedded == 2

        # Add 1 new chunk to SQLite
        doc_store.add_chunks([make_chunk(chunk_id="c3", doc_id="d1", text="Newly added chunk 3")])

        diff_report = engine.sync()
        assert diff_report.total_chunks == 3
        assert diff_report.new_chunks == 1
        assert diff_report.unchanged_chunks == 2
        assert diff_report.orphaned_chunks == 0

        # Total embedded should now be 3 (only 1 additional embedding computed)
        assert embedder.total_texts_embedded == 3
        assert vec_store.get_all_point_ids() == {"c1", "c2", "c3"}
        assert "c3" in sparse_idx.indexed_chunks

    def test_differential_deletions(
        self,
        setup_env: tuple[
            MockDocumentStore, MockVectorStore, MockSparseIndex, MockEmbedder, IndexSyncEngine
        ],
    ) -> None:
        """Deleting a chunk from SQLite triggers deletion in vector store and sparse index."""
        doc_store, vec_store, sparse_idx, _embedder, engine = setup_env

        doc_store.add_document(
            Document(id="d1", title="Doc 1", file_path="d1.md", raw_content="text", doc_hash="h1")
        )
        doc_store.add_chunks(
            [
                make_chunk(chunk_id="c1", doc_id="d1"),
                make_chunk(chunk_id="c2", doc_id="d1"),
            ]
        )

        engine.sync()
        assert vec_store.get_all_point_ids() == {"c1", "c2"}

        # Delete c1 from SQLite
        doc_store.delete_chunk_by_id("c1")

        purge_report = engine.sync()
        assert purge_report.total_chunks == 1
        assert purge_report.new_chunks == 0
        assert purge_report.unchanged_chunks == 1
        assert purge_report.orphaned_chunks == 1

        assert vec_store.get_all_point_ids() == {"c2"}
        assert "c1" in vec_store.deleted_ids
        assert "c1" in sparse_idx.deleted_ids
        assert "c1" not in sparse_idx.indexed_chunks

    def test_batching_logic_respects_batch_size(
        self,
        setup_env: tuple[
            MockDocumentStore, MockVectorStore, MockSparseIndex, MockEmbedder, IndexSyncEngine
        ],
    ) -> None:
        """Embedding and upserts must process chunks according to batch_size."""
        doc_store, vec_store, _, embedder, engine = setup_env

        doc_store.add_document(
            Document(id="d1", title="Doc 1", file_path="d1.md", raw_content="text", doc_hash="h1")
        )
        chunks = [make_chunk(chunk_id=f"c{i}", doc_id="d1", text=f"Text {i}") for i in range(5)]
        doc_store.add_chunks(chunks)

        report = engine.sync(batch_size=2)
        assert report.new_chunks == 5

        # 5 items with batch_size=2 should yield batches of sizes: 2, 2, 1
        upsert_batch_sizes = [len(b) for b in vec_store.upsert_batches]
        assert upsert_batch_sizes == [2, 2, 1]

        embed_batch_sizes = [len(call) for call in embedder.embed_calls]
        assert embed_batch_sizes == [2, 2, 1]

    def test_sync_uses_config_default_batch_size(
        self,
        setup_env: tuple[
            MockDocumentStore, MockVectorStore, MockSparseIndex, MockEmbedder, IndexSyncEngine
        ],
    ) -> None:
        """Verify sync() without batch_size defaults to settings.embedding_batch_size."""
        doc_store, vec_store, sparse_idx, embedder, _ = setup_env
        custom_settings = Settings(embedding_batch_size=3)
        engine = IndexSyncEngine(
            sqlite_store=doc_store,
            vector_store=vec_store,
            sparse_index=sparse_idx,
            embedder=embedder,
            settings=custom_settings,
        )
        assert engine.batch_size == 3

        doc_store.add_document(
            Document(id="d1", title="Doc 1", file_path="d1.md", raw_content="text", doc_hash="h1")
        )
        chunks = [make_chunk(chunk_id=f"c{i}", doc_id="d1", text=f"Text {i}") for i in range(7)]
        doc_store.add_chunks(chunks)

        report = engine.sync()
        assert report.new_chunks == 7

        # 7 items with batch_size=3 should yield batches of sizes: 3, 3, 1
        upsert_batch_sizes = [len(b) for b in vec_store.upsert_batches]
        assert upsert_batch_sizes == [3, 3, 1]

        embed_batch_sizes = [len(call) for call in embedder.embed_calls]
        assert embed_batch_sizes == [3, 3, 1]

    def test_sparse_index_persistence_when_configured(
        self,
        tmp_path: Path,
        setup_env: tuple[
            MockDocumentStore, MockVectorStore, MockSparseIndex, MockEmbedder, IndexSyncEngine
        ],
    ) -> None:
        """Sparse index is persisted to settings.bm25_index_path if configured."""
        doc_store, _, sparse_idx, _, _ = setup_env
        target_path = tmp_path / "bm25.json"

        settings = Settings(bm25_index_path=target_path)
        engine = IndexSyncEngine(
            sqlite_store=doc_store,
            vector_store=setup_env[1],
            sparse_index=sparse_idx,
            embedder=setup_env[3],
            settings=settings,
        )

        doc_store.add_chunks([make_chunk(chunk_id="c1")])
        engine.sync()

        assert target_path in sparse_idx.saved_paths

    def test_batch_failure_raises_sync_conflict_error(
        self,
        setup_env: tuple[
            MockDocumentStore, MockVectorStore, MockSparseIndex, MockEmbedder, IndexSyncEngine
        ],
    ) -> None:
        """Runtime failures during batch processing are shielded as SyncConflictError."""
        doc_store, _, _, embedder, engine = setup_env

        doc_store.add_chunks([make_chunk(chunk_id="c1"), make_chunk(chunk_id="c2")])
        embedder.should_fail = True

        with pytest.raises(SyncConflictError) as exc_info:
            engine.sync(batch_size=1)

        assert "Failed to synchronize batch" in str(exc_info.value)

    def test_compute_chunk_diff(
        self,
        setup_env: tuple[
            MockDocumentStore, MockVectorStore, MockSparseIndex, MockEmbedder, IndexSyncEngine
        ],
    ) -> None:
        """Verify chunk diff computation against document active hashes."""
        doc_store, _, _, _, engine = setup_env

        c1 = make_chunk(chunk_id="c1", doc_id="d1", chunk_hash="hash-1")
        c2 = make_chunk(chunk_id="c2", doc_id="d1", chunk_hash="hash-2")
        doc_store.add_chunks([c1, c2])

        c2_same = make_chunk(chunk_id="c2", doc_id="d1", chunk_hash="hash-2")
        c3_new = make_chunk(chunk_id="c3", doc_id="d1", chunk_hash="hash-3")

        diff = engine.compute_chunk_diff("d1", [c2_same, c3_new])
        assert diff.document_id == "d1"
        assert [c.chunk_hash for c in diff.new_chunks] == ["hash-3"]
        assert diff.unchanged_hashes == ["hash-2"]
        assert diff.orphaned_hashes == ["hash-1"]

    def test_empty_store_sync(
        self,
        setup_env: tuple[
            MockDocumentStore, MockVectorStore, MockSparseIndex, MockEmbedder, IndexSyncEngine
        ],
    ) -> None:
        """Sync on an empty store completes successfully with zero counts."""
        _, _, _, _, engine = setup_env
        report = engine.sync()

        assert report.total_documents == 0
        assert report.total_chunks == 0
        assert report.new_chunks == 0
        assert report.unchanged_chunks == 0
        assert report.orphaned_chunks == 0
        assert report.elapsed_ms >= 0.0

    def test_end_to_end_with_real_sqlite_and_bm25(self, tmp_path: Path) -> None:
        """Verify synchronization using real SQLite store and BM25Index."""
        test_settings = Settings(bm25_index_path=tmp_path / "bm25.json")
        with SQLiteStore(db_path=":memory:") as sqlite_store:
            bm25_idx = BM25Index(settings=test_settings)
            vec_store = MockVectorStore()
            embedder = MockEmbedder()

            engine = IndexSyncEngine(
                sqlite_store=sqlite_store,
                vector_store=vec_store,
                sparse_index=bm25_idx,
                embedder=embedder,
                settings=test_settings,
            )

            # Insert document and chunks into real SQLite ACID store
            doc = Document(
                id="doc-real",
                title="Python Architecture",
                file_path="/real/path.md",
                raw_content="Python content",
                doc_hash="doc-hash-real",
                tags=["python", "core"],
            )
            sqlite_store.add_document(doc)

            chunk1 = make_chunk(
                chunk_id="chunk-1",
                doc_id="doc-real",
                text="Python is an elegant interpreted programming language.",
                tags=["python"],
            )
            chunk2 = make_chunk(
                chunk_id="chunk-2",
                doc_id="doc-real",
                text="Rust provides memory safety without garbage collection.",
                tags=["rust"],
            )
            sqlite_store.add_chunks([chunk1, chunk2])

            report = engine.sync(batch_size=1)
            assert report.total_documents == 1
            assert report.total_chunks == 2
            assert report.new_chunks == 2
            assert report.unchanged_chunks == 0

            # Verify BM25 is now searchable
            res_python = bm25_idx.search("Python", limit=5)
            assert len(res_python) == 1
            assert res_python[0].chunk.id == "chunk-1"

            res_rust = bm25_idx.search("Rust", limit=5)
            assert len(res_rust) == 1
            assert res_rust[0].chunk.id == "chunk-2"

            # Sync second time -> 0 new chunks
            report2 = engine.sync()
            assert report2.new_chunks == 0
            assert report2.unchanged_chunks == 2

    def test_reingest_updated_document_preserves_unchanged_chunks(
        self,
        tmp_path: Path,
    ) -> None:
        """Regression test: re-ingesting an updated file must only re-index new/modified chunks."""
        db_path = tmp_path / "regression_canonical.db"
        bm25_path = tmp_path / "regression_bm25.json"
        sqlite_store = SQLiteStore(db_path=db_path)
        vec_store = MockVectorStore()
        sparse_idx = BM25Index(settings=Settings(bm25_index_path=bm25_path))
        embedder = MockEmbedder()
        engine = IndexSyncEngine(
            sqlite_store=sqlite_store,
            vector_store=vec_store,
            sparse_index=sparse_idx,
            embedder=embedder,
            settings=Settings(bm25_index_path=bm25_path),
        )

        reader = LocalFileReader()
        chunker = MarkdownChunker(min_chunk_size=5)

        # 1. Initial document with 3 distinct sections
        doc_path = tmp_path / "README.md"
        doc_path.write_text(
            "# Project Overview\n"
            "Strata is an offline air-gapped personal knowledge base engine.\n\n"
            "## Storage Engine\n"
            "SQLite serves as the canonical ACID relational store for raw text and chunks.\n\n"
            "## Vector Index\n"
            "Qdrant provides dense vector search for semantic ANN retrieval.\n",
            encoding="utf-8",
        )

        # Ingest initial version into SQLite
        doc = reader.read_file(doc_path)
        sections = chunker.extract_sections(doc)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 3

        sqlite_store.add_document(doc)
        sqlite_store.add_sections(sections)
        sqlite_store.add_chunks(chunks)

        # Initial sync: all 3 chunks embedded and indexed
        report1 = engine.sync()
        assert report1.total_documents == 1
        assert report1.total_chunks == 3
        assert report1.new_chunks == 3
        assert report1.unchanged_chunks == 0
        assert report1.orphaned_chunks == 0
        assert embedder.total_texts_embedded == 3
        assert vec_store.count() == 3
        assert sparse_idx.count == 3

        # 2. Append a 4th section (the exact problem statement scenario)
        doc_path.write_text(
            "# Project Overview\n"
            "Strata is an offline air-gapped personal knowledge base engine.\n\n"
            "## Storage Engine\n"
            "SQLite serves as the canonical ACID relational store for raw text and chunks.\n\n"
            "## Vector Index\n"
            "Qdrant provides dense vector search for semantic ANN retrieval.\n\n"
            "## Cache Layer\n"
            "In-memory caching accelerates recurring query resolution without inference.\n",
            encoding="utf-8",
        )

        # Re-ingest updated file (mimics strata add / ingest)
        updated_doc = reader.read_file(doc_path)
        new_sections = chunker.extract_sections(updated_doc)
        new_chunks = chunker.chunk_document(updated_doc)
        assert len(new_chunks) == 4

        # Reconcile in SQLite as done in main.py
        existing = sqlite_store.get_document_by_path(updated_doc.file_path)
        assert existing is not None
        sqlite_store.delete_document(existing.id)
        sqlite_store.add_document(updated_doc)
        sqlite_store.add_sections(new_sections)
        sqlite_store.add_chunks(new_chunks)

        # Sync after append: should detect 3 unchanged chunks and 1 added chunk
        report2 = engine.sync()
        assert report2.total_documents == 1
        assert report2.total_chunks == 4
        assert report2.new_chunks == 1
        assert report2.unchanged_chunks == 3
        assert report2.orphaned_chunks == 0
        # Critical assertion: total embedded texts must only increase by 1 (no re-embedding of existing 3)
        assert embedder.total_texts_embedded == 4
        assert vec_store.count() == 4
        assert sparse_idx.count == 4

        # 3. Insert a section at the beginning (shifts heading indices)
        doc_path.write_text(
            "# Introduction\n"
            "Welcome to Strata, an air-gapped hybrid retrieval engine.\n\n"
            "# Project Overview\n"
            "Strata is an offline air-gapped personal knowledge base engine.\n\n"
            "## Storage Engine\n"
            "SQLite serves as the canonical ACID relational store for raw text and chunks.\n\n"
            "## Vector Index\n"
            "Qdrant provides dense vector search for semantic ANN retrieval.\n\n"
            "## Cache Layer\n"
            "In-memory caching accelerates recurring query resolution without inference.\n",
            encoding="utf-8",
        )

        updated_doc2 = reader.read_file(doc_path)
        new_sections2 = chunker.extract_sections(updated_doc2)
        new_chunks2 = chunker.chunk_document(updated_doc2)
        assert len(new_chunks2) == 5

        existing2 = sqlite_store.get_document_by_path(updated_doc2.file_path)
        assert existing2 is not None
        sqlite_store.delete_document(existing2.id)
        sqlite_store.add_document(updated_doc2)
        sqlite_store.add_sections(new_sections2)
        sqlite_store.add_chunks(new_chunks2)

        # Sync after prepend: unchanged chunks preserve identity despite index shift
        report3 = engine.sync()
        assert report3.total_documents == 1
        assert report3.total_chunks == 5
        assert report3.new_chunks == 1
        assert report3.unchanged_chunks == 4
        assert report3.orphaned_chunks == 0
        assert embedder.total_texts_embedded == 5

        # 4. Modify an existing section in the middle
        doc_path.write_text(
            "# Introduction\n"
            "Welcome to Strata, an air-gapped hybrid retrieval engine.\n\n"
            "# Project Overview\n"
            "Strata is an offline air-gapped personal knowledge base engine.\n\n"
            "## Storage Engine\n"
            "SQLite is enhanced with WAL mode for concurrent ACID operations.\n\n"
            "## Vector Index\n"
            "Qdrant provides dense vector search for semantic ANN retrieval.\n\n"
            "## Cache Layer\n"
            "In-memory caching accelerates recurring query resolution without inference.\n",
            encoding="utf-8",
        )

        updated_doc3 = reader.read_file(doc_path)
        new_sections3 = chunker.extract_sections(updated_doc3)
        new_chunks3 = chunker.chunk_document(updated_doc3)

        existing3 = sqlite_store.get_document_by_path(updated_doc3.file_path)
        assert existing3 is not None
        sqlite_store.delete_document(existing3.id)
        sqlite_store.add_document(updated_doc3)
        sqlite_store.add_sections(new_sections3)
        sqlite_store.add_chunks(new_chunks3)

        # Sync after edit: 1 modified chunk (1 new + 1 orphaned), 4 unchanged
        report4 = engine.sync()
        assert report4.total_documents == 1
        assert report4.total_chunks == 5
        assert report4.new_chunks == 1
        assert report4.orphaned_chunks == 1
        assert report4.unchanged_chunks == 4
        assert embedder.total_texts_embedded == 6
