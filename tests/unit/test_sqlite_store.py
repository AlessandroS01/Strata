import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from strata.core.config import Settings
from strata.core.exceptions import DocumentNotFoundError
from strata.core.models import Chunk, Document, Section
from strata.core.protocols import DocumentStore
from strata.storage.sqlite_store import SQLiteStore


@pytest.fixture
def memory_store() -> Iterator[SQLiteStore]:
    """Provide an in-memory SQLiteStore for isolated unit testing."""
    store = SQLiteStore(":memory:")
    yield store
    store.close()


def make_document(
    doc_id: str = "doc-1",
    title: str = "Raft Consensus",
    file_path: str = "/docs/raft.md",
    tags: list[str] | None = None,
) -> Document:
    """Helper to construct a valid Document domain model."""
    return Document(
        id=doc_id,
        title=title,
        file_path=file_path,
        raw_content="# Raft\nDistributed consensus protocol notes.",
        doc_hash="hash-doc-123",
        tags=tags if tags is not None else ["distributed", "consensus"],
        metadata={"author": "Diego Ongaro", "year": "2014"},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def make_section(
    section_id: str = "sec-1",
    doc_id: str = "doc-1",
    title: str = "Leader Election",
    index: int = 0,
    level: int = 2,
) -> Section:
    """Helper to construct a valid Section domain model."""
    return Section(
        id=section_id,
        document_id=doc_id,
        title=title,
        level=level,
        heading_path=f"Raft > {title}",
        content=f"Detailed context text for {title}.",
        index=index,
        token_count=120,
    )


def make_chunk(
    chunk_id: str = "chunk-1",
    doc_id: str = "doc-1",
    index: int = 0,
    parent_section_id: str | None = None,
    chunk_hash: str | None = None,
    text: str | None = None,
    tags: list[str] | None = None,
) -> Chunk:
    """Helper to construct a valid Chunk domain model."""
    return Chunk(
        id=chunk_id,
        document_id=doc_id,
        chunk_hash=chunk_hash or f"sha256-hash-{chunk_id}",
        text=text or f"Search slice content for chunk {index}.",
        heading_path="Raft > Leader Election",
        index=index,
        parent_section_id=parent_section_id,
        token_count=45,
        tags=tags if tags is not None else ["distributed", "consensus"],
    )


def test_implements_document_store_protocol(memory_store: SQLiteStore) -> None:
    """Verify SQLiteStore conforms to the DocumentStore runtime Protocol."""
    assert isinstance(memory_store, DocumentStore)


def test_database_pragmas(memory_store: SQLiteStore) -> None:
    """Verify mandatory database pragmas are active."""
    cursor = memory_store._conn.execute("PRAGMA foreign_keys;")
    row = cursor.fetchone()
    assert row is not None
    assert int(row[0]) == 1


def test_document_crud_lifecycle(memory_store: SQLiteStore) -> None:
    """Verify document insertion, retrieval, update, counting, and deletion."""
    assert memory_store.count_documents() == 0

    doc = make_document()
    memory_store.add_document(doc)

    assert memory_store.count_documents() == 1

    # Fetch by ID
    retrieved = memory_store.get_document(doc.id)
    assert retrieved is not None
    assert retrieved.id == doc.id
    assert retrieved.title == doc.title
    assert retrieved.file_path == doc.file_path
    assert retrieved.raw_content == doc.raw_content
    assert retrieved.doc_hash == doc.doc_hash
    assert retrieved.tags == doc.tags
    assert retrieved.metadata == doc.metadata
    assert retrieved.created_at == doc.created_at
    assert retrieved.updated_at == doc.updated_at

    # Fetch by Path
    by_path = memory_store.get_document_by_path(doc.file_path)
    assert by_path is not None
    assert by_path.id == doc.id

    # Update document
    updated_doc = Document(
        id=doc.id,
        title="Raft Consensus (Revised)",
        file_path=doc.file_path,
        raw_content="# Raft Revised\nUpdated content.",
        doc_hash="hash-doc-456",
        tags=["distributed", "paxos"],
        metadata={"author": "Diego Ongaro", "version": "2"},
        created_at=doc.created_at,
        updated_at=datetime.now(UTC),
    )
    memory_store.update_document(updated_doc)

    after_update = memory_store.get_document(doc.id)
    assert after_update is not None
    assert after_update.title == "Raft Consensus (Revised)"
    assert after_update.raw_content == "# Raft Revised\nUpdated content."
    assert after_update.doc_hash == "hash-doc-456"
    assert after_update.tags == ["distributed", "paxos"]
    assert after_update.metadata["version"] == "2"

    # Delete document
    memory_store.delete_document(doc.id)
    assert memory_store.count_documents() == 0
    assert memory_store.get_document(doc.id) is None
    assert memory_store.get_document_by_path(doc.file_path) is None


def test_get_document_not_found(memory_store: SQLiteStore) -> None:
    """Verify missing document queries return None."""
    assert memory_store.get_document("non-existent-id") is None
    assert memory_store.get_document_by_path("/non/existent/path.md") is None


def test_update_nonexistent_document_raises(memory_store: SQLiteStore) -> None:
    """Verify updating a non-existent document raises DocumentNotFoundError."""
    doc = make_document(doc_id="missing-id")
    with pytest.raises(DocumentNotFoundError, match="missing-id"):
        memory_store.update_document(doc)


def test_document_unique_file_path_constraint(memory_store: SQLiteStore) -> None:
    """Verify duplicate file_path violates uniqueness constraint."""
    doc1 = make_document(doc_id="doc-1", file_path="/path/test.md")
    doc2 = make_document(doc_id="doc-2", file_path="/path/test.md")

    memory_store.add_document(doc1)
    with pytest.raises(sqlite3.IntegrityError):
        memory_store.add_document(doc2)


def test_list_documents_tag_filtering(memory_store: SQLiteStore) -> None:
    """Verify document listing with optional tag filtering."""
    doc1 = make_document(doc_id="d1", file_path="/d1.md", tags=["python", "backend"])
    doc2 = make_document(doc_id="d2", file_path="/d2.md", tags=["rust", "backend"])
    doc3 = make_document(doc_id="d3", file_path="/d3.md", tags=["frontend", "css"])

    memory_store.add_document(doc1)
    memory_store.add_document(doc2)
    memory_store.add_document(doc3)

    # All documents
    all_docs = memory_store.list_documents()
    assert len(all_docs) == 3

    # Filter with single tag
    backend_docs = memory_store.list_documents(tags=["backend"])
    assert len(backend_docs) == 2
    assert {d.id for d in backend_docs} == {"d1", "d2"}

    # Filter with multiple tags (AND match)
    py_backend = memory_store.list_documents(tags=["backend", "python"])
    assert len(py_backend) == 1
    assert py_backend[0].id == "d1"

    # Filter with non-matching tag
    empty_list = memory_store.list_documents(tags=["database"])
    assert len(empty_list) == 0

    # Filter with empty tag list
    all_again = memory_store.list_documents(tags=[])
    assert len(all_again) == 3


def test_section_crud_and_ordering(memory_store: SQLiteStore) -> None:
    """Verify adding and retrieving sections ordered by index."""
    doc = make_document()
    memory_store.add_document(doc)

    sec2 = make_section(section_id="s2", doc_id=doc.id, title="Section 2", index=2)
    sec0 = make_section(section_id="s0", doc_id=doc.id, title="Section 0", index=0)
    sec1 = make_section(section_id="s1", doc_id=doc.id, title="Section 1", index=1)

    # Insert out of order
    memory_store.add_sections([sec2, sec0, sec1])

    # Fetch individual section
    s0 = memory_store.get_section("s0")
    assert s0 is not None
    assert s0.title == "Section 0"
    assert s0.index == 0

    assert memory_store.get_section("nonexistent") is None

    # Fetch all for document (must be ordered by index)
    sections = memory_store.get_sections_by_document_id(doc.id)
    assert len(sections) == 3
    assert [s.id for s in sections] == ["s0", "s1", "s2"]
    assert [s.index for s in sections] == [0, 1, 2]


def test_section_foreign_key_constraint(memory_store: SQLiteStore) -> None:
    """Verify adding a section with invalid document_id fails."""
    sec = make_section(doc_id="non-existent-doc")
    with pytest.raises(sqlite3.IntegrityError):
        memory_store.add_sections([sec])


def test_chunk_crud_and_retrieval(memory_store: SQLiteStore) -> None:
    """Verify chunk batch insertion, retrieval, and counting."""
    doc1 = make_document(doc_id="doc-1", file_path="/d1.md")
    doc2 = make_document(doc_id="doc-2", file_path="/d2.md")
    memory_store.add_document(doc1)
    memory_store.add_document(doc2)

    c0 = make_chunk(chunk_id="c0", doc_id="doc-1", index=0, chunk_hash="hash0")
    c1 = make_chunk(chunk_id="c1", doc_id="doc-1", index=1, chunk_hash="hash1")
    c2 = make_chunk(chunk_id="c2", doc_id="doc-2", index=0, chunk_hash="hash2")

    memory_store.add_chunks([c0, c1, c2])

    assert memory_store.count_chunks() == 3

    # Fetch individual
    chunk = memory_store.get_chunk("c0")
    assert chunk is not None
    assert chunk.id == "c0"
    assert chunk.chunk_hash == "hash0"
    assert chunk.index == 0

    assert memory_store.get_chunk("missing") is None

    # Fetch by document
    doc1_chunks = memory_store.get_chunks_by_document_id("doc-1")
    assert len(doc1_chunks) == 2
    assert [c.id for c in doc1_chunks] == ["c0", "c1"]

    # Fetch all chunks
    all_chunks = memory_store.get_all_chunks()
    assert len(all_chunks) == 3


def test_chunk_foreign_key_constraint(memory_store: SQLiteStore) -> None:
    """Verify adding a chunk with invalid document_id fails."""
    chunk = make_chunk(doc_id="non-existent-doc")
    with pytest.raises(sqlite3.IntegrityError):
        memory_store.add_chunks([chunk])


def test_foreign_key_cascade_deletion(memory_store: SQLiteStore) -> None:
    """Verify deleting a document automatically cascades to its sections and chunks."""
    doc = make_document()
    memory_store.add_document(doc)

    sec = make_section(doc_id=doc.id)
    memory_store.add_sections([sec])

    c0 = make_chunk(chunk_id="c0", doc_id=doc.id, index=0, parent_section_id=sec.id)
    c1 = make_chunk(chunk_id="c1", doc_id=doc.id, index=1, parent_section_id=sec.id)
    memory_store.add_chunks([c0, c1])

    assert memory_store.count_documents() == 1
    assert len(memory_store.get_sections_by_document_id(doc.id)) == 1
    assert memory_store.count_chunks() == 2

    # Delete the parent document
    memory_store.delete_document(doc.id)

    assert memory_store.count_documents() == 0
    assert len(memory_store.get_sections_by_document_id(doc.id)) == 0
    assert memory_store.count_chunks() == 0
    assert memory_store.get_section(sec.id) is None
    assert memory_store.get_chunk("c0") is None
    assert memory_store.get_chunk("c1") is None


def test_section_delete_sets_chunk_parent_null(memory_store: SQLiteStore) -> None:
    """Verify deleting a section sets chunk parent_section_id to None."""
    doc = make_document()
    memory_store.add_document(doc)

    sec = make_section(doc_id=doc.id)
    memory_store.add_sections([sec])

    chunk = make_chunk(chunk_id="c0", doc_id=doc.id, index=0, parent_section_id=sec.id)
    memory_store.add_chunks([chunk])

    # Raw delete of section
    memory_store._conn.execute("DELETE FROM sections WHERE id = ?", (sec.id,))

    # Chunk should still exist with parent_section_id = None
    updated_chunk = memory_store.get_chunk("c0")
    assert updated_chunk is not None
    assert updated_chunk.parent_section_id is None


def test_active_chunk_hashes_and_deletion(memory_store: SQLiteStore) -> None:
    """Verify active chunk hashes retrieval and differential deletion by hash."""
    doc = make_document()
    memory_store.add_document(doc)

    chunks = [
        make_chunk(chunk_id=f"c{i}", doc_id=doc.id, index=i, chunk_hash=f"hash-{i}")
        for i in range(5)
    ]
    memory_store.add_chunks(chunks)

    active_hashes = memory_store.get_active_chunk_hashes(doc.id)
    assert active_hashes == ["hash-0", "hash-1", "hash-2", "hash-3", "hash-4"]

    # Delete subset of hashes
    memory_store.delete_chunks_by_hashes(["hash-1", "hash-3"])

    remaining_hashes = memory_store.get_active_chunk_hashes(doc.id)
    assert remaining_hashes == ["hash-0", "hash-2", "hash-4"]
    assert memory_store.count_chunks() == 3

    # Deleting empty list does nothing
    memory_store.delete_chunks_by_hashes([])
    assert memory_store.count_chunks() == 3


def test_delete_chunks_by_hashes_large_batch(memory_store: SQLiteStore) -> None:
    """Verify deleting large number of chunk hashes exceeding batch limit."""
    doc = make_document()
    memory_store.add_document(doc)

    chunks = [
        make_chunk(chunk_id=f"c_{i}", doc_id=doc.id, index=i, chunk_hash=f"bulk-hash-{i}")
        for i in range(600)
    ]
    memory_store.add_chunks(chunks)
    assert memory_store.count_chunks() == 600

    hashes_to_delete = [f"bulk-hash-{i}" for i in range(600)]
    memory_store.delete_chunks_by_hashes(hashes_to_delete)
    assert memory_store.count_chunks() == 0


def test_get_document_outline(memory_store: SQLiteStore) -> None:
    """Verify Table of Contents outline generation ordered by index."""
    doc = make_document()
    memory_store.add_document(doc)

    sec0 = Section(
        id="s0",
        document_id=doc.id,
        title="Introduction",
        level=1,
        heading_path="Introduction",
        content="Intro text",
        index=0,
        token_count=50,
    )
    sec1 = Section(
        id="s1",
        document_id=doc.id,
        title="Architecture",
        level=2,
        heading_path="Introduction > Architecture",
        content="Arch text",
        index=1,
        token_count=100,
    )
    sec2 = Section(
        id="s2",
        document_id=doc.id,
        title="Consensus Algorithm",
        level=2,
        heading_path="Introduction > Consensus Algorithm",
        content="Consensus text",
        index=2,
        token_count=150,
    )

    # Insert out of order
    memory_store.add_sections([sec2, sec0, sec1])

    outline = memory_store.get_document_outline(doc.id)
    assert outline.document_id == doc.id
    assert outline.document_title == doc.title
    assert len(outline.items) == 3

    assert outline.items[0].section_id == "s0"
    assert outline.items[0].title == "Introduction"
    assert outline.items[0].level == 1
    assert outline.items[0].token_count == 50

    assert outline.items[1].section_id == "s1"
    assert outline.items[1].title == "Architecture"
    assert outline.items[1].level == 2

    assert outline.items[2].section_id == "s2"
    assert outline.items[2].title == "Consensus Algorithm"
    assert outline.items[2].level == 2


def test_get_document_outline_not_found(memory_store: SQLiteStore) -> None:
    """Verify get_document_outline raises DocumentNotFoundError for missing document."""
    with pytest.raises(DocumentNotFoundError, match="non-existent"):
        memory_store.get_document_outline("non-existent")


def test_get_adjacent_chunks_boundaries(memory_store: SQLiteStore) -> None:
    """Verify get_adjacent_chunks respects sequential window boundaries."""
    doc = make_document(doc_id="doc-1", file_path="/d1.md")
    doc_other = make_document(doc_id="doc-2", file_path="/d2.md")
    memory_store.add_document(doc)
    memory_store.add_document(doc_other)

    chunks = [make_chunk(chunk_id=f"c{i}", doc_id=doc.id, index=i) for i in range(5)]
    other_chunk = make_chunk(chunk_id="other_c", doc_id=doc_other.id, index=2)

    memory_store.add_chunks([*chunks, other_chunk])

    # Middle chunk with window=1: [c1, c2, c3]
    mid = memory_store.get_adjacent_chunks(chunk_id="c2", window=1)
    assert [c.id for c in mid] == ["c1", "c2", "c3"]

    # First chunk with window=1: [c0, c1]
    start = memory_store.get_adjacent_chunks(chunk_id="c0", window=1)
    assert [c.id for c in start] == ["c0", "c1"]

    # Last chunk with window=1: [c3, c4]
    end = memory_store.get_adjacent_chunks(chunk_id="c4", window=1)
    assert [c.id for c in end] == ["c3", "c4"]

    # Window=0: target chunk only
    zero_win = memory_store.get_adjacent_chunks(chunk_id="c2", window=0)
    assert [c.id for c in zero_win] == ["c2"]

    # Window=2 on c2: all 5 chunks [c0..c4]
    wide_win = memory_store.get_adjacent_chunks(chunk_id="c2", window=2)
    assert [c.id for c in wide_win] == ["c0", "c1", "c2", "c3", "c4"]

    # Excessive window: clamps to available range without error
    huge_win = memory_store.get_adjacent_chunks(chunk_id="c2", window=10)
    assert [c.id for c in huge_win] == ["c0", "c1", "c2", "c3", "c4"]

    # Other document chunk with window=1 must not leak doc-1 chunks
    other_adj = memory_store.get_adjacent_chunks(chunk_id="other_c", window=1)
    assert [c.id for c in other_adj] == ["other_c"]

    # Non-existent chunk returns empty list
    missing = memory_store.get_adjacent_chunks(chunk_id="non-existent")
    assert missing == []

    # Negative window raises ValueError
    with pytest.raises(ValueError, match="non-negative"):
        memory_store.get_adjacent_chunks(chunk_id="c2", window=-1)


def test_disk_store_persistence_and_context_manager(tmp_path: Path) -> None:
    """Verify SQLiteStore persists data to disk and functions as a context manager."""
    db_file = tmp_path / "nested" / "test.db"

    doc = make_document(doc_id="persisted-doc")
    chunk = make_chunk(chunk_id="persisted-chunk", doc_id="persisted-doc", index=0)

    # Write using context manager
    with SQLiteStore(db_path=db_file) as store:
        store.add_document(doc)
        store.add_chunks([chunk])
        assert store.count_documents() == 1
        assert store.count_chunks() == 1

    # Reopen same database file
    with SQLiteStore(db_path=db_file) as reopened:
        retrieved_doc = reopened.get_document("persisted-doc")
        assert retrieved_doc is not None
        assert retrieved_doc.title == doc.title

        retrieved_chunk = reopened.get_chunk("persisted-chunk")
        assert retrieved_chunk is not None
        assert retrieved_chunk.id == chunk.id


def test_init_with_existing_connection() -> None:
    """Verify SQLiteStore accepts a pre-existing sqlite3.Connection."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    with SQLiteStore(connection=conn) as store:
        assert store.count_documents() == 0
        doc = make_document()
        store.add_document(doc)
        assert store.count_documents() == 1


def test_init_with_default_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SQLiteStore defaults to Settings path when no arguments provided."""
    custom_db = tmp_path / "default_dir" / "canonical.db"
    test_settings = Settings(db_path=custom_db)
    monkeypatch.setattr("strata.storage.sqlite_store.get_settings", lambda: test_settings)

    with SQLiteStore() as store:
        assert store.count_documents() == 0
        doc = make_document()
        store.add_document(doc)
        assert store.count_documents() == 1

    assert custom_db.exists()


def test_add_empty_sections_and_chunks(memory_store: SQLiteStore) -> None:
    """Verify adding empty lists of sections and chunks executes without error."""
    memory_store.add_sections([])
    memory_store.add_chunks([])
    assert memory_store.count_chunks() == 0
