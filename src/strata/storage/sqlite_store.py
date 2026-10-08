"""SQLite canonical ACID storage adapter implementing the DocumentStore protocol.

SQLite is the canonical source of truth in Strata: raw document text,
hierarchical sections, granular search chunks, and SHA-256 hashes live here.
"""

import json
import logging
import sqlite3
import types
from datetime import datetime
from pathlib import Path

from strata.core.config import get_settings
from strata.core.exceptions import DocumentNotFoundError
from strata.core.models import (
    Chunk,
    Document,
    DocumentOutline,
    DocumentOutlineItem,
    Section,
)
from strata.core.protocols import DocumentStore

logger = logging.getLogger(__name__)


class SQLiteStore(DocumentStore):
    """Canonical document and chunk storage engine backed by local SQLite.

    Fulfills the DocumentStore protocol with ACID transactional integrity,
    parameterized SQL queries, and foreign-key cascade deletions.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        """Initialize SQLite database connection, configure pragmas, and create schema.

        Args:
            db_path: Optional path to SQLite file or ':memory:'. Defaults to
                configured path in Settings.
            connection: Optional existing sqlite3.Connection instance (useful
                for tests or connection sharing).
        """
        if connection is not None:
            self._conn = connection
        elif db_path is not None:
            db_str = str(db_path)
            if db_str != ":memory:" and not db_str.startswith("file:"):
                Path(db_str).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(db_str, check_same_thread=False)
        else:
            settings_path = get_settings().db_path
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(settings_path), check_same_thread=False)

        self._conn.row_factory = sqlite3.Row
        self._init_pragmas()
        self._create_schema()

    def _init_pragmas(self) -> None:
        """Configure SQLite engine pragmas for performance and foreign key integrity."""
        # SQLite leaves foreign key validation disabled by default for backwards
        # compatibility. Turning this on enforces the ON DELETE CASCADE rules
        # defined in the schema.
        self._conn.execute("PRAGMA foreign_keys = ON;")
        # (Write-Ahead Logging): Instead of locking the entire database file
        # during writes, changes are appended to a separate -wal file.
        # This allows readers to continue querying without blocking writers,
        # keeping lookup latency well under the 5ms target.
        self._conn.execute("PRAGMA journal_mode = WAL;")
        # In WAL mode, NORMAL synchronizes disk buffers at critical checkpoints
        # rather than on every single write transaction.
        # It is safe against application crashes while dramatically increasing
        # batch write throughput.
        self._conn.execute("PRAGMA synchronous = NORMAL;")

    def _create_schema(self) -> None:
        """Create relational tables and indices if they do not exist."""
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    file_path TEXT NOT NULL UNIQUE,
                    raw_content TEXT NOT NULL,
                    doc_hash TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sections (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    level INTEGER NOT NULL,
                    heading_path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    index_order INTEGER NOT NULL,
                    token_count INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sections_document_order
                ON sections (document_id, index_order);

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    heading_path TEXT NOT NULL,
                    index_order INTEGER NOT NULL,
                    parent_section_id TEXT REFERENCES sections(id) ON DELETE SET NULL,
                    token_count INTEGER NOT NULL,
                    tags TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_hash
                ON chunks (chunk_hash);

                CREATE INDEX IF NOT EXISTS idx_chunks_document_order
                ON chunks (document_id, index_order);
                """
            )

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> Document:
        """Hydrate a Document domain model from an SQLite Row."""
        raw_tags: object = json.loads(str(row["tags"]))
        tags: list[str] = [str(t) for t in raw_tags] if isinstance(raw_tags, list) else []
        raw_meta: object = json.loads(str(row["metadata"]))
        metadata: dict[str, str] = (
            {str(k): str(v) for k, v in raw_meta.items()} if isinstance(raw_meta, dict) else {}
        )
        return Document(
            id=str(row["id"]),
            title=str(row["title"]),
            file_path=str(row["file_path"]),
            raw_content=str(row["raw_content"]),
            doc_hash=str(row["doc_hash"]),
            tags=tags,
            metadata=metadata,
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )

    @staticmethod
    def _row_to_section(row: sqlite3.Row) -> Section:
        """Hydrate a Section domain model from an SQLite Row."""
        return Section(
            id=str(row["id"]),
            document_id=str(row["document_id"]),
            title=str(row["title"]),
            level=int(row["level"]),
            heading_path=str(row["heading_path"]),
            content=str(row["content"]),
            index=int(row["index_order"]),
            token_count=int(row["token_count"]),
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        """Hydrate a Chunk domain model from an SQLite Row."""
        raw_tags: object = json.loads(str(row["tags"]))
        tags: list[str] = [str(t) for t in raw_tags] if isinstance(raw_tags, list) else []
        parent_sec: object = row["parent_section_id"]
        return Chunk(
            id=str(row["id"]),
            document_id=str(row["document_id"]),
            chunk_hash=str(row["chunk_hash"]),
            text=str(row["text"]),
            heading_path=str(row["heading_path"]),
            index=int(row["index_order"]),
            parent_section_id=str(parent_sec) if parent_sec is not None else None,
            token_count=int(row["token_count"]),
            tags=tags,
        )

    # -------------------------------------------------------------------------
    # Document Lifecycle Methods
    # -------------------------------------------------------------------------

    def add_document(self, document: Document) -> None:
        """Atomically insert a new canonical document."""
        query = """
        INSERT INTO documents (
            id, title, file_path, raw_content, doc_hash, tags, metadata, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            document.id,
            document.title,
            document.file_path,
            document.raw_content,
            document.doc_hash,
            json.dumps(document.tags),
            json.dumps(document.metadata),
            document.created_at.isoformat(),
            document.updated_at.isoformat(),
        )
        with self._conn:
            self._conn.execute(query, params)
        logger.debug("Added canonical document '%s' (id=%s)", document.title, document.id)

    def get_document(self, document_id: str) -> Document | None:
        """Fetch canonical document by UUID."""
        query = """
        SELECT id, title, file_path, raw_content, doc_hash, tags, metadata, created_at, updated_at
        FROM documents
        WHERE id = ?
        """
        cursor = self._conn.execute(query, (document_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    def get_document_by_path(self, file_path: str) -> Document | None:
        """Fetch canonical document by original file path."""
        query = """
        SELECT id, title, file_path, raw_content, doc_hash, tags, metadata, created_at, updated_at
        FROM documents
        WHERE file_path = ?
        """
        cursor = self._conn.execute(query, (file_path,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    def list_documents(self, tags: list[str] | None = None) -> list[Document]:
        """List documents, optionally filtered by user tags."""
        query = """
        SELECT id, title, file_path, raw_content, doc_hash, tags, metadata, created_at, updated_at
        FROM documents
        ORDER BY created_at ASC, id ASC
        """
        cursor = self._conn.execute(query)
        rows = cursor.fetchall()
        documents = [self._row_to_document(row) for row in rows]
        if not tags:
            return documents
        required_tags = set(tags)
        return [doc for doc in documents if required_tags.issubset(set(doc.tags))]

    def update_document(self, document: Document) -> None:
        """Update existing document content and metadata in SQLite."""
        query = """
        UPDATE documents
        SET title = ?, file_path = ?, raw_content = ?, doc_hash = ?, tags = ?,
            metadata = ?, created_at = ?, updated_at = ?
        WHERE id = ?
        """
        params = (
            document.title,
            document.file_path,
            document.raw_content,
            document.doc_hash,
            json.dumps(document.tags),
            json.dumps(document.metadata),
            document.created_at.isoformat(),
            document.updated_at.isoformat(),
            document.id,
        )
        with self._conn:
            cursor = self._conn.execute(query, params)
            if cursor.rowcount == 0:
                raise DocumentNotFoundError(f"Document with ID '{document.id}' not found.")
        logger.debug("Updated canonical document '%s' (id=%s)", document.title, document.id)

    def delete_document(self, document_id: str) -> None:
        """Delete document and cascade-delete its sections and chunks in SQLite."""
        query = "DELETE FROM documents WHERE id = ?"
        with self._conn:
            self._conn.execute(query, (document_id,))
        logger.debug("Deleted canonical document id=%s", document_id)

    def count_documents(self) -> int:
        """Return total count of stored canonical documents."""
        query = "SELECT COUNT(*) FROM documents"
        cursor = self._conn.execute(query)
        row = cursor.fetchone()
        return int(row[0]) if row is not None else 0

    # -------------------------------------------------------------------------
    # Section Queries
    # -------------------------------------------------------------------------

    def add_sections(self, sections: list[Section]) -> None:
        """Batch-insert hierarchical document sections."""
        if not sections:
            return
        query = """
        INSERT INTO sections (
            id, document_id, title, level, heading_path, content, index_order, token_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = [
            (
                sec.id,
                sec.document_id,
                sec.title,
                sec.level,
                sec.heading_path,
                sec.content,
                sec.index,
                sec.token_count,
            )
            for sec in sections
        ]
        with self._conn:
            self._conn.executemany(query, params)
        logger.debug("Inserted %d sections", len(sections))

    def get_section(self, section_id: str) -> Section | None:
        """Retrieve a specific section by UUID."""
        query = """
        SELECT id, document_id, title, level, heading_path, content, index_order, token_count
        FROM sections
        WHERE id = ?
        """
        cursor = self._conn.execute(query, (section_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_section(row)

    def get_sections_by_document_id(self, document_id: str) -> list[Section]:
        """Retrieve all sections belonging to a document ordered by index."""
        query = """
        SELECT id, document_id, title, level, heading_path, content, index_order, token_count
        FROM sections
        WHERE document_id = ?
        ORDER BY index_order ASC
        """
        cursor = self._conn.execute(query, (document_id,))
        rows = cursor.fetchall()
        return [self._row_to_section(row) for row in rows]

    # -------------------------------------------------------------------------
    # Chunk Queries
    # -------------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Batch-insert chunk records."""
        if not chunks:
            return
        query = """
        INSERT INTO chunks (
            id, document_id, chunk_hash, text, heading_path, index_order,
            parent_section_id, token_count, tags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = [
            (
                c.id,
                c.document_id,
                c.chunk_hash,
                c.text,
                c.heading_path,
                c.index,
                c.parent_section_id,
                c.token_count,
                json.dumps(c.tags),
            )
            for c in chunks
        ]
        with self._conn:
            self._conn.executemany(query, params)
        logger.debug("Inserted %d chunks", len(chunks))

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a specific chunk by UUID."""
        query = """
        SELECT id, document_id, chunk_hash, text, heading_path, index_order,
               parent_section_id, token_count, tags
        FROM chunks
        WHERE id = ?
        """
        cursor = self._conn.execute(query, (chunk_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    def get_chunks_by_document_id(self, document_id: str) -> list[Chunk]:
        """Retrieve all chunks belonging to a document ordered by index."""
        query = """
        SELECT id, document_id, chunk_hash, text, heading_path, index_order,
               parent_section_id, token_count, tags
        FROM chunks
        WHERE document_id = ?
        ORDER BY index_order ASC
        """
        cursor = self._conn.execute(query, (document_id,))
        rows = cursor.fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_all_chunks(self) -> list[Chunk]:
        """Retrieve all active chunks across all documents."""
        query = """
        SELECT id, document_id, chunk_hash, text, heading_path, index_order,
               parent_section_id, token_count, tags
        FROM chunks
        ORDER BY document_id ASC, index_order ASC
        """
        cursor = self._conn.execute(query)
        rows = cursor.fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_active_chunk_hashes(self, document_id: str) -> list[str]:
        """Return list of active SHA-256 chunk hashes for differential invalidation."""
        query = """
        SELECT chunk_hash
        FROM chunks
        WHERE document_id = ?
        ORDER BY index_order ASC
        """
        cursor = self._conn.execute(query, (document_id,))
        rows = cursor.fetchall()
        return [str(row["chunk_hash"]) for row in rows]

    def delete_chunks_by_hashes(self, chunk_hashes: list[str]) -> None:
        """Batch-delete chunks matching the provided hashes."""
        if not chunk_hashes:
            return
        # prevents operational error: too many SQL variables
        batch_size = 500
        with self._conn:
            for i in range(0, len(chunk_hashes), batch_size):
                batch = chunk_hashes[i : i + batch_size]
                placeholders = ",".join("?" for _ in batch)
                query = f"DELETE FROM chunks WHERE chunk_hash IN ({placeholders})"
                self._conn.execute(query, batch)
        logger.debug("Deleted chunks for %d hashes", len(chunk_hashes))

    def count_chunks(self) -> int:
        """Return total count of stored chunks."""
        query = "SELECT COUNT(*) FROM chunks"
        cursor = self._conn.execute(query)
        row = cursor.fetchone()
        return int(row[0]) if row is not None else 0

    # -------------------------------------------------------------------------
    # Agent Navigation Primitives
    # -------------------------------------------------------------------------

    def get_document_outline(self, document_id: str) -> DocumentOutline:
        """Construct Table of Contents outline from stored sections."""
        doc = self.get_document(document_id)
        if doc is None:
            raise DocumentNotFoundError(f"Document with ID '{document_id}' not found.")

        sections = self.get_sections_by_document_id(document_id)
        items = [
            DocumentOutlineItem(
                level=sec.level,
                title=sec.title,
                heading_path=sec.heading_path,
                section_id=sec.id,
                token_count=sec.token_count,
            )
            for sec in sections
        ]
        return DocumentOutline(
            document_id=document_id,
            document_title=doc.title,
            items=items,
        )

    def get_adjacent_chunks(self, chunk_id: str, window: int = 1) -> list[Chunk]:
        """Retrieve adjacent sibling chunks in sequential index order.

        Fetches chunks within [target_index - window, target_index + window]
        belonging to the same parent document.
        """
        target_chunk = self.get_chunk(chunk_id)
        if target_chunk is None:
            return []
        if window < 0:
            raise ValueError(f"window must be non-negative, got {window}")

        min_index = max(0, target_chunk.index - window)
        max_index = target_chunk.index + window
        query = """
        SELECT id, document_id, chunk_hash, text, heading_path, index_order,
               parent_section_id, token_count, tags
        FROM chunks
        WHERE document_id = ? AND index_order >= ? AND index_order <= ?
        ORDER BY index_order ASC
        """
        cursor = self._conn.execute(query, (target_chunk.document_id, min_index, max_index))
        rows = cursor.fetchall()
        return [self._row_to_chunk(row) for row in rows]

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "SQLiteStore":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit context manager and close connection."""
        self.close()
