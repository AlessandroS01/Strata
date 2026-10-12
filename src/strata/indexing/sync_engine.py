"""Differential Synchronization Engine for Strata.

Reconciles derived search indices (embedded Qdrant vector store and BM25 sparse index)
against canonical SQLite chunks using content-addressable cryptographic hashes (SHA-256).
Bypasses redundant model inferences and maintains strict synchronization between dense
and sparse search layers.
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

from strata.core.config import Settings, get_settings
from strata.core.exceptions import StrataError, SyncConflictError
from strata.core.models import Chunk, ChunkDiff, SyncReport, SyncStats
from strata.core.protocols import DocumentStore, Embedder, SparseIndex, VectorStore

logger = logging.getLogger(__name__)


class IndexSyncEngine:
    """Differential index synchronization engine.

    Reconciles Qdrant vector points and BM25 inverted lexical representations
    against the canonical SQLite ACID document store.
    """

    def __init__(
        self,
        sqlite_store: DocumentStore,
        vector_store: VectorStore,
        sparse_index: SparseIndex,
        embedder: Embedder,
        settings: Settings | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialize IndexSyncEngine.

        Args:
            sqlite_store: Canonical SQLite document and chunk primary store.
            vector_store: Derived dense vector store adapter (Qdrant).
            sparse_index: Derived lexical inverted index adapter (BM25).
            embedder: Local dense embedding model runtime.
            settings: Optional Strata Settings instance.
            batch_size: Optional default batch size for sync operations.
                Defaults to settings.embedding_batch_size.
        """
        self.sqlite_store: DocumentStore = sqlite_store
        self.vector_store: VectorStore = vector_store
        self.sparse_index: SparseIndex = sparse_index
        self.embedder: Embedder = embedder
        self.settings: Settings = settings or get_settings()
        self.batch_size: int = (
            batch_size if batch_size is not None else self.settings.embedding_batch_size
        )

    @property
    def document_store(self) -> DocumentStore:
        """Alias property for sqlite_store."""
        return self.sqlite_store

    def compute_chunk_diff(self, document_id: str, new_chunks: list[Chunk]) -> ChunkDiff:
        """Compute differential reconciliation delta for a single document against SQLite.

        Args:
            document_id: Target canonical document UUID.
            new_chunks: Candidate chunk records produced by chunking.

        Returns:
            ChunkDiff model detailing new, unchanged, and orphaned chunk hashes.
        """
        active_hashes = set(self.sqlite_store.get_active_chunk_hashes(document_id))
        new_hashes = {c.chunk_hash for c in new_chunks}

        to_add = [c for c in new_chunks if c.chunk_hash not in active_hashes]
        unchanged = list(active_hashes.intersection(new_hashes))
        orphaned = list(active_hashes - new_hashes)

        return ChunkDiff(
            document_id=document_id,
            new_chunks=to_add,
            unchanged_hashes=unchanged,
            orphaned_hashes=orphaned,
        )

    def sync(self, batch_size: int | None = None) -> SyncReport:
        """Execute differential index synchronization.

        Reconciles vector and sparse indices against SQLite canonical chunks:
        1. Retrieves canonical chunks and hashes from sqlite_store.
        2. Retrieves indexed point IDs from vector_store.get_all_point_ids().
        3. Identifies orphaned vector points and purges them from vector_store and sparse_index.
        4. Identifies new or modified chunks needing embedding.
        5. Computes dense embeddings in batches and upserts to vector_store and sparse_index.
        6. Persists sparse index to settings.bm25_index_path if configured.
        7. Returns SyncStats telemetry.

        Args:
            batch_size: Optional batch size for dense embedding and vector upserts.
                Defaults to self.batch_size (settings.embedding_batch_size).

        Returns:
            SyncStats (SyncReport) summarizing synchronization telemetry.

        Raises:
            SyncConflictError: If unexpected runtime failure occurs during synchronization.
        """
        start_time = time.perf_counter()
        logger.info("Starting differential index synchronization...")

        effective_batch_size = max(1, batch_size if batch_size is not None else self.batch_size)

        try:
            # 1. Retrieve canonical chunk records from sqlite_store
            canonical_chunks = self.sqlite_store.get_all_chunks()
            canonical_ids = {c.id for c in canonical_chunks}
            total_documents = (
                self.sqlite_store.count_documents()
                if hasattr(self.sqlite_store, "count_documents")
                else len({c.document_id for c in canonical_chunks})
            )

            # 2. Retrieve indexed point IDs from vector_store
            indexed_point_ids: set[str] = set()
            if hasattr(self.vector_store, "get_all_point_ids"):
                raw_point_ids = cast(Any, self.vector_store).get_all_point_ids()
                indexed_point_ids = set(raw_point_ids)

            # 3. Identify and purge orphaned vector and sparse points
            orphaned_ids = indexed_point_ids - canonical_ids
            if orphaned_ids:
                orphaned_list = list(orphaned_ids)
                logger.info("Purging %d orphaned chunk points from indices", len(orphaned_list))
                if hasattr(self.vector_store, "delete_by_ids"):
                    cast(Any, self.vector_store).delete_by_ids(orphaned_list)

                if hasattr(self.sparse_index, "delete_by_ids"):
                    cast(Any, self.sparse_index).delete_by_ids(orphaned_list)
                elif hasattr(self.sparse_index, "delete_chunks"):
                    cast(Any, self.sparse_index).delete_chunks(orphaned_list)

            # 4. Identify new or modified chunks needing embedding
            new_chunks = [c for c in canonical_chunks if c.id not in indexed_point_ids]
            unchanged_count = len(canonical_chunks) - len(new_chunks)

            # 5. Process new chunks in batches
            if new_chunks:
                logger.info(
                    "Embedding and indexing %d new chunks (batch_size=%d)...",
                    len(new_chunks),
                    effective_batch_size,
                )
                for i in range(0, len(new_chunks), effective_batch_size):
                    batch = new_chunks[i : i + effective_batch_size]
                    try:
                        texts = [c.text for c in batch]
                        if hasattr(self.embedder, "embed_batch"):
                            vectors = cast(Any, self.embedder).embed_batch(texts)
                        else:
                            vectors = self.embedder.embed_texts(texts)

                        self.vector_store.upsert_chunks(batch, vectors)
                        self.sparse_index.index_chunks(batch)
                    except StrataError:
                        raise
                    except Exception as exc:
                        logger.error(
                            "Failed to synchronize batch at offset [%d:%d]: %s",
                            i,
                            i + len(batch),
                            exc,
                        )
                        raise SyncConflictError(
                            f"Failed to synchronize batch at offset {i}: {exc}"
                        ) from exc
            else:
                # If dense store is up to date, ensure sparse index is also populated
                # if it was freshly initialized in-memory with 0 indexed items
                if (
                    hasattr(self.sparse_index, "count")
                    and self.sparse_index.count == 0
                    and canonical_chunks
                ):
                    self.sparse_index.index_chunks(canonical_chunks)

            # 6. Persist sparse index to settings.bm25_index_path if configured
            bm25_path = getattr(self.settings, "bm25_index_path", None)
            if bm25_path is not None and hasattr(self.sparse_index, "save"):
                try:
                    cast(Any, self.sparse_index).save(bm25_path)
                    logger.debug("Persisted BM25 index to %s", bm25_path)
                except Exception as exc:
                    logger.warning("Failed to persist BM25 index to %s: %s", bm25_path, exc)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            report = SyncStats(
                total_documents=total_documents,
                total_chunks=len(canonical_chunks),
                new_chunks=len(new_chunks),
                unchanged_chunks=unchanged_count,
                orphaned_chunks=len(orphaned_ids),
                elapsed_ms=round(elapsed_ms, 2),
            )
            logger.info(
                "Sync complete in %.2fms: %d new, %d unchanged, %d orphaned",
                report.elapsed_ms,
                report.new_chunks,
                report.unchanged_chunks,
                report.orphaned_chunks,
            )
            return report

        except StrataError:
            raise
        except Exception as exc:
            logger.error("Synchronization failed with unhandled error: %s", exc)
            raise SyncConflictError(f"Synchronization failed: {exc}") from exc
