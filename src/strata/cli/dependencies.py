"""Dependency factory and container for Strata command-line interface.

Wires storage, derived indices, embedding, re-ranking, and retrieval adapters
from application configuration into a typed dataclass container.
"""

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from strata.core.config import Settings, get_settings
from strata.indexing.bm25_index import BM25Index
from strata.indexing.embedder import FastEmbedder
from strata.indexing.qdrant_store import QdrantVectorStore
from strata.indexing.sync_engine import IndexSyncEngine
from strata.ingestion.chunker import MarkdownChunker
from strata.ingestion.reader import LocalFileReader
from strata.retrieval.hybrid_search import StrataHybridSearcher
from strata.retrieval.reranker import FlashRankReranker
from strata.storage.sqlite_store import SQLiteStore


@dataclass
class CLIDependencies:
    """Dependency container bundling storage, retrieval, and indexing adapters."""

    settings: Settings
    sqlite_store: SQLiteStore
    vector_store: QdrantVectorStore
    sparse_index: BM25Index
    embedder: FastEmbedder
    reranker: FlashRankReranker
    searcher: StrataHybridSearcher
    sync_engine: IndexSyncEngine
    chunker: MarkdownChunker
    reader: LocalFileReader

    @property
    def hybrid_searcher(self) -> StrataHybridSearcher:
        """Alias property for searcher."""
        return self.searcher

    def close(self) -> None:
        """Close opened resources and storage connections."""
        if hasattr(self.sqlite_store, "close"):
            self.sqlite_store.close()
        if hasattr(self.vector_store, "close"):
            self.vector_store.close()

    def __enter__(self) -> "CLIDependencies":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()


_dependency_override: object | None = None


def set_dependency_override(dependencies: object | None) -> None:
    """Set global dependency override for isolated testing."""
    global _dependency_override
    _dependency_override = dependencies


def reset_dependencies() -> None:
    """Reset global dependency override to default factory."""
    global _dependency_override
    _dependency_override = None


def create_dependencies(settings: Settings | None = None) -> CLIDependencies:
    """Instantiate and wire all storage and retrieval adapters from settings.

    Args:
        settings: Optional Strata Settings override. Defaults to global Settings singleton.

    Returns:
        Fully wired CLIDependencies container instance.
    """
    cfg = settings or get_settings()

    # Ensure parent data directories exist
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.qdrant_path.mkdir(parents=True, exist_ok=True)

    sqlite_store = SQLiteStore(db_path=cfg.db_path)
    vector_store = QdrantVectorStore(settings=cfg)

    sparse_index = BM25Index(settings=cfg)
    if cfg.bm25_index_path and Path(cfg.bm25_index_path).is_file():
        with contextlib.suppress(Exception):
            sparse_index.load_from_path(cfg.bm25_index_path)

    embedder = FastEmbedder(settings=cfg)
    reranker = FlashRankReranker(settings=cfg)

    sync_engine = IndexSyncEngine(
        sqlite_store=sqlite_store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        settings=cfg,
    )

    searcher = StrataHybridSearcher(
        sqlite_store=sqlite_store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
        settings=cfg,
    )

    chunker = MarkdownChunker(settings=cfg)
    reader = LocalFileReader()

    return CLIDependencies(
        settings=cfg,
        sqlite_store=sqlite_store,
        vector_store=vector_store,
        sparse_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
        searcher=searcher,
        sync_engine=sync_engine,
        chunker=chunker,
        reader=reader,
    )


def get_dependencies(settings: Settings | None = None) -> CLIDependencies:
    """Resolve active CLIDependencies instance, checking for test overrides first.

    Args:
        settings: Optional Settings override.

    Returns:
        Active CLIDependencies container.
    """
    if _dependency_override is not None:
        return cast(CLIDependencies, _dependency_override)
    return create_dependencies(settings)
