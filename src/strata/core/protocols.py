"""Abstract Protocols for Strata decoupled components.

Enforces structural subtyping (typing.Protocol) across storage, retrieval,
chunking, and generation layers without concrete class inheritance.
"""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from strata.core.models import (
    Chunk,
    Document,
    DocumentOutline,
    RetrievedChunk,
    RetrievedContext,
    SearchResult,
    Section,
)


@runtime_checkable
class DocumentStore(Protocol):
    """Canonical document and chunk storage interface (SQLite ACID source of truth)."""

    def add_document(self, document: Document) -> None:
        """Atomically insert a new canonical document."""
        ...

    def get_document(self, document_id: str) -> Document | None:
        """Fetch canonical document by UUID."""
        ...

    def get_document_by_path(self, file_path: str) -> Document | None:
        """Fetch canonical document by original file path."""
        ...

    def list_documents(self, tags: list[str] | None = None) -> list[Document]:
        """List documents, optionally filtered by user tags."""
        ...

    def update_document(self, document: Document) -> None:
        """Update existing document content and metadata in SQLite."""
        ...

    def delete_document(self, document_id: str) -> None:
        """Delete document and cascade-delete its sections and chunks in SQLite."""
        ...

    def add_sections(self, sections: list[Section]) -> None:
        """Batch-insert hierarchical document sections."""
        ...

    def get_section(self, section_id: str) -> Section | None:
        """Retrieve a specific section by UUID."""
        ...

    def get_sections_by_document_id(self, document_id: str) -> list[Section]:
        """Retrieve all sections belonging to a document ordered by index."""
        ...

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Batch-insert chunk records."""
        ...

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a specific chunk by UUID."""
        ...

    def get_chunks_by_document_id(self, document_id: str) -> list[Chunk]:
        """Retrieve all chunks belonging to a document ordered by index."""
        ...

    def get_all_chunks(self) -> list[Chunk]:
        """Retrieve all active chunks across all documents."""
        ...

    def get_active_chunk_hashes(self, document_id: str) -> list[str]:
        """Return list of active SHA-256 chunk hashes for differential invalidation."""
        ...

    def delete_chunks_by_hashes(self, chunk_hashes: list[str]) -> None:
        """Batch-delete chunks matching the provided hashes."""
        ...

    def get_document_outline(self, document_id: str) -> DocumentOutline:
        """Construct Table of Contents outline from stored sections."""
        ...

    def get_adjacent_chunks(self, chunk_id: str, window: int = 1) -> list[Chunk]:
        """Retrieve adjacent sibling chunks in sequential index order."""
        ...

    def count_documents(self) -> int:
        """Return total count of stored canonical documents."""
        ...

    def count_chunks(self) -> int:
        """Return total count of stored chunks."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Derived vector index interface (Embedded Qdrant)."""

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunk vectors and light metadata payloads into vector store."""
        ...

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunk points belonging to a document ID."""
        ...

    def delete_by_chunk_hashes(self, chunk_hashes: list[str]) -> None:
        """Delete specific chunk points matching chunk hashes."""
        ...

    def search(
        self,
        query_vector: list[float],
        limit: int,
        payload_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedContext]:
        """Execute approximate nearest neighbor search with optional payload filter."""
        ...

    def count(self) -> int:
        """Return total number of vectors in collection."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Local dense embedding model runtime."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compute normalized dense embeddings for a batch of text passages."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Compute normalized dense embedding for a single query."""
        ...

    @property
    def dimension(self) -> int:
        """Return embedding vector dimensionality."""
        ...


@runtime_checkable
class SparseIndex(Protocol):
    """Inverted lexical BM25 index."""

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Build or update BM25 index with chunks."""
        ...

    def search(self, query: str, limit: int) -> list[RetrievedChunk]:
        """Execute exact BM25 keyword matching and return ranked chunks."""
        ...

    def clear(self) -> None:
        """Clear the in-memory inverted index."""
        ...

    @property
    def count(self) -> int:
        """Return number of documents indexed."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Local cross-encoder scoring runtime (Stage 2)."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Re-rank candidate chunks using full cross-attention scoring."""
        ...


@runtime_checkable
class LLMGenerator(Protocol):
    """Local Ollama generation adapter."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Execute complete non-streaming completion."""
        ...

    def stream(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
        """Stream generated response tokens."""
        ...


@runtime_checkable
class Chunker(Protocol):
    """Hierarchical Small-to-Big markdown and text splitter."""

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split document into small search chunks with heading paths."""
        ...

    def extract_sections(self, document: Document) -> list[Section]:
        """Extract parent section hierarchy and content from document."""
        ...


@runtime_checkable
class HybridSearcher(Protocol):
    """Pure search engine orchestrating dense, sparse, RRF, and re-ranking."""

    def search(
        self,
        query: str,
        top_k: int | None = None,
        payload_filter: Mapping[str, object] | None = None,
    ) -> SearchResult:
        """Execute decoupled hybrid search pipeline in sub-200ms without LLM."""
        ...


@runtime_checkable
class DocumentReader(Protocol):
    """Local document reader and directory scanner interface."""

    def read_file(self, file_path: Path | str) -> Document:
        """Parse a single file into a canonical Document model."""
        ...

    def scan_directory(self, directory_path: Path | str, recursive: bool = True) -> list[Document]:
        """Discover and parse markdown documents in a directory."""
        ...
