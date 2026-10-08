"""Core domain layer of the Strata knowledge engine.

Contains pure, decoupled foundation components:
- models: Immutable domain entities (Document, Section, Chunk, DocumentOutline)[cite: 1].
- protocols: Structural interfaces (PEP 544) defining system component contracts[cite: 3].
- config: Centralized application configuration and runtime settings[cite: 2].
- exceptions: Exception shielding hierarchy for canonical and engine errors[cite: 2].
- hasher: Content-addressable SHA-256 cryptographic hashing utilities[cite: 1, 3].
"""

from .config import Settings, get_settings
from .exceptions import (
    ChunkNotFoundError,
    DocumentNotFoundError,
    ModelInferenceError,
    SectionNotFoundError,
    StrataError,
    SyncConflictError,
)
from .hasher import compute_chunk_hash, compute_document_hash
from .models import (
    AskResult,
    BenchmarkSample,
    BenchmarkScorecard,
    Chunk,
    ChunkDiff,
    Citation,
    Document,
    DocumentOutline,
    DocumentOutlineItem,
    ExpandedContext,
    RetrievedChunk,
    RetrievedContext,
    SearchResult,
    Section,
    SyncStats,
)
from .protocols import (
    Chunker,
    DocumentReader,
    DocumentStore,
    Embedder,
    HybridSearcher,
    LLMGenerator,
    Reranker,
    SparseIndex,
    VectorStore,
)

__all__ = [
    "AskResult",
    "BenchmarkSample",
    "BenchmarkScorecard",
    "Chunk",
    "ChunkDiff",
    "ChunkNotFoundError",
    "Chunker",
    "Citation",
    "Document",
    "DocumentNotFoundError",
    "DocumentOutline",
    "DocumentOutlineItem",
    "DocumentReader",
    "DocumentStore",
    "Embedder",
    "ExpandedContext",
    "HybridSearcher",
    "LLMGenerator",
    "ModelInferenceError",
    "Reranker",
    "RetrievedChunk",
    "RetrievedContext",
    "SearchResult",
    "Section",
    "SectionNotFoundError",
    "Settings",
    "SparseIndex",
    "StrataError",
    "SyncConflictError",
    "SyncStats",
    "VectorStore",
    "compute_chunk_hash",
    "compute_document_hash",
    "get_settings",
]
