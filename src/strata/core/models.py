"""Pure domain models for the Strata air-gapped knowledge engine.

All entities exchanged across module boundaries are immutable, validated Pydantic v2
models without any database or I/O framework dependencies.

DATA FLOW LIFECYCLE:
====================

[Ingestion Path]
  Raw Document File
         │
         ▼
    Document ──────────────► (Persisted to SQLite ACID canonical store)
         │
         ▼ (Hierarchical split)
     Section ──────────────► (Persisted to SQLite for parent context)
         │
         ▼ (Granular search split)
      Chunk  ──────────────► (Hashed via SHA-256 over doc_id:heading:text)
         │
         ▼ (Diff against SQLite indexed hashes)
    ChunkDiff ─────────────► (Reconciles new, unchanged, and orphaned chunks)
         │
         ▼ (Sync complete)
    SyncStats ─────────────► (Audit summary and execution telemetry)

[Retrieval Path (Zero LLM)]
    User Query
         │
         ├──────────────────────────────┐
         ▼                              ▼
    Qdrant Dense ANN             BM25 Lexical Index
         │                              │
         └──────────────┬───────────────┘
                        ▼
            Reciprocal Rank Fusion (RRF)
                        ▼
         Cross-Encoder Re-ranker (bge-reranker-base)
                        ▼
                 RetrievedChunk  (Hydrates parent context from SQLite Section)
                        │
                        ▼
                  SearchResult   (Ranked candidate chunks + stage latencies)

[Synthesis Path (Local LLM)]
    SearchResult + Grounding Prompt
         │
         ▼
     Local LLM (Ollama)
         │
         ├──────────────────────────────┐
         ▼                              ▼
     Citation                       AskResult
   [Doc: title, § Heading]        (Synthesized answer + token metrics)
"""

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

# ==============================================================================
# 1. CANONICAL STORAGE & HIERARCHICAL INGESTION (SQLite Source of Truth)
#    Entities representing raw text, parent sections, and search slices.
# ==============================================================================
# - Document
# - Section
# - Chunk


class Document(BaseModel):
    """Canonical document stored in SQLite ACID primary store.

    Represents an ingested text or Markdown document before chunking.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Unique UUID identifier for document.",
    )
    title: str = Field(
        description="Document title or sanitized filename.",
    )
    file_path: str = Field(
        description="Original source file path or URI.",
    )
    raw_content: str = Field(
        description="Unprocessed raw document text.",
    )
    doc_hash: str = Field(
        description="SHA-256 cryptographic hash of raw_content.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="User-defined or taxonomy tags for metadata pre-filtering.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary metadata key-value pairs (e.g., author, source, format).",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of canonical ingestion.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of last update.",
    )


class Section(BaseModel):
    """Document section representing heading hierarchy and parent context.

    Used by the hierarchical Small-to-Big retrieval mechanism to hydrate
    rich context for matched small child chunks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Unique section identifier.",
    )
    document_id: str = Field(
        description="Foreign key to parent Document.",
    )
    title: str = Field(
        description="Heading title text.",
    )
    level: int = Field(
        ge=1,
        le=6,
        description="Markdown heading level (1 for #, 2 for ##, etc.).",
    )
    heading_path: str = Field(
        description="Hierarchical breadcrumb path (e.g., 'Architecture > Raft > Leader Election').",
    )
    content: str = Field(
        description="Full text content of this section (parent context).",
    )
    index: int = Field(
        ge=0,
        description="0-based sequence order of section in document.",
    )
    token_count: int = Field(
        default=0,
        ge=0,
        description="Estimated token count of the section content.",
    )


class Chunk(BaseModel):
    """Granular search chunk for dense vector and sparse lexical indexing.

    Content-addressable via SHA-256 hash over doc_id:heading_path:text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Unique chunk identifier.",
    )
    document_id: str = Field(
        description="Foreign key to canonical Document in SQLite.",
    )
    chunk_hash: str = Field(
        description="SHA-256 hash computed over f'{document_id}:{heading_path}:{text}'.",
    )
    text: str = Field(
        description="Search chunk text slice (250-400 tokens).",
    )
    heading_path: str = Field(
        description="Hierarchical heading breadcrumb path.",
    )
    index: int = Field(
        ge=0,
        description="Sequential chunk index within parent document.",
    )
    parent_section_id: str | None = Field(
        default=None,
        description="Foreign key to parent Section in SQLite for Small-to-Big context expansion.",
    )
    token_count: int = Field(
        default=0,
        ge=0,
        description="Estimated token count of this chunk.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags inherited from parent document for vector payload pre-filtering.",
    )


# ==============================================================================
# 2. DIFFERENTIAL SYNCHRONIZATION ENGINE
#    Models governing chunk reconciliation, hashing deltas, and sync telemetry.
# ==============================================================================
# - ChunkDiff
# - SyncStats


class ChunkDiff(BaseModel):
    """Differential chunk reconciliation result for a single document.

    Computes delta between newly split chunks and active chunks in SQLite.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(
        description="Target document ID being reconciled.",
    )
    new_chunks: list[Chunk] = Field(
        default_factory=list,
        description="Newly added or modified chunks requiring vector embedding.",
    )
    unchanged_hashes: list[str] = Field(
        default_factory=list,
        description="Hashes of chunks unchanged in SQLite and Qdrant (0ms compute cost).",
    )
    orphaned_hashes: list[str] = Field(
        default_factory=list,
        description="Hashes of deleted chunks to purge from SQLite and Qdrant.",
    )


class SyncStats(BaseModel):
    """Summary statistics from a differential sync operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_documents: int = Field(
        default=0,
        ge=0,
        description="Total documents processed during sync.",
    )
    total_chunks: int = Field(
        default=0,
        ge=0,
        description="Total chunks currently indexed across documents.",
    )
    new_chunks: int = Field(
        default=0,
        ge=0,
        description="Number of new chunks embedded and upserted.",
    )
    unchanged_chunks: int = Field(
        default=0,
        ge=0,
        description="Number of chunks that bypassed re-embedding.",
    )
    orphaned_chunks: int = Field(
        default=0,
        ge=0,
        description="Number of orphaned chunks purged from indices.",
    )
    elapsed_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total execution time of synchronization in milliseconds.",
    )


# ==============================================================================
# 3. DECOUPLED RETRIEVAL & RE-RANKING PIPELINE (Zero LLM)
#    Outputs produced by dense search, BM25, RRF fusion, and cross-encoders.
# ==============================================================================
# - RetrievedChunk (RetrievedContext)
# - SearchResult


class RetrievedChunk(BaseModel):
    """Ranked chunk returned by retrieval, fusion, or re-ranking.

    Represents a search hit with detailed scoring provenance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk: Chunk = Field(
        description="Underlying granular chunk domain object.",
    )
    score: float = Field(
        description="Final assigned relevance score.",
    )
    dense_rank: int | None = Field(
        default=None,
        description="1-based ranking from dense vector search.",
    )
    sparse_rank: int | None = Field(
        default=None,
        description="1-based ranking from BM25 sparse lexical search.",
    )
    dense_score: float | None = Field(
        default=None,
        description="Raw cosine similarity score from dense search.",
    )
    sparse_score: float | None = Field(
        default=None,
        description="Raw lexical BM25 score.",
    )
    rrf_score: float | None = Field(
        default=None,
        description="Reciprocal Rank Fusion score before cross-encoder re-ranking.",
    )
    rerank_score: float | None = Field(
        default=None,
        description="Cross-encoder re-ranking score (bge-reranker-base).",
    )
    parent_context: str | None = Field(
        default=None,
        description="Hydrated parent section text from SQLite for Small-to-Big synthesis.",
    )
    document_title: str | None = Field(
        default=None,
        description="Parent document title for human display and citations.",
    )


# Alias for compatibility with protocol definitions
RetrievedContext = RetrievedChunk


class SearchResult(BaseModel):
    """Decoupled search output containing retrieved chunks and stage latencies.

    Produced in sub-200ms without touching any generative LLM.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(
        description="Original query string.",
    )
    chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Ranked retrieved chunks after RRF and optional re-ranking.",
    )
    dense_count: int = Field(
        default=0,
        ge=0,
        description="Number of candidates returned from dense vector index.",
    )
    sparse_count: int = Field(
        default=0,
        ge=0,
        description="Number of candidates returned from BM25 sparse index.",
    )
    total_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="End-to-end retrieval latency in milliseconds.",
    )
    dense_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Dense vector search latency in milliseconds.",
    )
    sparse_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Sparse BM25 search latency in milliseconds.",
    )
    rrf_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Reciprocal Rank Fusion latency in milliseconds.",
    )
    rerank_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Cross-encoder re-ranking latency in milliseconds.",
    )


# ==============================================================================
# 4. LOCAL LLM SYNTHESIS & STRICT GROUNDING (Ollama Consumer)
#    Synthesized responses, token throughput metrics, and verifiable citations.
# ==============================================================================
# - Citation
# - AskResult


class Citation(BaseModel):
    """Citation linking synthesized statements back to canonical SQLite sources.

    Strictly formatted as `[Doc: <title>, § <heading>]`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_title: str = Field(
        description="Source document title.",
    )
    heading_path: str = Field(
        description="Hierarchical section heading.",
    )
    chunk_id: str = Field(
        description="Source chunk ID in SQLite.",
    )
    source_index: int = Field(
        ge=1,
        description="1-based index of source in prompt context excerpts.",
    )

    def format_tag(self) -> str:
        """Return standardized citation string e.g. [Doc: title, § heading]."""
        return f"[Doc: {self.document_title}, § {self.heading_path}]"


class AskResult(BaseModel):
    """Complete answer synthesized by local Ollama model with strict grounding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(
        description="User question.",
    )
    answer: str = Field(
        description="Grounded synthesized response text.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Extracted grounding citations.",
    )
    sources: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Retrieved chunks supplied in context prompt.",
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total synthesis latency in milliseconds.",
    )
    time_to_first_token_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Time to first streamed token in milliseconds.",
    )
    tokens_generated: int = Field(
        default=0,
        ge=0,
        description="Number of tokens generated by local LLM.",
    )
    tokens_per_second: float = Field(
        default=0.0,
        ge=0.0,
        description="Generation throughput in tokens per second.",
    )


# ==============================================================================
# 5. AUTONOMOUS AGENT NAVIGATION PRIMITIVES
#    Typed tools for structural inspection and graph-like sibling chunk traversal.
# ==============================================================================
# - DocumentOutlineItem
# - DocumentOutline
# - ExpandedContext


class DocumentOutlineItem(BaseModel):
    """Single entry in a document structural outline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: int = Field(
        ge=1,
        le=6,
        description="Heading level depth.",
    )
    title: str = Field(
        description="Heading title.",
    )
    heading_path: str = Field(
        description="Full breadcrumb path.",
    )
    section_id: str = Field(
        description="Section UUID in SQLite.",
    )
    token_count: int = Field(
        default=0,
        ge=0,
        description="Estimated token count of section.",
    )


class DocumentOutline(BaseModel):
    """Structural outline of a document for agent inspection without loading full body."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(
        description="Target document UUID.",
    )
    document_title: str = Field(
        description="Document title.",
    )
    items: list[DocumentOutlineItem] = Field(
        default_factory=list,
        description="Ordered list of structural headings.",
    )


class ExpandedContext(BaseModel):
    """Context expanded by traversing adjacent sibling chunks in SQLite."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_chunk_id: str = Field(
        description="Identifier of central chunk.",
    )
    document_id: str = Field(
        description="Parent document UUID.",
    )
    window: int = Field(
        ge=1,
        description="Window size of siblings retrieved (preceding and succeeding).",
    )
    preceding_chunks: list[Chunk] = Field(
        default_factory=list,
        description="Chronologically preceding sibling chunks.",
    )
    target_chunk: Chunk = Field(
        description="Central target chunk.",
    )
    succeeding_chunks: list[Chunk] = Field(
        default_factory=list,
        description="Chronologically succeeding sibling chunks.",
    )
    combined_text: str = Field(
        description="Concatenated text of preceding, target, and succeeding chunks.",
    )


# ==============================================================================
# 6. EVALUATION & RAGAS BENCHMARK HARNESS
#    Test-set samples and automated scorecard reporting for regression tracking.
# ==============================================================================
# - BenchmarkSample
# - BenchmarkScorecard


class BenchmarkSample(BaseModel):
    """Single evaluation triple for offline local RAGAS evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(
        description="Evaluation question.",
    )
    ground_truth: str = Field(
        description="Ground truth verified answer.",
    )
    contexts: list[str] = Field(
        default_factory=list,
        description="Ground truth context passages.",
    )


class BenchmarkScorecard(BaseModel):
    """Output evaluation metrics from local RAGAS benchmark runner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    retrieval_strategy: str = Field(
        description="Strategy label (e.g., 'Hybrid (RRF) + bge-reranker-base').",
    )
    reranker_model: str | None = Field(
        default=None,
        description="Re-ranker model identifier if enabled.",
    )
    context_precision: float = Field(
        ge=0.0,
        le=1.0,
        description="RAGAS context precision score.",
    )
    context_recall: float = Field(
        ge=0.0,
        le=1.0,
        description="RAGAS context recall score.",
    )
    faithfulness: float = Field(
        ge=0.0,
        le=1.0,
        description="RAGAS generation faithfulness score (no hallucination).",
    )
    answer_relevance: float = Field(
        ge=0.0,
        le=1.0,
        description="RAGAS answer relevance score.",
    )
    avg_retrieval_latency_ms: float = Field(
        ge=0.0,
        description="Average hybrid retrieval latency in milliseconds.",
    )
    sample_count: int = Field(
        ge=1,
        description="Number of evaluation samples in benchmark run.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of scorecard generation.",
    )
