"""Configuration management for Strata using Pydantic Settings.

Provides strongly typed, validated settings loaded from environment variables
or .env files with sensible air-gapped defaults.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings for Strata air-gapped knowledge engine.

    All settings can be overridden via environment variables prefixed with `STRATA_`
    or via a `.env` file in the root workspace.
    """

    model_config = SettingsConfigDict(
        env_prefix="STRATA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # -------------------------------------------------------------------------
    # Canonical Storage & Derived Index Paths
    # -------------------------------------------------------------------------
    data_dir: Path = Field(
        default=Path("./data"),
        description="Root directory for local data persistence.",
    )
    db_path: Path = Field(
        default=Path("./data/canonical.db"),
        description="Path to canonical SQLite database (source of truth).",
    )
    qdrant_path: Path = Field(
        default=Path("./data/qdrant"),
        description="Filesystem directory for embedded Qdrant vector index.",
    )

    # -------------------------------------------------------------------------
    # Dense Vector Index & Embeddings
    # -------------------------------------------------------------------------
    qdrant_collection_name: str = Field(
        default="strata_chunks",
        description="Name of the Qdrant vector collection.",
    )
    distance_metric: str = Field(
        default="Cosine",
        description="Vector distance metric for nearest neighbor search.",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Local dense embedding model name (FastEmbed / Sentence-Transformers).",
    )
    embedding_dimension: int = Field(
        default=384,
        ge=1,
        description="Dense embedding vector dimension.",
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for generating embeddings.",
    )

    # -------------------------------------------------------------------------
    # Sparse Retrieval (BM25)
    # -------------------------------------------------------------------------
    sparse_top_k: int = Field(
        default=20,
        ge=1,
        description="Candidate chunks retrieved by BM25 sparse index.",
    )
    bm25_b: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 document length normalization parameter.",
    )
    bm25_k1: float = Field(
        default=1.5,
        ge=0.0,
        description="BM25 term frequency saturation parameter.",
    )

    # -------------------------------------------------------------------------
    # Hybrid Retrieval & Reciprocal Rank Fusion (RRF)
    # -------------------------------------------------------------------------
    dense_top_k: int = Field(
        default=20,
        ge=1,
        description="Candidate chunks retrieved by dense vector search.",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="Reciprocal Rank Fusion smoothing constant k.",
    )
    fused_top_k: int = Field(
        default=20,
        ge=1,
        description="Top candidates after RRF fusion routed to re-ranker.",
    )

    # -------------------------------------------------------------------------
    # Cross-Encoder Re-ranking
    # -------------------------------------------------------------------------
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="Local cross-encoder re-ranking model.",
    )
    reranker_batch_size: int = Field(
        default=16,
        ge=1,
        description="Batch size for re-ranking cross-encoder scoring.",
    )
    final_top_k: int = Field(
        default=3,
        ge=1,
        description="Final number of re-ranked chunks returned for synthesis or search.",
    )

    # -------------------------------------------------------------------------
    # Hierarchical Small-to-Big Chunking
    # -------------------------------------------------------------------------
    chunk_size: int = Field(
        default=350,
        ge=50,
        description="Target chunk token length (250-400 tokens).",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Token overlap between adjacent chunks.",
    )
    min_chunk_size: int = Field(
        default=50,
        ge=10,
        description="Minimum tokens for a chunk to be preserved.",
    )

    # -------------------------------------------------------------------------
    # Local LLM Inference (Ollama)
    # -------------------------------------------------------------------------
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Local Ollama endpoint URL.",
    )
    llm_model: str = Field(
        default="qwen3.5:0.8b",
        description="Default local LLM model tag.",
    )
    llm_timeout: float = Field(
        default=120.0,
        ge=1.0,
        description="Request timeout for LLM inference in seconds.",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for grounded generation.",
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=1,
        description="Maximum tokens to generate.",
    )

    # -------------------------------------------------------------------------
    # Evaluation & Benchmarking
    # -------------------------------------------------------------------------
    eval_judge_model: str = Field(
        default="qwen2.5:7b-instruct",
        description="Local judge model for automated RAGAS benchmarks.",
    )
    eval_dataset_path: Path = Field(
        default=Path("./tests/eval/golden_dataset.json"),
        description="Default path to golden evaluation dataset.",
    )
    eval_output_path: Path = Field(
        default=Path("./BENCHMARK.md"),
        description="Output path for benchmark scorecard markdown.",
    )

    # -------------------------------------------------------------------------
    # General & Logging
    # -------------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Structured logging level (DEBUG, INFO, WARNING, ERROR).",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton instance of Strata Settings."""
    return Settings()
