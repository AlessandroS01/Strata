# GEMINI.md: Agent Context & Development Guide for Strata

This file guides AI agents operating in this repository. Follow these engineering constraints, architectural invariants, and command workflows strictly.

---

## 1. Project Mission & Core Invariants

Strata is a 100% offline, air-gapped personal knowledge base and hybrid RAG engine.

### Non-Negotiable Invariants:
1. **SQLite is the Canonical Source of Truth:**
   * Raw text, document metadata, chunk records, heading hierarchies, and SHA-256 hashes live exclusively in SQLite.
   * Qdrant and BM25 are **derived search indices**. Never write directly to Qdrant without a corresponding record in SQLite.
   * If Qdrant is wiped, running `strata sync` must deterministically reconstruct the entire vector index from SQLite.
2. **Content-Addressable Differential Invalidation:**
   * Never re-embed an entire document on update.
   * Chunks are hashed via SHA-256 (`doc_id + heading_path + text`).
   * Compare active hashes in SQLite with new hashes. Only embed newly added chunk hashes. Purge orphaned hashes from SQLite and Qdrant. Skip unchanged chunks completely.
3. **100% Offline / Air-Gapped:**
   * Zero external API calls (no OpenAI, Anthropic, or external telemetry).
   * Dense embeddings (`bge-small-en-v1.5`) and re-ranking (`bge-reranker-base`) run via local tokenizers/runtimes (`fastembed` or `sentence-transformers`).
   * Generative synthesis runs through local Ollama endpoints (`http://localhost:11434`).
4. **Decoupled Search vs. LLM:**
   * The core search engine (`strata search`) must never initialize or call an LLM. It returns raw, ranked `RetrievedChunk` dataclasses/models in sub-200ms.
   * The LLM is strictly used for synthesis (`strata ask`), evaluation judges (`strata benchmark`), or future agent reasoning loops.

---

## 2. Tech Stack & Environment

* **Language:** Python 3.12+ (strictly typed)
* **Package & Env Manager:** `uv`
* **Canonical DB:** SQLite via `sqlite3` or `SQLModel` / `SQLAlchemy` (sync/async compliant)
* **Vector Store:** `qdrant-client` in embedded local mode (`QdrantClient(path="./data/qdrant")`)
* **Sparse Index:** `rank-bm25`
* **Local Embeddings & Reranking:** `fastembed` / `sentence-transformers` (`BAAI/bge-small-en-v1.5`, `BAAI/bge-reranker-base`)
* **Local LLM Engine:** Ollama (`qwen2.5:7b-instruct` or `llama3.2:3b`) via `ollama-python` or OpenAI-compatible client
* **Evaluation:** `ragas` with local Ollama judge
* **CLI:** `typer[all]` with `rich` formatting
* **Code Quality:** `ruff` (linter/formatter), `mypy` (strict mode), `pytest`

---

## 3. Directory Layout & Module Boundaries

Maintain strict layer separation when editing or introducing files:

```text
src/strata/
├── core/             # Pure models, settings, and abstract Protocols (NO DB or IO imports)
│   ├── config.py     # Pydantic Settings (paths, models, hyperparameters)
│   ├── models.py     # Pydantic domain models (Document, Chunk, SearchResult)
│   └── protocols.py  # typing.Protocol interfaces (DocumentStore, VectorStore, etc.)
├── storage/          # Concrete storage adapters
│   ├── sqlite_store.py  # SQLite ACID operations and metadata queries
│   └── qdrant_store.py  # Embedded Qdrant CRUD, points, and payload filters
├── indexing/         # Chunking, hashing, and sync engine
│   ├── chunker.py    # Hierarchical Markdown splitter (Small-to-Big)
│   ├── hasher.py     # SHA-256 chunk hashing and diff computation
│   └── sync_worker.py# Reconciles SQLite against Qdrant/BM25
├── retrieval/        # Pure retrieval pipeline (Zero LLM)
│   ├── embedder.py   # Local embedding generator (dense)
│   ├── sparse_bm25.py# BM25 lexical inverted index
│   ├── fusion.py     # Reciprocal Rank Fusion (RRF) algorithm
│   └── reranker.py   # Cross-encoder scoring (bge-reranker-base)
├── llm/              # Generation layer (Isolated consumer)
│   ├── client.py     # Ollama client with streaming token yielders
│   └── prompts.py    # Strict grounding prompt templates & citation formatters
├── agents/           # Agent tool primitives
│   └── tools.py      # Typed tool interfaces (filter, outline, expand)
└── cli/              # Terminal entrypoints
    └── main.py       # Typer CLI application
```

---

## 4. Coding Conventions & Standards

### Type Annotations & Static Analysis
* 100% type annotations on all function signatures and class definitions.
* Avoid `Any`. Use explicit generics, `TypeVar`, or `Protocol`.
* Must pass `mypy --strict` with zero warnings.
* All domain entities exchanged across module boundaries must be immutable or validated Pydantic v2 `BaseModel` instances.

### Protocols over Inheritance
Define interfaces in `src/strata/core/protocols.py` using `typing.Protocol`. Concrete implementations in `storage/` or `retrieval/` must conform to these protocols without rigid class inheritance:

```python
from typing import Protocol
from strata.core.models import Chunk, RetrievedContext

class VectorStore(Protocol):
    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def delete_by_document_id(self, document_id: str) -> None: ...
    def delete_by_chunk_hashes(self, chunk_hashes: list[str]) -> None: ...
    def search(self, query_vector: list[float], limit: int, payload_filter: dict | None = None) -> list[RetrievedContext]: ...
```

### Error Handling & Logging
* Never use raw `print()` statements outside of the Typer presentation layer in `src/strata/cli/`.
* Use Python's standard `logging` module or `loguru` with structured contextual key-values.
* Handle database locks, missing files, and embedding timeouts gracefully with explicit custom domain exceptions (`DocumentNotFoundError`, `SyncConflictError`, `ModelInferenceError`).

---

## 5. Development Workflows & Essential Commands

When running tasks or validating code changes, use `uv run`:

### Environment Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,eval]"
```

### Code Quality & Testing Checks
Run these before completing any multi-step task:
```bash
# 1. Format and lint checks
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# 2. Strict static typing
uv run mypy --strict src/

# 3. Unit and integration test suite
uv run pytest -v tests/
```

### Running the Application Locally
```bash
# Ingest and sync notes
uv run strata add ./notes/example.md --tag test
uv run strata sync

# Fast search (No LLM, sub-200ms)
uv run strata search "query phrase" --top-k 3

# Synthesize answer with local Ollama
uv run strata ask "query phrase" --stream

# Run RAGAS benchmark
uv run strata benchmark --dataset tests/eval/golden_dataset.json
```

---

## 6. Prohibited Anti-Patterns

When modifying or generating code:
* ❌ **DO NOT** import cloud LLM SDKs (`openai`, `anthropic`, `cohere`) or make external network calls.
* ❌ **DO NOT** store full document bodies inside Qdrant point payloads. Only store `doc_id`, `chunk_id`, `heading_path`, and `tags` in Qdrant payloads. The full text body is fetched from SQLite when needed.
* ❌ **DO NOT** execute naive full re-indexing of documents on update. Always run differential chunk diffing via SHA-256 hashes.
* ❌ **DO NOT** delete records from Qdrant without updating the corresponding sync state in SQLite.
* ❌ **DO NOT** hardcode hyperparameters (chunk size, overlap, embedding model names, RRF constants). Expose them via `src/strata/core/config.py` using `pydantic-settings`.
* ❌ **DO NOT** place LLM generation logic inside retrieval or storage modules.