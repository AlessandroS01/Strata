"""Sparse lexical BM25 index adapter for Strata.

Implements the SparseIndex protocol backed by rank-bm25 (BM25Okapi),
managing tokenization, inverted indexing, differential chunk deletion,
and deterministic search scoring without external network or LLM dependencies.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from strata.core.config import Settings, get_settings
from strata.core.models import Chunk, RetrievedChunk
from strata.core.protocols import SparseIndex

logger = logging.getLogger(__name__)

# Default regex tokenizer matching words, numbers, and identifiers with hyphens/underscores
DEFAULT_TOKEN_PATTERN = r"\b[\w\-]+\b"
_COMPILED_DEFAULT_TOKEN_PATTERN = re.compile(DEFAULT_TOKEN_PATTERN)


class BM25Index(SparseIndex):
    """Inverted lexical BM25 index adapter fulfilling the SparseIndex protocol.

    Maintains tokenized texts, an internal BM25Okapi model, two-way integer-to-chunk
    mappings, and exact keyword retrieval scoring.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk] | None = None,
        settings: Settings | None = None,
        k1: float | None = None,
        b: float | None = None,
        epsilon: float = 0.25,
        token_pattern: str = DEFAULT_TOKEN_PATTERN,
        batch_size: int | None = None,
    ) -> None:
        """Initialize BM25Index.

        Args:
            chunks: Optional initial sequence of chunks to index.
            settings: Optional Strata Settings instance.
            k1: BM25 term frequency saturation parameter override.
            b: BM25 document length normalization parameter override.
            epsilon: Floor multiplier for negative IDF terms in BM25Okapi.
            token_pattern: Regular expression pattern for text tokenization.
            batch_size: Optional chunk batch size. Defaults to
                settings.embedding_batch_size.
        """
        self.settings: Settings = settings or get_settings()
        self.k1: float = k1 if k1 is not None else self.settings.bm25_k1
        self.b: float = b if b is not None else self.settings.bm25_b
        self.epsilon: float = epsilon
        self.token_pattern: str = token_pattern
        self._token_pattern: re.Pattern[str] = re.compile(token_pattern)
        self.batch_size: int = (
            batch_size if batch_size is not None else self.settings.embedding_batch_size
        )

        # Internal corpus and mappings
        self._chunks: dict[str, Chunk] = {}
        self._tokenized_cache: dict[str, list[str]] = {}
        self._corpus: list[list[str]] = []
        self._idx_to_chunk_id: list[str] = []
        self._chunk_id_to_idx: dict[str, int] = {}
        self._bm25: BM25Okapi | None = None

        if chunks:
            self.index_chunks(list(chunks))

    @staticmethod
    def default_tokenize(text: str) -> list[str]:
        """Normalize text to lowercase and tokenize using the default regex pattern."""
        if not text:
            return []
        return _COMPILED_DEFAULT_TOKEN_PATTERN.findall(text.lower())

    def tokenize(self, text: str) -> list[str]:
        """Normalize text to lowercase and tokenize using the configured pattern."""
        if not text:
            return []
        return self._token_pattern.findall(text.lower())

    @property
    def count(self) -> int:
        """Return number of documents indexed."""
        return len(self._chunks)

    def __len__(self) -> int:
        """Return number of documents indexed."""
        return self.count

    def __contains__(self, chunk_id: str) -> bool:
        """Return True if chunk_id is present in the index."""
        return chunk_id in self._chunks

    @property
    def corpus(self) -> list[list[str]]:
        """Internal list of tokenized texts."""
        return self._corpus

    @property
    def bm25(self) -> BM25Okapi | None:
        """Underlying BM25Okapi model instance."""
        return self._bm25

    @property
    def idx_to_chunk_id(self) -> list[str]:
        """Two-way mapping: integer position to chunk ID."""
        return self._idx_to_chunk_id

    @property
    def chunk_id_to_idx(self) -> dict[str, int]:
        """Two-way mapping: chunk ID to integer position."""
        return self._chunk_id_to_idx

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve indexed chunk by UUID."""
        return self._chunks.get(chunk_id)

    def get_all_chunks(self) -> list[Chunk]:
        """Return all active indexed chunks."""
        return list(self._chunks.values())

    def _rebuild_index(self) -> None:
        """Rebuild BM25Okapi model, corpus list, and two-way mappings."""
        self._idx_to_chunk_id = list(self._chunks.keys())
        self._chunk_id_to_idx = {
            chunk_id: idx for idx, chunk_id in enumerate(self._idx_to_chunk_id)
        }
        self._corpus = [self._tokenized_cache[chunk_id] for chunk_id in self._idx_to_chunk_id]

        if self._corpus and any(len(doc) > 0 for doc in self._corpus):
            self._bm25 = BM25Okapi(
                self._corpus,
                k1=self.k1,
                b=self.b,
                epsilon=self.epsilon,
            )
            # Ensure term IDFs have a positive floor so matching terms in small
            # corpora receive positive weights while non-matching terms score 0.0.
            for word, idf_val in self._bm25.idf.items():
                if idf_val <= 0.0:
                    self._bm25.idf[word] = 1e-4
        else:
            self._bm25 = None

    def index_chunks(self, chunks: list[Chunk], batch_size: int | None = None) -> None:
        """Build or update BM25 index with chunks.

        Maintains tokenized texts and two-way mappings between integer positions
        and chunk identifiers.

        Args:
            chunks: List of Chunk objects to index.
            batch_size: Optional batch size override. Defaults to self.batch_size
                (settings.embedding_batch_size).
        """
        if not chunks:
            return

        effective_batch_size = max(1, batch_size if batch_size is not None else self.batch_size)
        for i in range(0, len(chunks), effective_batch_size):
            batch = chunks[i : i + effective_batch_size]
            for chunk in batch:
                self._chunks[chunk.id] = chunk
                self._tokenized_cache[chunk.id] = self.tokenize(chunk.text)

        self._rebuild_index()
        logger.debug("Indexed %d chunks (total active: %d)", len(chunks), len(self._chunks))

    def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        """Remove chunks by their unique identifiers and update the index."""
        removed = False
        for cid in chunk_ids:
            if cid in self._chunks:
                del self._chunks[cid]
                self._tokenized_cache.pop(cid, None)
                removed = True

        if removed:
            self._rebuild_index()
            logger.debug("Deleted chunks; remaining active: %d", len(self._chunks))

    def delete_by_ids(self, chunk_ids: Sequence[str]) -> None:
        """Delete chunks by list of chunk IDs."""
        self.delete_chunks(chunk_ids)

    def delete_chunk(self, chunk_id: str) -> None:
        """Delete a single chunk by chunk ID."""
        self.delete_chunks([chunk_id])

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks belonging to a document ID."""
        to_delete = [
            chunk.id for chunk in self._chunks.values() if chunk.document_id == document_id
        ]
        if to_delete:
            self.delete_chunks(to_delete)

    def delete_by_chunk_hashes(self, chunk_hashes: Sequence[str]) -> None:
        """Delete all chunks matching the given chunk hashes."""
        hashes_set = set(chunk_hashes)
        to_delete = [chunk.id for chunk in self._chunks.values() if chunk.chunk_hash in hashes_set]
        if to_delete:
            self.delete_chunks(to_delete)

    def clear(self) -> None:
        """Clear the in-memory inverted index."""
        self._chunks.clear()
        self._tokenized_cache.clear()
        self._idx_to_chunk_id.clear()
        self._chunk_id_to_idx.clear()
        self._corpus.clear()
        self._bm25 = None
        logger.debug("BM25 index cleared.")

    def search(
        self,
        query: str,
        limit: int,
        payload_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        """Execute exact BM25 keyword matching and return ranked chunks.

        Args:
            query: User search query string.
            limit: Maximum number of ranked results to return.
            payload_filter: Optional filter dict (e.g., {'tags': ['tag1']}).

        Returns:
            List of RetrievedChunk models sorted by BM25 score descending.
        """
        limit = get_settings().sparse_top_k if limit is None else limit

        if limit <= 0:
            raise ValueError(f"Limit must be strictly positive, got {limit}")
        if self._bm25 is None or not self._corpus:
            return []

        tokens = self.tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Retain only chunks with positive relevance scores
        candidates: list[tuple[int, float]] = []
        for idx, raw_score in enumerate(scores):
            score_val = float(raw_score)
            if score_val > 0.0:
                candidates.append((idx, score_val))

        if not candidates:
            return []

        # Sort descending by score; break ties by sequential index
        candidates.sort(key=lambda item: (-item[1], item[0]))

        # Filter by tags if payload_filter specified
        required_tags: set[str] | None = None
        if payload_filter is not None and "tags" in payload_filter:
            raw_tags = payload_filter["tags"]
            if isinstance(raw_tags, (list, tuple, set)):
                required_tags = {str(t) for t in raw_tags}

        results: list[RetrievedChunk] = []
        rank = 1
        for idx, score_val in candidates:
            chunk_id = self._idx_to_chunk_id[idx]
            chunk = self._chunks[chunk_id]

            if required_tags and not required_tags.intersection(set(chunk.tags)):
                continue

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score_val,
                    sparse_score=score_val,
                    sparse_rank=rank,
                )
            )
            rank += 1
            if len(results) >= limit:
                break

        return results

    def save(self, file_path: Path | str | None = None) -> Path:
        """Save BM25 index state to a file.

        Args:
            file_path: Optional destination path (.json or .pkl). Defaults to
                self.settings.bm25_index_path if not provided.

        Returns:
            The Path where the index was saved.

        Raises:
            ValueError: If file_path is None and settings.bm25_index_path is not configured.
        """
        target = file_path if file_path is not None else self.settings.bm25_index_path
        if target is None:
            raise ValueError(
                "No file path provided and settings.bm25_index_path is not configured."
            )
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "token_pattern": self.token_pattern,
            "chunks": [chunk.model_dump(mode="json") for chunk in self._chunks.values()],
        }

        if path.suffix.lower() in (".pkl", ".pickle"):
            with open(path, "wb") as f:
                pickle.dump(data, f)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug("Saved BM25 index with %d chunks to %s", len(self._chunks), path)
        return path

    def load_from_path(self, file_path: Path | str | None = None) -> Path:
        """Load BM25 index state from a file into this instance.

        Args:
            file_path: Optional source path to load from. Defaults to
                self.settings.bm25_index_path if not provided.

        Returns:
            The Path from where the index was loaded.

        Raises:
            ValueError: If file_path is None and settings.bm25_index_path is not configured.
            FileNotFoundError: If the index file does not exist.
        """
        target = file_path if file_path is not None else self.settings.bm25_index_path
        if target is None:
            raise ValueError(
                "No file path provided and settings.bm25_index_path is not configured."
            )
        path = Path(target)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        data: dict[str, Any]
        if path.suffix.lower() in (".pkl", ".pickle"):
            with open(path, "rb") as f:
                data = pickle.load(f)
        else:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                with open(path, "rb") as f:
                    data = pickle.load(f)

        self.clear()
        self.k1 = float(data.get("k1", self.k1))
        self.b = float(data.get("b", self.b))
        self.epsilon = float(data.get("epsilon", self.epsilon))
        token_pat = str(data.get("token_pattern", self.token_pattern))
        self.token_pattern = token_pat
        self._token_pattern = re.compile(token_pat)

        raw_chunks = data.get("chunks", [])
        chunks = [Chunk.model_validate(c) for c in raw_chunks]
        self.index_chunks(chunks)
        logger.debug("Loaded BM25 index with %d chunks from %s", len(self._chunks), path)
        return path

    @classmethod
    def load(
        cls,
        file_path: Path | str | None = None,
        settings: Settings | None = None,
        batch_size: int | None = None,
    ) -> BM25Index:
        """Load and return a new BM25Index instance from a saved file.

        Args:
            file_path: Optional file path to load from. Defaults to
                settings.bm25_index_path.
            settings: Optional Settings instance.
            batch_size: Optional chunk batch size override. Defaults to
                settings.embedding_batch_size.

        Returns:
            Reconstructed BM25Index instance.

        Raises:
            ValueError: If file_path is None and settings.bm25_index_path is not configured.
            FileNotFoundError: If the index file does not exist.
        """
        resolved_settings = settings or get_settings()
        target = file_path if file_path is not None else resolved_settings.bm25_index_path
        if target is None:
            raise ValueError(
                "No file path provided and settings.bm25_index_path is not configured."
            )
        index = cls(settings=resolved_settings, batch_size=batch_size)
        index.load_from_path(target)
        return index

    @classmethod
    def from_file(
        cls,
        file_path: Path | str | None = None,
        settings: Settings | None = None,
        batch_size: int | None = None,
    ) -> BM25Index:
        """Alias for load."""
        return cls.load(file_path=file_path, settings=settings, batch_size=batch_size)
