"""Local FastEmbed dense embedding adapter for Strata.

Wraps the FastEmbed TextEmbedding ONNX runtime to compute normalized dense
vector embeddings for documents and search queries, satisfying the Embedder protocol.
"""

import logging
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding

from strata.core.config import Settings, get_settings
from strata.core.exceptions import ModelInferenceError
from strata.core.protocols import Embedder

logger = logging.getLogger(__name__)


class FastEmbedder(Embedder):
    """Local dense vector embedder backed by FastEmbed ONNX runtime.

    Implements the Embedder protocol with lazy-loaded model weights,
    graceful handling of empty inputs, and shielded exception wrapping.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        model_name: str | None = None,
        cache_dir: str | Path | None = None,
        threads: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self.model_name: str = model_name or self.settings.embedding_model
        self.cache_dir: str | None = str(cache_dir) if cache_dir is not None else None
        self.threads: int | None = threads
        self.batch_size: int = batch_size or self.settings.embedding_batch_size
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        """Lazy-loaded singleton instance of FastEmbed TextEmbedding."""
        if self._model is None:
            self._model = self._init_model()
        return self._model

    def _init_model(self) -> TextEmbedding:
        """Initialize the underlying TextEmbedding model with shielded error handling."""
        try:
            logger.debug(
                "Initializing FastEmbed TextEmbedding model '%s' (threads=%s)",
                self.model_name,
                self.threads,
            )
            return TextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                threads=self.threads,
            )
        except Exception as exc:
            raise ModelInferenceError(
                f"Failed to initialize FastEmbed model '{self.model_name}': {exc}"
            ) from exc

    @property
    def dimension(self) -> int:
        """Return embedding vector dimensionality configured in settings."""
        return self.settings.embedding_dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compute normalized dense embeddings for a batch of text passages.

        Args:
            texts: List of text passages to embed.

        Returns:
            List of float embedding vectors matching input texts length.
        """
        if not texts:
            return []

        try:
            model = self.model
            # Prefer passage_embed if present, fallback to embed
            embed_fn: Any = getattr(model, "passage_embed", getattr(model, "embed", None))
            if embed_fn is None:
                raise AttributeError("Model has neither 'passage_embed' nor 'embed' method")

            raw_embeddings = embed_fn(texts, batch_size=self.batch_size)
            result: list[list[float]] = []
            for vec in raw_embeddings:
                if hasattr(vec, "tolist"):
                    float_list: list[float] = [float(x) for x in vec.tolist()]
                else:
                    float_list = [float(x) for x in vec]
                result.append(float_list)

            logger.debug("Computed dense embeddings for %d texts", len(result))
            return result
        except ModelInferenceError:
            raise
        except Exception as exc:
            raise ModelInferenceError(f"Failed to compute text embeddings: {exc}") from exc

    def embed_query(self, query: str) -> list[float]:
        """Compute normalized dense embedding for a single query.

        Args:
            query: Query string to embed.

        Returns:
            Float vector embedding for the query.
        """
        try:
            model = self.model
            # Prefer query_embed if present, fallback to embed
            if hasattr(model, "query_embed"):
                raw_embeddings = model.query_embed(query)
            elif hasattr(model, "embed"):
                raw_embeddings = model.embed([query], batch_size=1)
            else:
                raise AttributeError("Model has neither 'query_embed' nor 'embed' method")

            first = next(iter(raw_embeddings), None)
            if first is None:
                return [0.0] * self.dimension

            if hasattr(first, "tolist"):
                result: list[float] = [float(x) for x in first.tolist()]
            else:
                result = [float(x) for x in first]

            logger.debug("Computed dense query embedding (dimension=%d)", len(result))
            return result
        except ModelInferenceError:
            raise
        except Exception as exc:
            raise ModelInferenceError(f"Failed to compute query embedding: {exc}") from exc
