"""Unit tests for strata.indexing.embedder (FastEmbed adapter)."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from strata.core.config import Settings
from strata.core.exceptions import ModelInferenceError
from strata.core.protocols import Embedder
from strata.indexing.embedder import FastEmbedder


def _dummy_vectors(count: int, dim: int = 384) -> Generator[np.ndarray, None, None]:
    """Yield dummy float32 numpy vector embeddings."""
    for i in range(count):
        yield np.full(dim, float(i + 1), dtype=np.float32)


def test_protocol_conformance() -> None:
    """Verify FastEmbedder inherits from and satisfies the Embedder protocol."""
    embedder = FastEmbedder()
    assert isinstance(embedder, Embedder)
    assert Embedder in FastEmbedder.__mro__


def test_dimension_property_matches_settings() -> None:
    """Verify dimension property reflects settings without initializing model."""
    embedder = FastEmbedder()
    assert embedder.dimension == 384
    # Model should not have been initialized
    assert embedder._model is None

    custom_settings = Settings(embedding_dimension=768)
    custom_embedder = FastEmbedder(settings=custom_settings)
    assert custom_embedder.dimension == 768
    assert custom_embedder._model is None


def test_lazy_loading_model_initialization() -> None:
    """Verify model is only initialized on first access and cached thereafter."""
    embedder = FastEmbedder()
    assert embedder._model is None

    mock_instance = MagicMock()
    with patch("strata.indexing.embedder.TextEmbedding", return_value=mock_instance) as mock_cls:
        # First access initializes
        model1 = embedder.model
        assert model1 is mock_instance
        mock_cls.assert_called_once_with(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=None,
            threads=None,
        )

        # Subsequent access returns cached singleton
        model2 = embedder.model
        assert model2 is mock_instance
        assert mock_cls.call_count == 1


def test_custom_parameters_passed_to_text_embedding() -> None:
    """Verify custom settings, model_name, cache_dir, and threads are forwarded."""
    custom_settings = Settings(
        embedding_model="custom/model-name",
        embedding_dimension=512,
        embedding_batch_size=16,
    )
    embedder = FastEmbedder(
        settings=custom_settings,
        cache_dir="/custom/cache",
        threads=4,
    )

    assert embedder.model_name == "custom/model-name"
    assert embedder.dimension == 512
    assert embedder.batch_size == 16

    mock_instance = MagicMock()
    with patch("strata.indexing.embedder.TextEmbedding", return_value=mock_instance) as mock_cls:
        _ = embedder.model
        mock_cls.assert_called_once_with(
            model_name="custom/model-name",
            cache_dir="/custom/cache",
            threads=4,
        )


def _fake_passage_embed(
    texts: list[str], batch_size: int = 32
) -> Generator[np.ndarray, None, None]:
    return _dummy_vectors(len(texts), 384)


def _fake_query_embed(
    query: str | list[str], **kwargs: object
) -> Generator[np.ndarray, None, None]:
    return _dummy_vectors(1, 384)


def _fake_embed(
    docs: str | list[str], batch_size: int = 256, **kwargs: object
) -> Generator[np.ndarray, None, None]:
    count = 1 if isinstance(docs, str) else len(docs)
    return _dummy_vectors(count, 384)


def test_embed_texts_returns_correct_shape_and_type() -> None:
    """Verify batch passage embedding returns standard list[list[float]] of correct dimension."""
    mock_instance = MagicMock()
    mock_instance.passage_embed.side_effect = _fake_passage_embed

    embedder = FastEmbedder()
    embedder._model = mock_instance

    texts = ["First passage about Raft.", "Second passage about Paxos.", "Third passage."]
    embeddings = embedder.embed_texts(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    for vec in embeddings:
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(val, float) for val in vec)

    mock_instance.passage_embed.assert_called_once_with(texts, batch_size=32)


def test_embed_texts_empty_list_returns_immediately() -> None:
    """Verify embed_texts([]) returns empty list without invoking model."""
    embedder = FastEmbedder()
    mock_instance = MagicMock()
    embedder._model = mock_instance

    result = embedder.embed_texts([])
    assert result == []
    mock_instance.passage_embed.assert_not_called()
    mock_instance.embed.assert_not_called()


def test_embed_query_returns_single_vector() -> None:
    """Verify embed_query returns a 1D list[float] matching model dimension."""
    mock_instance = MagicMock()
    mock_instance.query_embed.side_effect = _fake_query_embed

    embedder = FastEmbedder()
    embedder._model = mock_instance

    query_vec = embedder.embed_query("What is leader election in Raft?")

    assert isinstance(query_vec, list)
    assert len(query_vec) == 384
    assert all(isinstance(val, float) for val in query_vec)
    mock_instance.query_embed.assert_called_once_with("What is leader election in Raft?")


def test_embed_empty_strings_handled_gracefully() -> None:
    """Verify empty or whitespace strings are processed without crashing."""
    mock_instance = MagicMock()
    mock_instance.passage_embed.side_effect = _fake_passage_embed
    mock_instance.query_embed.side_effect = _fake_query_embed

    embedder = FastEmbedder()
    embedder._model = mock_instance

    # Empty string query
    q_vec = embedder.embed_query("")
    assert isinstance(q_vec, list)
    assert len(q_vec) == 384

    # Whitespace-only query
    q_ws_vec = embedder.embed_query("   ")
    assert isinstance(q_ws_vec, list)
    assert len(q_ws_vec) == 384

    # Passages with empty and whitespace strings
    p_vecs = embedder.embed_texts(["", "   ", "Valid text"])
    assert len(p_vecs) == 3
    for vec in p_vecs:
        assert len(vec) == 384


def test_embed_query_empty_generator_fallback() -> None:
    """Verify fallback to zero-vector if query embedding generator yields empty."""
    mock_instance = MagicMock()
    mock_instance.query_embed.return_value = iter([])

    embedder = FastEmbedder()
    embedder._model = mock_instance

    vec = embedder.embed_query("empty response")
    assert vec == [0.0] * 384


def test_fallback_to_embed_method_when_specialized_methods_missing() -> None:
    """Verify fallback to .embed() when passage_embed or query_embed are not present."""
    mock_instance = MagicMock(spec=["embed"])
    mock_instance.embed.side_effect = _fake_embed

    embedder = FastEmbedder()
    embedder._model = mock_instance

    # Texts
    texts = ["doc1", "doc2"]
    t_res = embedder.embed_texts(texts)
    assert len(t_res) == 2
    mock_instance.embed.assert_called_with(texts, batch_size=32)

    # Query
    q_res = embedder.embed_query("test query")
    assert len(q_res) == 384
    mock_instance.embed.assert_called_with(["test query"], batch_size=1)


def test_model_initialization_error_wrapped_in_model_inference_error() -> None:
    """Verify model instantiation exceptions raise ModelInferenceError."""
    embedder = FastEmbedder()
    with (
        patch(
            "strata.indexing.embedder.TextEmbedding", side_effect=RuntimeError("ONNX load failed")
        ),
        pytest.raises(ModelInferenceError, match="Failed to initialize FastEmbed model"),
    ):
        _ = embedder.model


def test_passage_inference_error_wrapped_in_model_inference_error() -> None:
    """Verify runtime inference errors during embed_texts raise ModelInferenceError."""
    mock_instance = MagicMock()
    mock_instance.passage_embed.side_effect = RuntimeError("GPU out of memory")

    embedder = FastEmbedder()
    embedder._model = mock_instance

    with pytest.raises(ModelInferenceError, match="Failed to compute text embeddings"):
        embedder.embed_texts(["Some document passage."])


def test_query_inference_error_wrapped_in_model_inference_error() -> None:
    """Verify runtime inference errors during embed_query raise ModelInferenceError."""
    mock_instance = MagicMock()
    mock_instance.query_embed.side_effect = RuntimeError("Tokenizer failed")

    embedder = FastEmbedder()
    embedder._model = mock_instance

    with pytest.raises(ModelInferenceError, match="Failed to compute query embedding"):
        embedder.embed_query("Search query phrase")


def test_missing_embed_methods_raises_model_inference_error() -> None:
    """Verify model lacking embedding methods raises ModelInferenceError."""
    mock_instance = MagicMock(spec=[])  # No methods

    embedder = FastEmbedder()
    embedder._model = mock_instance

    with pytest.raises(ModelInferenceError, match="Failed to compute text embeddings"):
        embedder.embed_texts(["text"])

    with pytest.raises(ModelInferenceError, match="Failed to compute query embedding"):
        embedder.embed_query("query")
