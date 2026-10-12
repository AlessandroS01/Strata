"""Derived search representations and synchronization subsystem.

Transforms canonical chunks into dense and sparse search representations:
- embedder: Local dense vector generation using ONNX (FastEmbed) or PyTorch (Sentence-Transformers).
"""

from .bm25_index import BM25Index
from .embedder import FastEmbedder
from .qdrant_store import QdrantVectorStore

__all__ = ["BM25Index", "FastEmbedder", "QdrantVectorStore"]
