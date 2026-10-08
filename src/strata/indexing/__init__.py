"""Derived search representations and synchronization subsystem.

Transforms canonical chunks into dense and sparse search representations:
- embedder: Local dense vector generation using ONNX (FastEmbed) or PyTorch (Sentence-Transformers).
"""

from .embedder import FastEmbedder

__all__ = ["FastEmbedder"]
