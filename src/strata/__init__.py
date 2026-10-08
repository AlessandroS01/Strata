"""Strata: Offline-first, air-gapped personal knowledge engine.

Architected with domain-driven design, immutable data contracts,
and zero-daemon local storage.
"""

from . import core, indexing, ingestion

__version__ = "0.1.0"

__all__ = ["__version__", "core", "indexing", "ingestion"]
