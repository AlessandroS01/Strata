"""Domain exceptions for the Strata air-gapped knowledge engine.

Defines custom exceptions for canonical storage, sync reconciliation,
and local model inference errors.
"""


class StrataError(Exception):
    """Base exception for all Strata domain errors."""


class DocumentNotFoundError(StrataError):
    """Raised when a requested canonical document is not found."""


class SectionNotFoundError(StrataError):
    """Raised when a requested section is not found."""


class ChunkNotFoundError(StrataError):
    """Raised when a requested chunk is not found."""


class SyncConflictError(StrataError):
    """Raised when differential sync encounters a state conflict."""


class ModelInferenceError(StrataError):
    """Raised when local embedding, reranking, or LLM inference fails."""
