"""Content-addressable cryptographic hashing engine for Strata.

Provides deterministic SHA-256 hashing for documents and chunks to support
content-addressable differential invalidation and sync reconciliation.
"""

import hashlib


def compute_chunk_hash(heading_path: str, chunk_text: str) -> str:
    """Compute deterministic SHA-256 chunk hash over heading path and text body."""
    return hashlib.sha256(f"{heading_path}:{chunk_text}".encode()).hexdigest()


def compute_document_hash(raw_content: str | bytes) -> str:
    """Compute deterministic SHA-256 hash over raw document content or bytes."""
    data = raw_content if isinstance(raw_content, bytes) else raw_content.encode("utf-8")
    return hashlib.sha256(data).hexdigest()
