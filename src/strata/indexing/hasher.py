"""Content-addressable cryptographic hashing engine for Strata.

Provides deterministic SHA-256 hashing for documents and chunks to support
content-addressable differential invalidation and sync reconciliation.
"""

import hashlib


def compute_chunk_hash(heading_path: str, chunk_text: str) -> str:
    """Compute deterministic SHA-256 chunk hash over heading path and text body."""
    return hashlib.sha256(f"{heading_path}:{chunk_text}".encode()).hexdigest()


def compute_document_hash(raw_content: str) -> str:
    """Compute deterministic SHA-256 hash over raw document content."""
    return hashlib.sha256(raw_content.encode()).hexdigest()
