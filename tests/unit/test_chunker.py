"""Unit tests for the Hierarchical Markdown and Text Chunker.

Validates:
- Protocol compliance with Chunker protocol.
- Hyperparameter handling and custom settings injection.
- Section extraction and heading ancestor hierarchy.
- Code block fence handling (ignoring comments).
- Fallback for documents without headings and empty documents.
- Small-to-Big chunking, min_chunk_size filtering, and oversized splitting.
- Deterministic SHA-256 chunk hashing.
- Foreign key linking between Chunk.parent_section_id and Section.id.
- Sequential index ordering across sections and chunks.
"""

import hashlib
import re

import pytest

from strata.core.config import Settings
from strata.core.models import Document
from strata.core.protocols import Chunker
from strata.indexing.chunker import (
    MarkdownChunker,
    count_tokens,
    get_token_encoder,
)
from strata.indexing.hasher import compute_chunk_hash, compute_document_hash


def make_document(
    doc_id: str = "doc-test-1",
    title: str = "Distributed Consensus",
    raw_content: str = "",
    tags: list[str] | None = None,
) -> Document:
    """Helper to construct a valid Document domain model."""
    doc_hash = hashlib.sha256(raw_content.encode()).hexdigest()
    return Document(
        id=doc_id,
        title=title,
        file_path=f"/path/to/{title.lower().replace(' ', '_')}.md",
        raw_content=raw_content,
        doc_hash=doc_hash,
        tags=tags if tags is not None else ["systems", "consensus"],
    )


# ==============================================================================
# 1. Protocol Compliance & Initialization
# ==============================================================================


def test_chunker_protocol_compliance() -> None:
    """Verify MarkdownChunker conforms to the runtime Chunker protocol."""
    chunker = MarkdownChunker()
    assert isinstance(chunker, Chunker)


def test_chunker_initialization_defaults() -> None:
    """Verify MarkdownChunker initializes with default Settings hyperparameters."""
    chunker = MarkdownChunker()
    assert chunker.chunk_size == 350
    assert chunker.chunk_overlap == 50
    assert chunker.min_chunk_size == 50


def test_chunker_custom_settings_injection() -> None:
    """Verify custom Settings object can be injected into MarkdownChunker."""
    custom_settings = Settings(
        chunk_size=200,
        chunk_overlap=30,
        min_chunk_size=40,
    )
    chunker = MarkdownChunker(settings=custom_settings)
    assert chunker.chunk_size == 200
    assert chunker.chunk_overlap == 30
    assert chunker.min_chunk_size == 40


def test_chunker_direct_parameter_overrides() -> None:
    """Verify constructor parameters override Settings hyperparameters."""
    chunker = MarkdownChunker(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=15,
    )
    assert chunker.chunk_size == 100
    assert chunker.chunk_overlap == 20
    assert chunker.min_chunk_size == 15


@pytest.mark.parametrize(
    ("size", "overlap", "min_size", "err_msg"),
    [
        (0, 0, 0, "chunk_size must be >= 1"),
        (-10, 0, 0, "chunk_size must be >= 1"),
        (100, -5, 10, "chunk_overlap must be >= 0"),
        (100, 10, -1, "min_chunk_size must be >= 0"),
        (100, 100, 10, "chunk_overlap (100) must be strictly less than chunk_size (100)"),
        (100, 120, 10, "chunk_overlap (120) must be strictly less than chunk_size (100)"),
    ],
)
def test_chunker_invalid_hyperparameters(
    size: int, overlap: int, min_size: int, err_msg: str
) -> None:
    """Verify validation errors for illegal chunking hyperparameters."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        MarkdownChunker(chunk_size=size, chunk_overlap=overlap, min_chunk_size=min_size)


# ==============================================================================
# 2. Section Extraction & Hierarchy
# ==============================================================================


def test_extract_sections_nested_headings() -> None:
    """Verify hierarchical breadcrumb path computation across nested headings."""
    content = """# Architecture Overview
High-level architectural overview.

## Storage Layer
Storage engine details.

### SQLite Store
ACID relational source of truth.

### Qdrant Store
Embedded vector index.

## Retrieval Pipeline
Hybrid search and ranking.

# Operational Runbooks
Deployment guidelines.
"""
    doc = make_document(raw_content=content)
    chunker = MarkdownChunker()
    sections = chunker.extract_sections(doc)

    assert len(sections) == 6

    # Verify sequential indices
    assert [s.index for s in sections] == [0, 1, 2, 3, 4, 5]

    # Verify levels and heading paths
    assert sections[0].title == "Architecture Overview"
    assert sections[0].level == 1
    assert sections[0].heading_path == "Architecture Overview"
    assert "High-level architectural overview." in sections[0].content

    assert sections[1].title == "Storage Layer"
    assert sections[1].level == 2
    assert sections[1].heading_path == "Architecture Overview > Storage Layer"

    assert sections[2].title == "SQLite Store"
    assert sections[2].level == 3
    assert sections[2].heading_path == "Architecture Overview > Storage Layer > SQLite Store"

    assert sections[3].title == "Qdrant Store"
    assert sections[3].level == 3
    assert sections[3].heading_path == "Architecture Overview > Storage Layer > Qdrant Store"

    assert sections[4].title == "Retrieval Pipeline"
    assert sections[4].level == 2
    assert sections[4].heading_path == "Architecture Overview > Retrieval Pipeline"

    assert sections[5].title == "Operational Runbooks"
    assert sections[5].level == 1
    assert sections[5].heading_path == "Operational Runbooks"


def test_extract_sections_ignores_code_block_fences() -> None:
    """Verify headings and # comments inside fenced code blocks are not treated as headings."""
    content = """# Developer Guide
Getting started with the codebase.

```python
# This is a Python comment, NOT a markdown heading
def raft_vote():
    # Another comment inside code
    return True
```

~~~bash
# Bash comment inside tilde fence
uv run strata sync
~~~

## Verification
Run tests to verify behavior.
"""
    doc = make_document(raw_content=content)
    chunker = MarkdownChunker()
    sections = chunker.extract_sections(doc)

    assert len(sections) == 2
    assert sections[0].title == "Developer Guide"
    assert sections[0].level == 1
    assert sections[0].heading_path == "Developer Guide"
    assert "def raft_vote():" in sections[0].content
    assert "uv run strata sync" in sections[0].content

    assert sections[1].title == "Verification"
    assert sections[1].level == 2
    assert sections[1].heading_path == "Developer Guide > Verification"
    assert "Run tests to verify behavior." in sections[1].content


def test_extract_sections_preamble_content() -> None:
    """Verify non-empty introductory text before the first heading forms a section."""
    content = """This is preamble text introducing the document before any heading.
Second line of preamble.

# Chapter 1
Content of chapter 1.
"""
    doc = make_document(title="Doc With Preamble", raw_content=content)
    chunker = MarkdownChunker()
    sections = chunker.extract_sections(doc)

    assert len(sections) == 2

    # Preamble section
    assert sections[0].index == 0
    assert sections[0].title == "Doc With Preamble"
    assert sections[0].level == 1
    assert sections[0].heading_path == "Doc With Preamble"
    assert "This is preamble text" in sections[0].content

    # First heading
    assert sections[1].index == 1
    assert sections[1].title == "Chapter 1"
    assert sections[1].level == 1
    assert sections[1].heading_path == "Chapter 1"


def test_extract_sections_no_headings_fallback() -> None:
    """Verify documents without markdown headings fall back to document title."""
    content = "Plain text document with no markdown headers.\nSecond line of plain text."
    doc = make_document(title="Unstructured Note", raw_content=content)
    chunker = MarkdownChunker()
    sections = chunker.extract_sections(doc)

    assert len(sections) == 1
    sec = sections[0]
    assert sec.index == 0
    assert sec.document_id == doc.id
    assert sec.title == "Unstructured Note"
    assert sec.level == 1
    assert sec.heading_path == "Unstructured Note"
    assert sec.content == content
    assert sec.token_count > 0


def test_extract_sections_empty_document() -> None:
    """Verify empty document produces a fallback section with zero tokens."""
    doc = make_document(title="Empty Document", raw_content="")
    chunker = MarkdownChunker()
    sections = chunker.extract_sections(doc)

    assert len(sections) == 1
    assert sections[0].title == "Empty Document"
    assert sections[0].content == ""
    assert sections[0].token_count == 0


def test_extract_sections_level_skipping() -> None:
    """Verify heading level jumping (H1 directly to H4) builds valid hierarchy."""
    content = """# Root Level
Root body.

#### Deep Child
Deep child body.

# Next Root
Next root body.
"""
    doc = make_document(raw_content=content)
    chunker = MarkdownChunker()
    sections = chunker.extract_sections(doc)

    assert len(sections) == 3
    assert sections[0].heading_path == "Root Level"
    assert sections[1].heading_path == "Root Level > Deep Child"
    assert sections[2].heading_path == "Next Root"


# ==============================================================================
# 3. Small-to-Big Chunking Strategy
# ==============================================================================


def test_chunk_document_short_sections_retained() -> None:
    """Verify sections meeting min_chunk_size are retained as single atomic chunks."""
    content = """# Section 1
This is a concise section that contains enough tokens to satisfy the minimum chunk
size requirement. We want to ensure that it stays as a single chunk rather than being
split into multiple smaller chunks or discarded prematurely.

# Section 2
Here is another section that easily satisfies the minimum chunk size threshold.
It provides sufficient technical context regarding storage indices and query engines
without exceeding the maximum target chunk size.
"""
    doc = make_document(raw_content=content)
    chunker = MarkdownChunker(chunk_size=350, chunk_overlap=50, min_chunk_size=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    assert chunks[0].heading_path == "Section 1"
    assert chunks[0].index == 0
    assert chunks[1].heading_path == "Section 2"
    assert chunks[1].index == 1


def test_chunk_document_shorter_than_min_chunk_size() -> None:
    """Boundary condition: Document shorter than min_chunk_size produces zero chunks."""
    content = "Very brief phrase."  # ~3 tokens
    doc = make_document(raw_content=content)
    chunker = MarkdownChunker(min_chunk_size=50)

    # Section is extracted structurally
    sections = chunker.extract_sections(doc)
    assert len(sections) == 1
    assert sections[0].token_count < 50

    # Chunker filters it out because it fails min_chunk_size
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 0


def test_chunk_document_oversized_text_splitting() -> None:
    """Verify oversized section is split into overlapping chunks of target size."""
    # Create ~200 tokens of text
    passage = (
        "Distributed consensus ensures that a cluster of machines agrees on state. "
        "The Raft protocol organizes time into terms and elects a leader via heartbeats. "
        "Once a leader is elected, client proposals are appended to the replicated log. "
    )
    long_content = f"# Raft Consensus\n{passage * 8}"

    doc = make_document(raw_content=long_content)
    chunker = MarkdownChunker(chunk_size=100, chunk_overlap=25, min_chunk_size=20)
    chunks = chunker.chunk_document(doc)

    # Must be split into multiple chunks
    assert len(chunks) >= 3

    # Sequential index ordering
    assert [c.index for c in chunks] == list(range(len(chunks)))

    # All chunks must preserve heading path and document ID
    for c in chunks:
        assert c.heading_path == "Raft Consensus"
        assert c.document_id == doc.id
        assert c.token_count <= 100
        assert c.token_count >= 20
        assert len(c.text.strip()) > 0


def test_chunk_document_parent_section_id_links() -> None:
    """Verify all Chunk.parent_section_id links match the corresponding Section IDs."""
    content = """# Architecture
Architecture overview content with sufficient length to pass minimum chunk size.

## Storage
Detailed description of SQLite ACID canonical storage and embedded Qdrant vector index.

## Networking
Gossip protocol and RPC messaging mechanics across nodes in the cluster.
"""
    doc = make_document(raw_content=content)
    chunker = MarkdownChunker(min_chunk_size=10)

    sections = chunker.extract_sections(doc)
    section_ids = {s.id for s in sections}

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 3

    for chunk in chunks:
        assert chunk.parent_section_id is not None
        assert chunk.parent_section_id in section_ids

    # Specific parent section matching
    assert chunks[0].parent_section_id == sections[0].id
    assert chunks[1].parent_section_id == sections[1].id
    assert chunks[2].parent_section_id == sections[2].id


def test_chunk_document_tags_inheritance() -> None:
    """Verify chunks inherit tags from parent Document."""
    content = """# Document With Tags
This document has tags and sufficient text to be retained as a chunk in the index.
"""
    doc = make_document(raw_content=content, tags=["raft", "distributed", "sqlite"])
    chunker = MarkdownChunker(min_chunk_size=10)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].tags == ["raft", "distributed", "sqlite"]


def test_chunk_document_sequential_index_across_multiple_sections() -> None:
    """Verify sequential index numbering across multiple sections and split chunks."""
    content = """# First Section
Short section text that satisfies the minimum chunk size requirement.

# Second Section
First paragraph of second section with sufficient tokens to ensure it exceeds the chunk limit.
Second paragraph of second section providing extended technical discussion of indexing mechanics.
Third paragraph of second section elaborating on vector databases and sparse search algorithms.
Fourth paragraph of second section with even more detailed content explaining reciprocal rank fusion.
Fifth paragraph explaining cross-encoder re-ranking and score normalisation mechanisms in detail.

# Third Section
Third section text that also satisfies the minimum chunk size requirement.
"""
    doc = make_document(raw_content=content)
    chunker = MarkdownChunker(chunk_size=40, chunk_overlap=10, min_chunk_size=10)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 4
    # Indices must be strictly sequential 0, 1, 2, ...
    indices = [c.index for c in chunks]
    assert indices == list(range(len(chunks)))


# ==============================================================================
# 4. Deterministic Hashing & Content Addressability
# ==============================================================================


def test_chunk_hashing_deterministic_across_calls() -> None:
    """Verify chunk hashing and identifiers are 100% deterministic across multiple calls."""
    content = """# Consensus
Raft achieves consensus via leader election and log replication across quorum nodes.
"""
    doc = make_document(raw_content=content)
    chunker = MarkdownChunker(min_chunk_size=10)

    run_1 = chunker.chunk_document(doc)
    run_2 = chunker.chunk_document(doc)

    assert len(run_1) == len(run_2) == 1
    assert run_1[0].chunk_hash == run_2[0].chunk_hash
    assert run_1[0].id == run_2[0].id
    assert run_1[0].text == run_2[0].text
    assert run_1[0].parent_section_id == run_2[0].parent_section_id


def test_compute_chunk_hash_matches_sha256_specification() -> None:
    """Verify compute_chunk_hash follows the exact SHA-256 specification."""
    heading_path = "Architecture > Raft"
    text = "Leader election protocol details."
    expected_hash = hashlib.sha256(f"{heading_path}:{text}".encode()).hexdigest()

    assert compute_chunk_hash(heading_path, text) == expected_hash


def test_chunk_hash_changes_on_content_or_path_modification() -> None:
    """Verify modifying either heading path or text produces a different hash."""
    h1 = compute_chunk_hash("Overview > Storage", "SQLite database.")
    h2 = compute_chunk_hash("Overview > Storage", "Qdrant vector index.")
    h3 = compute_chunk_hash("Overview > Database", "SQLite database.")

    assert h1 != h2
    assert h1 != h3


def test_compute_document_hash_matches_sha256_specification() -> None:
    """Verify compute_document_hash produces deterministic SHA-256 for documents."""
    raw = "Raw document text content."
    expected = hashlib.sha256(raw.encode()).hexdigest()
    assert compute_document_hash(raw) == expected


# ==============================================================================
# 5. Token Counter & Utilities
# ==============================================================================


def test_count_tokens_accuracy() -> None:
    """Verify token counting accuracy and edge cases."""
    assert count_tokens("") == 0
    assert count_tokens("hello") == 1
    assert count_tokens("hello world") == 2
    assert count_tokens("Distributed consensus protocols like Raft and Paxos") >= 6


def test_get_token_encoder_cached() -> None:
    """Verify get_token_encoder returns singleton cached tokenizer."""
    enc1 = get_token_encoder()
    enc2 = get_token_encoder()
    assert enc1 is enc2


# ==============================================================================
# 6. Branch Coverage: Fallbacks and Boundary Conditions
# ==============================================================================


def test_chunk_document_skips_empty_section_content() -> None:
    """Verify sections with empty or whitespace-only content are skipped during chunking (line 218)."""
    content = """# Heading 1 Without Body
## Heading 2 Without Body
### Heading 3 With Body
This section actually contains valid content with enough tokens to meet min_chunk_size.
"""
    doc = make_document(raw_content=content)
    chunker = MarkdownChunker(min_chunk_size=10)

    sections = chunker.extract_sections(doc)
    assert len(sections) == 3
    assert sections[0].content == ""
    assert sections[1].content == ""
    assert len(sections[2].content) > 0

    chunks = chunker.chunk_document(doc)
    # The two empty sections should be skipped via line 218
    assert len(chunks) == 1
    assert (
        chunks[0].heading_path
        == "Heading 1 Without Body > Heading 2 Without Body > Heading 3 With Body"
    )
    assert chunks[0].parent_section_id == sections[2].id


def test_chunk_document_oversized_trailing_slice_shifted_backward() -> None:
    """Verify trailing slice below min_chunk_size shifts start backward (lines 276-298)."""
    enc = get_token_encoder()
    # Decode 55 distinct token IDs to ensure total tokens is ~55-56
    text = enc.decode(list(range(1000, 1055)))

    # Configure chunk_size=50, chunk_overlap=10 (stride=40), min_chunk_size=25
    # First chunk: tokens 0 to 50 (50 tokens)
    # Stride advances to 40. Remaining tokens: 40 to ~55 (15 tokens < min_chunk_size 25)
    # Triggers adjusted_start = max(0, total_tokens - 50) and lines 276-298
    content = f"# Section Header\n{text}"
    doc = make_document(raw_content=content)
    chunker = MarkdownChunker(
        chunk_size=50,
        chunk_overlap=10,
        min_chunk_size=25,
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 2
    assert chunks[0].index == 0
    assert chunks[1].index == 1
    assert chunks[0].text != chunks[1].text
    assert chunks[1].token_count >= 25


def test_count_tokens_fallback_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify count_tokens fallback to whitespace approximation when encoder fails (lines 40-43)."""

    def failing_encoder() -> None:
        raise RuntimeError("Simulated tokenizer initialization failure")

    monkeypatch.setattr("strata.indexing.chunker.get_token_encoder", failing_encoder)

    # Multi-word string fallback: max(1, int(len(words) * 1.3))
    text_multi = "alpha beta gamma delta epsilon"  # 5 words -> int(5 * 1.3) = 6
    assert count_tokens(text_multi) == 6

    # Single word string fallback: max(1, int(1 * 1.3)) = 1
    assert count_tokens("word") == 1

    # Whitespace-only string fallback (falsy words list): returns 0
    assert count_tokens("   \n\t   ") == 0


def test_encode_text_fallback_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify MarkdownChunker._encode_text fallback when encoder fails (lines 331-333)."""

    def failing_encoder() -> None:
        raise RuntimeError("Simulated tokenizer encoding failure")

    monkeypatch.setattr("strata.indexing.chunker.get_token_encoder", failing_encoder)

    chunker = MarkdownChunker()
    tokens = chunker._encode_text("first second third")
    assert tokens == [0, 1, 2]


def test_decode_tokens_fallback_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify MarkdownChunker._decode_tokens fallback when encoder fails (lines 340-341)."""

    def failing_encoder() -> None:
        raise RuntimeError("Simulated tokenizer decoding failure")

    monkeypatch.setattr("strata.indexing.chunker.get_token_encoder", failing_encoder)

    chunker = MarkdownChunker()
    decoded = chunker._decode_tokens([10, 20, 30])
    assert decoded == "10 20 30"
