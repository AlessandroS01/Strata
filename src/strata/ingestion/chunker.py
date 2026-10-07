"""Hierarchical Markdown and text chunker for Strata.

Implements Small-to-Big retrieval splitting:
1. Parses markdown heading hierarchies into structured parent Section entities.
2. Subdivides section bodies into granular, overlapping Chunk entities.
3. Computes deterministic SHA-256 chunk hashes for content-addressable differential invalidation.
4. Preserves heading breadcrumb paths and parent section foreign keys.
"""

import hashlib
import re
import uuid
from collections.abc import Sequence
from functools import lru_cache

import tiktoken

from strata.core.config import Settings, get_settings
from strata.core.models import Chunk, Document, Section


# Detects the start and end of code blocks in Markdown (e.g. # This is a comment).
_FENCE_PATTERN = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
# Detects Markdown headings (e.g., # Heading, ## Subheading).
_HEADING_PATTERN = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")


@lru_cache(maxsize=1)
def get_token_encoder() -> tiktoken.Encoding:
    """Return cached singleton instance of tiktoken cl100k_base tokenizer."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Accurately count tokens in text using local tiktoken tokenizer."""
    if not text:
        return 0
    try:
        encoder = get_token_encoder()
        return len(encoder.encode(text))
    except Exception:
        # Fallback to whitespace approximation (~1.3 tokens per word)
        words = text.split()
        return max(1, int(len(words) * 1.3)) if words else 0


def compute_chunk_hash(heading_path: str, chunk_text: str) -> str:
    """Compute deterministic SHA-256 chunk hash over heading path and text body."""
    return hashlib.sha256(f"{heading_path}:{chunk_text}".encode()).hexdigest()


class MarkdownChunker:
    """Hierarchical Small-to-Big Markdown and text splitter.

    Satisfies the Chunker protocol defined in strata.core.protocols.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_size: int | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self.settings: Settings = cfg
        self.chunk_size: int = chunk_size if chunk_size is not None else cfg.chunk_size
        self.chunk_overlap: int = chunk_overlap if chunk_overlap is not None else cfg.chunk_overlap
        self.min_chunk_size: int = (
            min_chunk_size if min_chunk_size is not None else cfg.min_chunk_size
        )

        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.min_chunk_size < 0:
            raise ValueError(f"min_chunk_size must be >= 0, got {self.min_chunk_size}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than "
                f"chunk_size ({self.chunk_size})"
            )

    def extract_sections(self, document: Document) -> list[Section]:
        """Extract parent section hierarchy and content from document.

        Parses markdown heading levels (#, ##, ###, etc.) while ignoring lines
        inside fenced code blocks. If no headings are present, falls back to a
        single section using the document title.
        """
        raw_content = document.raw_content
        lines = raw_content.splitlines()

        # Parse headings outside of code blocks
        in_code_block = False
        fence_char = ""
        fence_len = 0
        parsed_headings: list[tuple[int, int, str]] = []  # (line_index, level, title)

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not in_code_block:
                fence_match = _FENCE_PATTERN.match(line)
                if fence_match:
                    fence = fence_match.group(1)
                    in_code_block = True
                    fence_char = fence[0]
                    fence_len = len(fence)
                    continue

                heading_match = _HEADING_PATTERN.match(line)
                if heading_match:
                    level = len(heading_match.group(1))
                    raw_title = heading_match.group(2).strip()
                    title = raw_title if raw_title else f"Section {level}"
                    parsed_headings.append((i, level, title))
            else:
                close_match = _FENCE_PATTERN.match(line)
                if close_match and stripped.startswith(fence_char * fence_len):
                    in_code_block = False
                    fence_char = ""
                    fence_len = 0

        # Boundary condition: Document without headings (fallback to document title)
        if not parsed_headings:
            content = raw_content.strip()
            sec_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{document.id}:section:0").hex
            return [
                Section(
                    id=sec_id,
                    document_id=document.id,
                    title=document.title,
                    level=1,
                    heading_path=document.title,
                    content=content,
                    index=0,
                    token_count=count_tokens(content),
                )
            ]

        sections: list[Section] = []
        section_index = 0

        # Check for preamble content before the first heading
        first_heading_line_idx = parsed_headings[0][0]
        if first_heading_line_idx > 0:
            preamble_lines = lines[:first_heading_line_idx]
            preamble_content = "\n".join(preamble_lines).strip()
            if preamble_content:
                sec_id = uuid.uuid5(
                    uuid.NAMESPACE_DNS, f"{document.id}:section:{section_index}"
                ).hex
                sections.append(
                    Section(
                        id=sec_id,
                        document_id=document.id,
                        title=document.title,
                        level=1,
                        heading_path=document.title,
                        content=preamble_content,
                        index=section_index,
                        token_count=count_tokens(preamble_content),
                    )
                )
                section_index += 1

        # Process each heading and its body
        heading_stack: list[tuple[int, str]] = []  # (level, title)
        num_headings = len(parsed_headings)

        for k, (line_idx, level, title) in enumerate(parsed_headings):
            # Calculate slice for section body
            next_line_idx = parsed_headings[k + 1][0] if k + 1 < num_headings else len(lines)
            body_lines = lines[line_idx + 1 : next_line_idx]
            body_content = "\n".join(body_lines).strip()

            # Maintain hierarchical heading stack (for Breadcrumb trail in retrieved results).
            # ### SQLite needs to know it belongs to # Architecture > ## Storage > ### SQLite
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            heading_path = " > ".join(t for _, t in heading_stack)

            sec_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{document.id}:section:{section_index}").hex
            sections.append(
                Section(
                    id=sec_id,
                    document_id=document.id,
                    title=title,
                    level=level,
                    heading_path=heading_path,
                    content=body_content,
                    index=section_index,
                    token_count=count_tokens(body_content),
                )
            )
            section_index += 1

        return sections

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split document into small search chunks with heading paths.

        Uses Small-to-Big strategy:
        - Extracts parent sections first.
        - Short sections meeting min_chunk_size are preserved as single chunks.
        - Oversized sections are subdivided into overlapping chunks of chunk_size with chunk_overlap.
        - Links each chunk to parent section ID and preserves heading_path.
        - Computes deterministic SHA-256 chunk_hash.
        """
        sections = self.extract_sections(document)
        chunks: list[Chunk] = []
        chunk_index = 0

        for section in sections:
            content = section.content.strip()
            if not content:
                continue

            section_chunks = self._chunk_section(
                document=document,
                section=section,
                text=content,
                start_index=chunk_index,
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return chunks

    def _chunk_section(
        self,
        document: Document,
        section: Section,
        text: str,
        start_index: int,
    ) -> list[Chunk]:
        """Subdivide a single section body into one or more granular search chunks."""
        tokens = self._encode_text(text)
        total_tokens = len(tokens)

        # Boundary condition: text shorter than min_chunk_size is discarded
        if total_tokens < self.min_chunk_size:
            return []

        # Retain short section as a single chunk if it meets min_chunk_size and fits in chunk_size
        if total_tokens <= self.chunk_size:
            chunk_hash = compute_chunk_hash(section.heading_path, text)
            chunk_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{document.id}:chunk:{start_index}").hex
            return [
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    chunk_hash=chunk_hash,
                    text=text,
                    heading_path=section.heading_path,
                    index=start_index,
                    parent_section_id=section.id,
                    token_count=total_tokens,
                    tags=list(document.tags),
                )
            ]

        # Oversized body: sliding window with chunk_size and chunk_overlap
        stride = max(1, self.chunk_size - self.chunk_overlap)
        chunks: list[Chunk] = []
        current_start = 0
        current_idx = start_index

        while current_start < total_tokens:
            current_end = min(current_start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[current_start:current_end]

            # If the remaining slice is below min_chunk_size, try shifting backward to cover end
            if len(chunk_tokens) < self.min_chunk_size and chunks:
                adjusted_start = max(0, total_tokens - self.chunk_size)
                if adjusted_start != current_start:
                    chunk_tokens = tokens[adjusted_start:total_tokens]
                    chunk_text = self._decode_tokens(chunk_tokens).strip()
                    if chunks and chunk_text != chunks[-1].text:
                        chunk_hash = compute_chunk_hash(section.heading_path, chunk_text)
                        chunk_id = uuid.uuid5(
                            uuid.NAMESPACE_DNS, f"{document.id}:chunk:{current_idx}"
                        ).hex
                        chunks.append(
                            Chunk(
                                id=chunk_id,
                                document_id=document.id,
                                chunk_hash=chunk_hash,
                                text=chunk_text,
                                heading_path=section.heading_path,
                                index=current_idx,
                                parent_section_id=section.id,
                                token_count=len(chunk_tokens),
                                tags=list(document.tags),
                            )
                        )
                break

            chunk_text = self._decode_tokens(chunk_tokens).strip()
            if chunk_text:
                chunk_hash = compute_chunk_hash(section.heading_path, chunk_text)
                chunk_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{document.id}:chunk:{current_idx}").hex
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document.id,
                        chunk_hash=chunk_hash,
                        text=chunk_text,
                        heading_path=section.heading_path,
                        index=current_idx,
                        parent_section_id=section.id,
                        token_count=len(chunk_tokens),
                        tags=list(document.tags),
                    )
                )
                current_idx += 1

            if current_end >= total_tokens:
                break

            current_start += stride

        return chunks

    def _encode_text(self, text: str) -> list[int]:
        """Encode text to token integer sequences using tiktoken with fallback."""
        try:
            encoder = get_token_encoder()
            return list(encoder.encode(text))
        except Exception:
            words = text.split()
            return list(range(len(words)))

    def _decode_tokens(self, tokens: Sequence[int]) -> str:
        """Decode token integer sequence back to text string."""
        try:
            encoder = get_token_encoder()
            return encoder.decode(tokens, errors="replace")
        except Exception:
            return " ".join(str(t) for t in tokens)
