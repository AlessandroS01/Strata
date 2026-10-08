"""Document ingestion and structural parsing subsystem.

Responsible for file discovery and AST-aware Markdown processing:
- reader: Discovers local Markdown files, parses YAML frontmatter, and hydratates Documents[cite: 1].
- chunker: Parses heading hierarchies into Sections and generates granular Chunks (Small-to-Big retrieval)[cite: 1, 3].
"""

from .chunker import MarkdownChunker, count_tokens, get_token_encoder
from .reader import LocalFileReader, read_file, scan_directory

__all__ = [
    "LocalFileReader",
    "MarkdownChunker",
    "count_tokens",
    "get_token_encoder",
    "read_file",
    "scan_directory",
]
