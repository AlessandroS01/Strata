"""Local Markdown reader and directory scanner for Strata.

Parses local Markdown files into canonical Document domain models with
YAML frontmatter extraction, deterministic SHA-256 hashing, and title fallback cascades.
"""

import json
import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path

import yaml

from strata.core.hasher import compute_document_hash
from strata.core.models import Document
from strata.core.protocols import DocumentReader

logger = logging.getLogger(__name__)

DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset({".git", "node_modules", "__pycache__", ".venv"})
DEFAULT_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown"})

_FENCE_PATTERN = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_H1_PATTERN = re.compile(r"^\s{0,3}#\s+(.*?)(?:\s+#+)?\s*$")


def _extract_first_h1(text: str) -> str | None:
    """Extract the first Markdown H1 heading outside of code fences."""
    in_code_block = False
    for line in text.splitlines():
        if _FENCE_PATTERN.match(line):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        match = _H1_PATTERN.match(line)
        if match:
            heading = match.group(1).strip()
            if heading:
                return heading
    return None


def _parse_frontmatter_and_body(content: str, file_path: Path) -> tuple[dict[str, object], str]:
    """Parse YAML frontmatter delimited by '---' and separate markdown body.

    If frontmatter is malformed or invalid YAML, logs a warning and returns
    an empty frontmatter dictionary with the entire content as plain body.
    """
    clean_content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    lines = clean_content.splitlines(keepends=True)
    if not lines:
        return {}, ""

    # Frontmatter must begin with '---' on the first line
    if lines[0].strip() != "---":
        return {}, clean_content.strip()

    # Search for closing delimiter line
    closing_idx = -1
    for idx in range(1, len(lines)):
        line_stripped = lines[idx].strip()
        if line_stripped in ("---", "..."):
            closing_idx = idx
            break

    if closing_idx == -1:
        # No closing delimiter found; treat as plain markdown body
        return {}, clean_content.strip()

    yaml_text = "".join(lines[1:closing_idx])
    body_text = "".join(lines[closing_idx + 1 :])

    try:
        parsed: object = yaml.safe_load(yaml_text)
        if isinstance(parsed, dict):
            # Convert keys to str for safety
            frontmatter_dict: dict[str, object] = {str(k): v for k, v in parsed.items()}
            return frontmatter_dict, body_text.strip()
        elif parsed is None:
            return {}, body_text.strip()
        else:
            logger.warning(
                "Frontmatter in %s is not a dictionary (got %s); treating as empty",
                file_path,
                type(parsed).__name__,
            )
            return {}, body_text.strip()
    except yaml.YAMLError as exc:
        logger.warning(
            "Failed to parse YAML frontmatter in %s: %s; treating as plain markdown body",
            file_path,
            exc,
        )
        return {}, clean_content.strip()


def _extract_title(frontmatter: dict[str, object], body: str, file_path: Path) -> str:
    """Extract document title with priority: frontmatter title -> H1 heading -> file stem."""
    # 1. Frontmatter title (case-insensitive key check)
    for k, v in frontmatter.items():
        if k.lower() == "title" and v is not None:
            title_str = str(v).strip()
            if title_str:
                return title_str

    # 2. First Markdown # H1 heading in body
    h1_title = _extract_first_h1(body)
    if h1_title:
        return h1_title

    # 3. Fallback to file stem (e.g. notes.md -> 'notes')
    return file_path.stem


def _extract_tags(frontmatter: dict[str, object]) -> list[str]:
    """Extract and normalize tags from frontmatter into an ordered, deduplicated list."""
    raw_tags: object = None
    for k, v in frontmatter.items():
        if k.lower() == "tags":
            raw_tags = v
            break

    if raw_tags is None:
        return []

    tags: list[str] = []
    if isinstance(raw_tags, (list, tuple, set)):
        for item in raw_tags:
            tag_str = str(item).strip()
            if tag_str:
                tags.append(tag_str)
    elif isinstance(raw_tags, str):
        for part in raw_tags.split(","):
            tag_str = part.strip()
            if tag_str:
                tags.append(tag_str)
    else:
        tag_str = str(raw_tags).strip()
        if tag_str:
            tags.append(tag_str)

    # Deduplicate while preserving insertion order
    return list(dict.fromkeys(tags))


def _extract_metadata(frontmatter: dict[str, object]) -> dict[str, str]:
    """Extract remaining frontmatter attributes into a string-keyed and string-valued dictionary."""
    metadata: dict[str, str] = {}
    for k, v in frontmatter.items():
        if k.lower() in ("title", "tags"):
            continue
        val_str: str
        if v is None:
            val_str = ""
        elif isinstance(v, (dict, list)):
            val_str = json.dumps(v, ensure_ascii=False)
        else:
            val_str = str(v)
        metadata[str(k)] = val_str
    return metadata


class LocalFileReader(DocumentReader):
    """Scanner and parser for local Markdown documents into canonical Document models.

    Implements the DocumentReader protocol.
    """

    def __init__(
        self,
        ignored_dirs: Iterable[str] | None = None,
        supported_extensions: Iterable[str] | None = None,
    ) -> None:
        self.ignored_dirs: frozenset[str] = (
            frozenset(ignored_dirs) if ignored_dirs is not None else DEFAULT_IGNORED_DIRS
        )
        self.supported_extensions: frozenset[str] = (
            frozenset(
                ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                for ext in supported_extensions
            )
            if supported_extensions is not None
            else DEFAULT_SUPPORTED_EXTENSIONS
        )

    def read_file(self, file_path: Path | str) -> Document:
        """Read a single UTF-8 Markdown file from disk into a canonical Document model.

        Detects and strips YAML frontmatter, extracts title, tags, and metadata,
        and computes a deterministic SHA-256 hash over the raw bytes.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.is_dir():
            raise IsADirectoryError(f"Expected a file, got directory: {file_path}")

        raw_bytes = path.read_bytes()
        doc_hash = compute_document_hash(raw_bytes)

        content = raw_bytes.decode("utf-8")

        frontmatter, raw_content = _parse_frontmatter_and_body(content, path)
        title = _extract_title(frontmatter, raw_content, path)
        tags = _extract_tags(frontmatter)
        metadata = _extract_metadata(frontmatter)

        return Document(
            title=title,
            file_path=str(path),
            raw_content=raw_content,
            doc_hash=doc_hash,
            tags=tags,
            metadata=metadata,
        )

    def scan_directory(self, directory_path: Path | str, recursive: bool = True) -> list[Document]:
        """Discover and parse all Markdown documents within a directory.

        Skips hidden files and directories (starting with '.') as well as common
        vendor directories (.git, node_modules, __pycache__, .venv).
        """
        root = Path(directory_path).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        if not root.is_dir():
            raise NotADirectoryError(f"Expected a directory, got file: {directory_path}")

        discovered_files: list[Path] = []

        if recursive:
            for current_dir, dirnames, filenames in os.walk(root):
                # Filter dirnames in-place to avoid recursing into ignored/hidden dirs
                dirnames[:] = [
                    d for d in dirnames if not d.startswith(".") and d not in self.ignored_dirs
                ]

                for filename in filenames:
                    if filename.startswith("."):
                        continue
                    file_path = Path(current_dir) / filename
                    if file_path.suffix.lower() in self.supported_extensions:
                        discovered_files.append(file_path)
        else:
            for entry in root.iterdir():
                if entry.name.startswith(".") or entry.name in self.ignored_dirs:
                    continue
                if entry.is_file() and entry.suffix.lower() in self.supported_extensions:
                    discovered_files.append(entry)

        # Sort files for deterministic parsing order
        discovered_files.sort()

        documents: list[Document] = []
        for file_path in discovered_files:
            try:
                doc = self.read_file(file_path)
                documents.append(doc)
            except (UnicodeDecodeError, OSError) as exc:
                logger.warning("Skipping unreadable file %s: %s", file_path, exc)
                continue

        return documents


def read_file(file_path: Path | str) -> Document:
    """Read a local Markdown file into a canonical Document model."""
    return LocalFileReader().read_file(file_path)


def scan_directory(directory_path: Path | str, recursive: bool = True) -> list[Document]:
    """Scan a local directory for Markdown files and return canonical Document models."""
    return LocalFileReader().scan_directory(directory_path, recursive=recursive)
