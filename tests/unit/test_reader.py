"""Unit tests for strata.ingestion.reader (Local Markdown Reader)."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from strata.core.models import Document
from strata.core.protocols import DocumentReader
from strata.ingestion.reader import (
    LocalFileReader,
    read_file,
    scan_directory,
)


def test_protocol_conformance() -> None:
    """Verify LocalFileReader satisfies DocumentReader typing.Protocol."""
    reader = LocalFileReader()
    assert isinstance(reader, DocumentReader)


def test_read_file_standard_yaml_frontmatter(tmp_path: Path) -> None:
    """Verify parsing a Markdown file with standard YAML frontmatter."""
    content = (
        "---\n"
        "title: Raft Consensus Algorithm\n"
        "tags:\n"
        "  - distributed-systems\n"
        "  - consensus\n"
        "author: Diego Ongaro\n"
        "year: 2014\n"
        "verified: true\n"
        "---\n"
        "\n"
        "# Raft Overview\n"
        "\n"
        "Raft is a consensus algorithm designed to be understandable.\n"
    )
    file_path = tmp_path / "raft.md"
    file_path.write_bytes(content.encode("utf-8"))

    doc = read_file(file_path)

    assert isinstance(doc, Document)
    assert doc.title == "Raft Consensus Algorithm"
    assert doc.tags == ["distributed-systems", "consensus"]
    assert doc.metadata == {
        "author": "Diego Ongaro",
        "year": "2014",
        "verified": "True",
    }
    assert (
        doc.raw_content
        == "# Raft Overview\n\nRaft is a consensus algorithm designed to be understandable."
    )
    assert doc.file_path == str(file_path.resolve())
    assert doc.doc_hash == hashlib.sha256(content.encode("utf-8")).hexdigest()


def test_read_file_comma_separated_tags(tmp_path: Path) -> None:
    """Verify parsing comma-separated tags string in frontmatter."""
    content = (
        "---\ntitle: Database Internals\ntags: storage, b-tree, lsm-tree\n---\nBody content here."
    )
    file_path = tmp_path / "db.md"
    file_path.write_text(content, encoding="utf-8")

    doc = read_file(file_path)
    assert doc.tags == ["storage", "b-tree", "lsm-tree"]


def test_read_file_tag_normalization_and_deduplication(tmp_path: Path) -> None:
    """Verify tags are stripped of whitespace and deduplicated while preserving order."""
    content = "---\ntitle: Tag Test\ntags:\n  - python\n  - rag\n  - python\n  -  ai  \n---\nBody."
    file_path = tmp_path / "tags.md"
    file_path.write_text(content, encoding="utf-8")

    doc = read_file(file_path)
    assert doc.tags == ["python", "rag", "ai"]


def test_read_file_metadata_nested_structures(tmp_path: Path) -> None:
    """Verify nested dictionary and list metadata attributes are serialized as JSON strings."""
    content = (
        "---\n"
        "title: Complex Metadata\n"
        "config:\n"
        "  timeout: 30\n"
        "  retries: 3\n"
        "reviewers:\n"
        "  - Alice\n"
        "  - Bob\n"
        "---\n"
        "Body."
    )
    file_path = tmp_path / "complex.md"
    file_path.write_text(content, encoding="utf-8")

    doc = read_file(file_path)
    assert json.loads(doc.metadata["config"]) == {"timeout": 30, "retries": 3}
    assert json.loads(doc.metadata["reviewers"]) == ["Alice", "Bob"]


def test_read_file_title_fallback_to_h1(tmp_path: Path) -> None:
    """Verify title falls back to first # H1 heading when frontmatter has no title."""
    content = (
        "---\n"
        "tags: [notes]\n"
        "author: Alessandro\n"
        "---\n"
        "\n"
        "# Heading From Markdown Body\n"
        "\n"
        "Some details."
    )
    file_path = tmp_path / "notes.md"
    file_path.write_text(content, encoding="utf-8")

    doc = read_file(file_path)
    assert doc.title == "Heading From Markdown Body"
    assert doc.tags == ["notes"]
    assert doc.metadata == {"author": "Alessandro"}


def test_read_file_title_fallback_to_h1_with_trailing_hashes(tmp_path: Path) -> None:
    """Verify # H1 heading with trailing hashes is parsed cleanly."""
    content = "# Closed Heading #\n\nBody content."
    file_path = tmp_path / "closed.md"
    file_path.write_text(content, encoding="utf-8")

    doc = read_file(file_path)
    assert doc.title == "Closed Heading"


def test_read_file_title_ignores_h1_inside_code_fence(tmp_path: Path) -> None:
    """Verify # comments inside fenced code blocks are not treated as document H1 titles."""
    content = (
        "```python\n"
        "# This is a python comment, not the document title\n"
        "x = 42\n"
        "```\n"
        "\n"
        "# Legitimate Document Title\n"
        "\n"
        "More content."
    )
    file_path = tmp_path / "script_doc.md"
    file_path.write_text(content, encoding="utf-8")

    doc = read_file(file_path)
    assert doc.title == "Legitimate Document Title"


def test_read_file_title_fallback_to_file_stem(tmp_path: Path) -> None:
    """Verify title falls back to filename stem when neither frontmatter nor H1 exists."""
    content = "## Subheading Only\n\nParagraph text without any level 1 heading.\n"
    file_path = tmp_path / "system_architecture.markdown"
    file_path.write_text(content, encoding="utf-8")

    doc = read_file(file_path)
    assert doc.title == "system_architecture"


def test_read_file_empty_file(tmp_path: Path) -> None:
    """Verify parsing an empty (0 bytes) file returns valid Document with defaults."""
    file_path = tmp_path / "empty_note.md"
    file_path.write_text("", encoding="utf-8")

    doc = read_file(file_path)
    assert doc.title == "empty_note"
    assert doc.raw_content == ""
    assert doc.tags == []
    assert doc.metadata == {}
    assert doc.doc_hash == hashlib.sha256(b"").hexdigest()


def test_read_file_whitespace_only_file(tmp_path: Path) -> None:
    """Verify parsing a whitespace-only file returns clean empty raw_content."""
    file_path = tmp_path / "blank.md"
    file_path.write_text("   \n\n\t  \n", encoding="utf-8")

    doc = read_file(file_path)
    assert doc.title == "blank"
    assert doc.raw_content == ""
    assert doc.tags == []
    assert doc.metadata == {}


def test_read_file_only_frontmatter_no_body(tmp_path: Path) -> None:
    """Verify parsing a file containing only frontmatter and no body text."""
    content = "---\ntitle: Pure Metadata\ntags: [meta]\n---\n"
    file_path = tmp_path / "meta_only.md"
    file_path.write_text(content, encoding="utf-8")

    doc = read_file(file_path)
    assert doc.title == "Pure Metadata"
    assert doc.tags == ["meta"]
    assert doc.raw_content == ""


def test_read_file_malformed_yaml_frontmatter_handled_gracefully(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify invalid YAML frontmatter logs a warning and treats the document as plain markdown."""
    content = (
        "---\ntitle: [unclosed list\nbroken : :\n---\n# Recovered Title\n\nBody content here.\n"
    )
    file_path = tmp_path / "broken_fm.md"
    file_path.write_text(content, encoding="utf-8")

    # Must not crash
    doc = read_file(file_path)

    # Title recovered from # H1 heading in plain markdown body
    assert doc.title == "Recovered Title"
    assert doc.tags == []
    assert doc.metadata == {}
    assert "# Recovered Title" in doc.raw_content
    assert "Body content here." in doc.raw_content
    # Check warning was logged
    assert any("Failed to parse YAML frontmatter" in record.message for record in caplog.records)


def test_read_file_non_dict_yaml_frontmatter_handled_gracefully(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify non-dictionary YAML (e.g. list or scalar) logs warning and treats frontmatter as empty."""
    content = "---\n- item 1\n- item 2\n---\n# List FM Doc\n\nBody."
    file_path = tmp_path / "list_fm.md"
    file_path.write_text(content, encoding="utf-8")

    doc = read_file(file_path)
    assert doc.title == "List FM Doc"
    assert doc.tags == []
    assert doc.metadata == {}
    assert any("is not a dictionary" in record.message for record in caplog.records)


def test_read_file_unclosed_frontmatter_treated_as_plain_markdown(
    tmp_path: Path,
) -> None:
    """Verify a file starting with '---' but lacking closing delimiter is treated as plain body."""
    content = "---\nThis is not valid frontmatter and never closes\n# Unclosed Doc\nBody."
    file_path = tmp_path / "unclosed.md"
    file_path.write_text(content, encoding="utf-8")

    doc = read_file(file_path)
    assert doc.title == "Unclosed Doc"
    assert doc.tags == []
    assert doc.metadata == {}
    assert "This is not valid frontmatter" in doc.raw_content


def test_read_file_utf8_bom_handling(tmp_path: Path) -> None:
    """Verify UTF-8 BOM at file start is cleanly stripped and frontmatter parses."""
    bom_content = (
        b"\xef\xbb\xbf---\ntitle: BOM Note\ntags: [windows]\n---\n# Content\nHello world.\n"
    )
    file_path = tmp_path / "bom.md"
    file_path.write_bytes(bom_content)

    doc = read_file(file_path)
    assert doc.title == "BOM Note"
    assert doc.tags == ["windows"]
    assert doc.raw_content == "# Content\nHello world."
    assert doc.doc_hash == hashlib.sha256(bom_content).hexdigest()


def test_read_file_crlf_line_endings(tmp_path: Path) -> None:
    """Verify Windows CRLF (\\r\\n) line endings are properly parsed."""
    crlf_content = (
        "---\r\ntitle: Windows CRLF Note\r\ntags: [crlf]\r\n---\r\n# Heading\r\nText line.\r\n"
    )
    file_path = tmp_path / "crlf.md"
    file_path.write_bytes(crlf_content.encode("utf-8"))

    doc = read_file(file_path)
    assert doc.title == "Windows CRLF Note"
    assert doc.tags == ["crlf"]
    assert "Text line." in doc.raw_content


def test_read_file_malformed_utf8_raises_unicode_decode_error(
    tmp_path: Path,
) -> None:
    """Verify reading an invalid UTF-8 file raises UnicodeDecodeError."""
    bad_bytes = b"\xff\xfe\x00\x00\x80\x81"
    file_path = tmp_path / "corrupted.md"
    file_path.write_bytes(bad_bytes)

    with pytest.raises(UnicodeDecodeError):
        read_file(file_path)


def test_read_file_nonexistent_file_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """Verify reading a non-existent file raises FileNotFoundError."""
    missing_path = tmp_path / "does_not_exist.md"
    with pytest.raises(FileNotFoundError):
        read_file(missing_path)


def test_read_file_directory_raises_is_a_directory_error(
    tmp_path: Path,
) -> None:
    """Verify passing a directory path to read_file raises IsADirectoryError."""
    with pytest.raises(IsADirectoryError):
        read_file(tmp_path)


def test_document_immutability(tmp_path: Path) -> None:
    """Verify Document models returned by read_file are frozen and immutable."""
    file_path = tmp_path / "immutable.md"
    file_path.write_text("# Title\nBody", encoding="utf-8")

    doc = read_file(file_path)
    with pytest.raises(ValidationError):
        setattr(doc, "title", "Changed")  # noqa: B010


def test_scan_directory_recursive_and_flat(tmp_path: Path) -> None:
    """Verify directory scanning discovers .md and .markdown files in flat and recursive modes."""
    # Root level files
    (tmp_path / "doc1.md").write_text("# Doc 1\nBody 1", encoding="utf-8")
    (tmp_path / "doc2.markdown").write_text("# Doc 2\nBody 2", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("Plain text file", encoding="utf-8")

    # Subdirectory files
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    (sub_dir / "doc3.md").write_text("# Doc 3\nBody 3", encoding="utf-8")

    nested_dir = sub_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "doc4.markdown").write_text("# Doc 4\nBody 4", encoding="utf-8")

    # Flat scan (recursive=False)
    flat_docs = scan_directory(tmp_path, recursive=False)
    assert len(flat_docs) == 2
    flat_titles = {d.title for d in flat_docs}
    assert flat_titles == {"Doc 1", "Doc 2"}

    # Recursive scan (recursive=True)
    rec_docs = scan_directory(tmp_path, recursive=True)
    assert len(rec_docs) == 4
    rec_titles = {d.title for d in rec_docs}
    assert rec_titles == {"Doc 1", "Doc 2", "Doc 3", "Doc 4"}


def test_scan_directory_skips_hidden_and_vendor_directories(
    tmp_path: Path,
) -> None:
    """Verify scan_directory skips hidden dirs/files and common vendor directories."""
    # Valid file
    (tmp_path / "valid.md").write_text("# Valid Note\nContent", encoding="utf-8")

    # Hidden file
    (tmp_path / ".hidden_doc.md").write_text("# Hidden\nContent", encoding="utf-8")

    # Vendor and hidden directories
    for vendor_dir_name in [
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        ".cache",
        ".hidden_folder",
    ]:
        vdir = tmp_path / vendor_dir_name
        vdir.mkdir()
        (vdir / "ignored.md").write_text("# Ignored", encoding="utf-8")

    docs = scan_directory(tmp_path, recursive=True)
    assert len(docs) == 1
    assert docs[0].title == "Valid Note"


def test_scan_directory_gracefully_skips_corrupted_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify scan_directory skips unreadable/corrupted files without aborting the scan."""
    (tmp_path / "good.md").write_text("# Good Note\nContent", encoding="utf-8")
    (tmp_path / "corrupted.md").write_bytes(b"\xff\xfe\x00\x00\x80\x81")

    docs = scan_directory(tmp_path, recursive=True)
    assert len(docs) == 1
    assert docs[0].title == "Good Note"
    assert any("Skipping unreadable file" in record.message for record in caplog.records)


def test_scan_directory_nonexistent_dir_raises(tmp_path: Path) -> None:
    """Verify scan_directory on non-existent path raises FileNotFoundError."""
    missing = tmp_path / "missing_dir"
    with pytest.raises(FileNotFoundError):
        scan_directory(missing)


def test_scan_directory_file_path_raises_not_a_directory_error(
    tmp_path: Path,
) -> None:
    """Verify scan_directory on a file path raises NotADirectoryError."""
    file_path = tmp_path / "file.md"
    file_path.write_text("# File", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        scan_directory(file_path)


def test_custom_reader_configuration(tmp_path: Path) -> None:
    """Verify LocalFileReader with custom ignored dirs and extensions."""
    (tmp_path / "note.txt").write_text("# Text Note", encoding="utf-8")
    (tmp_path / "note.md").write_text("# Markdown Note", encoding="utf-8")

    custom_vendor = tmp_path / "custom_vendor"
    custom_vendor.mkdir()
    (custom_vendor / "vendor.txt").write_text("# Vendor", encoding="utf-8")

    custom_reader = LocalFileReader(
        ignored_dirs={"custom_vendor"},
        supported_extensions={".txt"},
    )
    docs = custom_reader.scan_directory(tmp_path, recursive=True)
    assert len(docs) == 1
    assert docs[0].title == "Text Note"
