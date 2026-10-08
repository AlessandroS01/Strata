"""Local document ingestion and file parsing module for Strata."""

from .reader import LocalFileReader, read_file, scan_directory

__all__ = ["LocalFileReader", "read_file", "scan_directory"]
