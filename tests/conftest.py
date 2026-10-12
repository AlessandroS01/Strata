"""Pytest configuration and global test guards for Strata test suite."""

from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def guard_real_data_dir() -> Generator[None, None, None]:
    """Ensure unit tests never leak or overwrite data in the workspace ./data directory."""
    data_dir = Path("./data")
    initial_files = set(data_dir.rglob("*")) if data_dir.exists() else set()

    yield

    current_files = set(data_dir.rglob("*")) if data_dir.exists() else set()
    leaked_files = current_files - initial_files
    for leaked in leaked_files:
        if leaked.is_file():
            leaked.unlink()
    assert not leaked_files, (
        f"Test leaked files into repository data directory {data_dir.resolve()}: "
        f"{[str(p) for p in leaked_files]}. Unit tests must use tmp_path or isolated Settings."
    )
