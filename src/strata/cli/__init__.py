"""Command-line interface subsystem for Strata air-gapped knowledge engine.

Exposes Typer application entrypoint, dependency factory, and output renderers.
"""

from .main import app

__all__ = ["app"]
