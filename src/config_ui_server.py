"""Backward-compatible import path. New entrypoint: ``api.app:app``."""

from api.app import app

__all__ = ["app"]
