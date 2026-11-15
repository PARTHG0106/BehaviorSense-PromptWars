"""Shared upload bounds. MB in the public API means 1,048,576 bytes."""

from __future__ import annotations

import os

DEFAULT_MAX_UPLOAD_MB = 100
UPLOAD_CHUNK_BYTES = 8 << 20
MAX_METADATA_BYTES = 1 << 20


def max_upload_mb() -> int:
    """Read a positive integer cap; fail visibly on malformed configuration."""
    raw = os.environ.get("BS_MAX_UPLOAD_MB", str(DEFAULT_MAX_UPLOAD_MB))
    try:
        value = int(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("BS_MAX_UPLOAD_MB must be a positive integer") from exc
    if value < 1:
        raise ValueError("BS_MAX_UPLOAD_MB must be a positive integer")
    return value


def upload_health(limit_mb: int | None = None) -> dict:
    return {"max_upload_mb": max_upload_mb() if limit_mb is None else limit_mb,
            "upload_protocol": "chunks-v1", "upload_chunk_bytes": UPLOAD_CHUNK_BYTES}
