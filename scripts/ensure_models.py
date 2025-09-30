"""Utility helpers for fetching (or refreshing) local model checkpoints.

Run this script before starting the FastAPI service to make sure the
TinyLlama weights live under ``agent/mobile_models/TinyLlama-1.1B-Chat-v1.0``.

Usage::

    python scripts/ensure_models.py

It will download the checkpoint if it is missing. Set ``HF_TOKEN`` in your
environment if the download requires authentication.  Use ``--force`` to
re-download even when weights are present or ``--reset`` to delete the existing
directory before pulling the checkpoint again.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover - helper script only
    raise SystemExit(
        "huggingface_hub is required to download TinyLlama weights. Install it with"
        " `pip install huggingface_hub`."
    ) from exc

DEFAULT_MODELS = (
    {
        "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "local_dir": "agent/mobile_models/TinyLlama-1.1B-Chat-v1.0",
        "min_bytes": int(os.getenv("MIN_TINYLLAMA_WEIGHT_BYTES", str(500 * 1024 * 1024))),
    },
)


def _safetensor_paths(path: Path) -> list[Path]:
    if not path.exists():
        return []
    files = list(path.glob("*.safetensors"))
    if not files:
        files = list(path.rglob("*.safetensors"))
    return files


def _total_size(paths: Iterable[Path]) -> int:
    total = 0
    for item in paths:
        try:
            total += item.stat().st_size
        except OSError:
            continue
    return total


def _format_bytes(num: int) -> str:
    if num <= 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024 or unit == "TB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{num} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _purge_directory(path: Path) -> None:
    if not path.exists():
        return
    shutil.rmtree(path)


def ensure_model(
    repo_id: str,
    local_dir: Path,
    *,
    min_bytes: int,
    force: bool = False,
    reset: bool = False,
) -> None:
    """Download ``repo_id`` into ``local_dir`` when weights are missing or invalid."""

    tensors = _safetensor_paths(local_dir)
    total_size = _total_size(tensors)
    looks_valid = total_size >= min_bytes and bool(tensors)

    if reset and local_dir.exists():
        print(f"🧹 Removing existing directory: {local_dir}")
        _purge_directory(local_dir)
        tensors = []
        total_size = 0
        looks_valid = False

    if looks_valid and not force:
        print(
            f"✔️  Weights already present: {local_dir} "
            f"({_format_bytes(total_size)} total)"
        )
        return

    if tensors and (force or not looks_valid):
        reason = (
            "forcing re-download"
            if force
            else f"only {_format_bytes(total_size)} detected (< {min_bytes} bytes)"
        )
        print(
            f"⚠️ Existing weights at {local_dir} appear invalid ({reason}); refreshing from Hugging Face."
        )

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"⬇️  Downloading {repo_id} → {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        force_download=force or not looks_valid,
    )
    tensors = _safetensor_paths(local_dir)
    total_size = _total_size(tensors)
    print(
        f"✅ Download complete: {local_dir} ({_format_bytes(total_size)} across {len(tensors)} file(s))"
    )


def ensure_all(
    configs: Iterable[dict[str, object]],
    *,
    force: bool = False,
    reset: bool = False,
) -> None:
    for item in configs:
        repo_id = str(item["repo_id"])
        local_dir = Path(str(item["local_dir"]))
        min_bytes = int(item.get("min_bytes", 0))
        ensure_model(
            repo_id,
            local_dir,
            min_bytes=min_bytes,
            force=force,
            reset=reset,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ensure local TinyLlama weights are available")
    parser.add_argument(
        "--only",
        nargs="*",
        metavar="REPO_ID",
        help="Limit downloads to specific Hugging Face repo ids",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download weights even if they appear valid",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing directories before downloading",
    )
    args = parser.parse_args(argv)

    targets = DEFAULT_MODELS
    if args.only:
        normalized = {item["repo_id"]: item for item in DEFAULT_MODELS}
        missing = [repo_id for repo_id in args.only if repo_id not in normalized]
        if missing:
            parser.error(f"Unknown repo ids requested: {', '.join(missing)}")
        targets = tuple(normalized[repo_id] for repo_id in args.only)

    ensure_all(targets, force=args.force, reset=args.reset)
    return 0


if __name__ == "__main__":  # pragma: no cover - helper script entrypoint
    sys.exit(main())
