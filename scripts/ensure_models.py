"""Utility helpers for fetching local model checkpoints on demand.

Run this script before starting the FastAPI service to make sure the
TinyLlama weights live under ``agent/mobile_models/TinyLlama-1.1B-Chat-v1.0``.

Usage::

    python scripts/ensure_models.py

It will download the checkpoint if it is missing. Set ``HF_TOKEN`` in your
environment if the download requires authentication.
"""

from __future__ import annotations

import argparse
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
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "agent/mobile_models/TinyLlama-1.1B-Chat-v1.0"),
)


def _has_weights(path: Path) -> bool:
    """Return True when the target directory already contains safetensor weights."""
    return path.exists() and any(path.glob("*.safetensors"))


def ensure_model(repo_id: str, local_dir: Path) -> None:
    """Download ``repo_id`` into ``local_dir`` when no weights are present."""
    if _has_weights(local_dir):
        print(f"✔️  Weights already present: {local_dir}")
        return

    print(f"⬇️  Downloading {repo_id} → {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"✅ Download complete: {local_dir}")


def ensure_all(pairs: Iterable[tuple[str, str]]) -> None:
    for repo_id, local_dir in pairs:
        ensure_model(repo_id, Path(local_dir))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ensure local TinyLlama weights are available")
    parser.add_argument(
        "--only",
        nargs="*",
        metavar="REPO_ID",
        help="Limit downloads to specific Hugging Face repo ids",
    )
    args = parser.parse_args(argv)

    targets = DEFAULT_MODELS
    if args.only:
        normalized = {item[0]: item for item in DEFAULT_MODELS}
        missing = [repo_id for repo_id in args.only if repo_id not in normalized]
        if missing:
            parser.error(f"Unknown repo ids requested: {', '.join(missing)}")
        targets = tuple(normalized[repo_id] for repo_id in args.only)

    ensure_all(targets)
    return 0


if __name__ == "__main__":  # pragma: no cover - helper script entrypoint
    sys.exit(main())
