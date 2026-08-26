"""Hash the executable source tree recorded with every experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance_snapshot(repo_root: str | Path) -> dict:
    """Return a deterministic hash manifest for code used by ``main.py``."""

    repo_root = Path(repo_root).resolve()
    paths = [repo_root / "main.py", *sorted((repo_root / "src").rglob("*.py"))]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot hash executable source files: {missing}")
    files = {
        path.relative_to(repo_root).as_posix(): _sha256_file(path)
        for path in paths
    }
    canonical = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "source_hash": hashlib.sha256(canonical).hexdigest(),
        "source_hash_kind": "source_files_manifest_sha256",
        "source_files_sha256": files,
    }
