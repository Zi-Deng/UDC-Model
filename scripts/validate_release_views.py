"""Validate NICME release views.

Checks that curated release directories do not contain large binaries or copied
artifact directories. This is intentionally lightweight so it can run in CI.
"""

from __future__ import annotations

import re
from pathlib import Path

RELEASE_ROOTS = [Path("releases/anonymous_supplement"), Path("releases/camera_ready")]
FORBIDDEN_PARTS = {"data", "weights", "checkpoints", "results", "__pycache__"}
FORBIDDEN_SUFFIXES = {".bin", ".pth", ".pt", ".ckpt", ".duckdb", ".png", ".pdf", ".html", ".pyc"}
MAX_FILE_SIZE_BYTES = 1_000_000


def main() -> None:
    failures: list[str] = []

    for root in RELEASE_ROOTS:
        if not root.exists():
            failures.append(f"Missing release directory: {root}")
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if FORBIDDEN_PARTS.intersection(relative.parts):
                failures.append(f"{path}: forbidden artifact directory name in release view")
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                failures.append(f"{path}: forbidden generated/binary suffix")
            if path.stat().st_size > MAX_FILE_SIZE_BYTES:
                failures.append(f"{path}: file exceeds {MAX_FILE_SIZE_BYTES} bytes")
            if path.suffix.lower() == ".md":
                text = path.read_text()
                for ref in re.findall(r"`((?:\.\./)+[^`]+)`", text):
                    target = (path.parent / ref).resolve()
                    if not target.exists():
                        failures.append(f"{path}: relative reference does not exist: {ref}")

    if failures:
        raise SystemExit("Release-view validation failed:\n" + "\n".join(failures))

    print("Release-view validation passed.")


if __name__ == "__main__":
    main()
