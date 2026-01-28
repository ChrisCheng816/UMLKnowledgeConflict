#!/usr/bin/env python3
"""
Recursively find every data.txt under a root folder, and in-place de-duplicate lines
based on the full line content (all whitespace-separated tokens), case-insensitively.

Rule:
- Use the entire stripped line as the key for de-duplication.
- Keep the first occurrence, preserve original order.
- Modify each data.txt in place (no new file), using an atomic replace.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List


def dedup_file_in_place(path: Path) -> int:
    """
    Returns the number of removed lines.
    """
    original_text = path.read_text(encoding="utf-8", errors="replace")
    lines = original_text.splitlines(keepends=True)

    seen = set()
    kept: List[str] = []
    removed = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue

        # Use the full line (all tokens) as the key; case-insensitive.
        key = stripped.casefold()

        if key in seen:
            removed += 1
            continue

        seen.add(key)
        kept.append(line)

    new_text = "".join(kept)

    if new_text == original_text:
        return 0

    tmp_path = path.with_name(path.name + ".tmp_dedup")
    tmp_path.write_text(new_text, encoding="utf-8", newline="")

    try:
        os.chmod(tmp_path, path.stat().st_mode)
    except OSError:
        pass

    os.replace(tmp_path, path)
    return removed


def count_removed(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    seen = set()
    removed = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Same key rule as dedup_file_in_place.
        key = stripped.casefold()

        if key in seen:
            removed += 1
        else:
            seen.add(key)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory to scan recursively.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would change; do not modify files.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    total_files = 0
    total_removed = 0

    for path in root.rglob("data.txt"):
        if not path.is_file():
            continue

        total_files += 1
        removed = count_removed(path) if args.dry_run else dedup_file_in_place(path)

        if removed > 0:
            total_removed += removed
            rel = path.relative_to(root)
            action = "WOULD_DEDUP" if args.dry_run else "DEDUP"
            print(f"{action} {rel}  removed_lines={removed}")

    print(f"Scanned_files={total_files}  removed_lines_total={total_removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
