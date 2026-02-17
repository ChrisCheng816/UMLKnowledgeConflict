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
from typing import Dict, Iterable, List


def _dedup_key(stripped: str) -> str:
    parts = stripped.split()
    if len(parts) == 2:
        a, b = parts[0].casefold(), parts[1].casefold()
        if a <= b:
            return f"{a}\t{b}"
        return f"{b}\t{a}"
    if len(parts) == 3:
        a, b, c = parts[0].casefold(), parts[1].casefold(), parts[2].casefold()
        a, b, c = sorted([a, b, c])
        return f"{a}\t{b}\t{c}"
    return stripped.casefold()


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

        # Use unordered pair for 2-token lines; otherwise full line, case-insensitive.
        key = _dedup_key(stripped)

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
        key = _dedup_key(stripped)

        if key in seen:
            removed += 1
        else:
            seen.add(key)
    return removed


def dedup_files_in_place(paths: Iterable[Path], seen: set[str]) -> int:
    """
    De-duplicate across multiple files using a shared 'seen' set.
    Returns total removed lines across all files.
    """
    removed_total = 0
    for path in paths:
        original_text = path.read_text(encoding="utf-8", errors="replace")
        lines = original_text.splitlines(keepends=True)

        kept: List[str] = []
        removed = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue

            key = _dedup_key(stripped)
            if key in seen:
                removed += 1
                continue

            seen.add(key)
            kept.append(line)

        new_text = "".join(kept)
        if new_text != original_text:
            tmp_path = path.with_name(path.name + ".tmp_dedup")
            tmp_path.write_text(new_text, encoding="utf-8", newline="")
            try:
                os.chmod(tmp_path, path.stat().st_mode)
            except OSError:
                pass
            os.replace(tmp_path, path)

        removed_total += removed
    return removed_total


def collect_grouped_data_files(root: Path, splits: List[str]) -> Dict[str, List[Path]]:
    """
    Collect data.txt files grouped by split and class folder, e.g.:
    - forward/2Class_Aggregation
    - reverse/3Class_Dependency
    """
    groups: Dict[str, List[Path]] = {}

    for split in splits:
        split_root = root / f"data_{split}"
        if not split_root.exists() or not split_root.is_dir():
            continue

        class_dirs = [
            p
            for p in split_root.iterdir()
            if p.is_dir() and (p.name.startswith("2Class_") or p.name.startswith("3Class_"))
        ]
        for class_dir in sorted(class_dirs, key=lambda p: p.name.lower()):
            files = [p for p in sorted(class_dir.rglob("data.txt")) if p.is_file()]
            if files:
                groups[f"{split}/{class_dir.name}"] = files

    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Project root directory. Default: parent directory of this script.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["forward", "reverse"],
        default=["forward", "reverse"],
        help="Which dataset splits to process. Default: forward reverse",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would change; do not modify files.",
    )
    args = parser.parse_args()

    root = (
        Path(args.root).expanduser().resolve()
        if args.root
        else Path(__file__).resolve().parent.parent
    )
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    groups = collect_grouped_data_files(root, args.splits)
    if not groups:
        raise SystemExit(
            f"No matching data.txt found under: {root} "
            f"(splits={','.join(args.splits)}, class dirs=2Class_*/3Class_*)"
        )

    total_files = 0
    total_removed = 0
    for paths in groups.values():
        total_files += len(paths)

    for group, paths in sorted(groups.items(), key=lambda kv: kv[0].lower()):
        # Dry-run: count removals using a shared seen set across the group.
        if args.dry_run:
            seen: set[str] = set()
            removed = 0
            for path in paths:
                text = path.read_text(encoding="utf-8", errors="replace")
                for line in text.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    key = _dedup_key(stripped)
                    if key in seen:
                        removed += 1
                    else:
                        seen.add(key)
            if removed > 0:
                total_removed += removed
                print(f"WOULD_DEDUP {group}  removed_lines={removed}")
            continue

        # In-place: de-duplicate across all files in the group.
        removed = dedup_files_in_place(paths, seen=set())
        if removed > 0:
            total_removed += removed
            print(f"DEDUP {group}  removed_lines={removed}")

    print(f"Scanned_files={total_files}  removed_lines_total={total_removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
