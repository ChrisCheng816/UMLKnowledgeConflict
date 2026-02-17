#!/usr/bin/env python3
"""
Recursively find every data.txt under a root folder, and in-place de-duplicate lines
based on the full line content (all whitespace-separated tokens), case-insensitively.

Rule:
- 2 tokens: unordered pair key (A B == B A), case-insensitive.
- 3+ tokens in 3Class relation groups: use relation-specific unordered pair.
  - Inheritance/Dependency -> unordered (a,b)
  - Composition/Aggregation -> unordered (b,c)
- Fallback: use the full stripped line key, case-insensitive.
- Keep the first occurrence, preserve original order.
- Modify each data.txt in place (no new file), using an atomic replace.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _dedup_key(stripped: str, pair_mode: str | None = None) -> str:
    parts = stripped.split()
    if len(parts) == 2:
        a, b = parts[0].casefold(), parts[1].casefold()
        if a <= b:
            return f"{a}\t{b}"
        return f"{b}\t{a}"
    if len(parts) >= 3 and pair_mode in {"ab", "bc"}:
        a, b, c = parts[0].casefold(), parts[1].casefold(), parts[2].casefold()
        left, right = (a, b) if pair_mode == "ab" else (b, c)
        if left > right:
            left, right = right, left
        if pair_mode == "ab":
            return f"ab\t{left}\t{right}"
        return f"bc\t{left}\t{right}"
    if len(parts) == 3:
        a, b, c = parts[0].casefold(), parts[1].casefold(), parts[2].casefold()
        a, b, c = sorted([a, b, c])
        return f"{a}\t{b}\t{c}"
    return stripped.casefold()


def dedup_file_in_place(path: Path, pair_mode: str | None = None) -> int:
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
        key = _dedup_key(stripped, pair_mode=pair_mode)

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


def count_removed(path: Path, pair_mode: str | None = None) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    seen = set()
    removed = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Same key rule as dedup_file_in_place.
        key = _dedup_key(stripped, pair_mode=pair_mode)

        if key in seen:
            removed += 1
        else:
            seen.add(key)
    return removed


RemovedDetail = Tuple[Path, int, str, Path, int, str]


def dedup_files_in_place(
    paths: Iterable[Path],
    pair_mode: str | None = None,
    dry_run: bool = False,
) -> tuple[int, List[RemovedDetail]]:
    """
    De-duplicate across multiple files using a shared seen map.
    Returns total removed lines and removed line details.
    """
    seen: dict[str, tuple[Path, int, str]] = {}
    removed_details: List[RemovedDetail] = []
    removed_total = 0
    for path in paths:
        original_text = path.read_text(encoding="utf-8", errors="replace")
        lines = original_text.splitlines(keepends=True)

        kept: List[str] = []
        removed = 0

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue

            key = _dedup_key(stripped, pair_mode=pair_mode)
            if key in seen:
                removed += 1
                first_path, first_line_num, first_stripped = seen[key]
                removed_details.append(
                    (path, line_num, stripped, first_path, first_line_num, first_stripped)
                )
                continue

            seen[key] = (path, line_num, stripped)
            kept.append(line)

        new_text = "".join(kept)
        if not dry_run and new_text != original_text:
            tmp_path = path.with_name(path.name + ".tmp_dedup")
            tmp_path.write_text(new_text, encoding="utf-8", newline="")
            try:
                os.chmod(tmp_path, path.stat().st_mode)
            except OSError:
                pass
            os.replace(tmp_path, path)

        removed_total += removed
    return removed_total, removed_details


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


def pair_mode_for_group(group: str) -> str | None:
    """
    For 3Class groups:
    - Inheritance/Dependency -> ab
    - Composition/Aggregation -> bc
    Return None for 2Class or unmatched relations.
    """
    _, _, class_name = group.partition("/")
    if not class_name.startswith("3Class_"):
        return None

    relation = class_name.casefold()
    if "inheritance" in relation or "dependency" in relation:
        return "ab"
    if "composition" in relation or "aggregation" in relation:
        return "bc"
    return None


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
    parser.add_argument(
        "--show-removed",
        action="store_true",
        help="Print each removed line and where its first occurrence was kept.",
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
        pair_mode = pair_mode_for_group(group)
        removed, removed_details = dedup_files_in_place(
            paths,
            pair_mode=pair_mode,
            dry_run=args.dry_run,
        )
        if removed > 0:
            total_removed += removed
            prefix = "WOULD_DEDUP" if args.dry_run else "DEDUP"
            print(f"{prefix} {group}  removed_lines={removed}")
            if args.show_removed:
                for (
                    path,
                    line_num,
                    stripped,
                    first_path,
                    first_line_num,
                    first_stripped,
                ) in removed_details:
                    path_rel = path.relative_to(root)
                    first_path_rel = first_path.relative_to(root)
                    print(f"  REMOVED {path_rel}:{line_num} :: {stripped}")
                    print(
                        f"    FIRST  {first_path_rel}:{first_line_num} :: {first_stripped}"
                    )

    print(f"Scanned_files={total_files}  removed_lines_total={total_removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
