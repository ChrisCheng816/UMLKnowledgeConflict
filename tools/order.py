#!/usr/bin/env python3
"""
Reorder 3Class reverse data lines across subfolders.

Scope:
- Only process `data_reverse/3Class_*` groups.
- Reorder instances within each 3Class group only (no cross-group moves).

Rule:
- For each non-empty line in `data.txt`, use the first token as instance name.
- Find a suitable target subfolder by instance ownership:
  - Build ownership only from existing `data.txt` lines inside the same 3Class group.
- Move the line only when the owner resolves to exactly one subfolder.
- If no suitable owner (or ambiguous owner), keep the line unchanged.

Guarantee:
- Total non-empty line count is preserved.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorder instances among subfolders inside data_reverse/3Class_*."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Project root directory. Default: parent directory of this script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned moves without writing files.",
    )
    parser.add_argument(
        "--show-moves",
        action="store_true",
        help="Print each moved line with source and destination.",
    )
    return parser.parse_args()


def _read_data_txt(path: Path) -> List[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _write_data_txt(path: Path, lines: List[str]) -> None:
    if lines:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        path.write_text("", encoding="utf-8")


def build_owner_index(group_dir: Path) -> Dict[str, List[Path]]:
    owner_to_dirs: DefaultDict[str, List[Path]] = defaultdict(list)
    child_dirs = sorted([p for p in group_dir.iterdir() if p.is_dir()], key=lambda p: p.name.casefold())

    for child in child_dirs:
        owners: Set[str] = set()
        data_path = child / "data.txt"
        if data_path.exists() and data_path.is_file():
            for line in _read_data_txt(data_path):
                stripped = line.strip()
                if not stripped:
                    continue
                owners.add(stripped.split()[0].casefold())
        for owner in owners:
            owner_to_dirs[owner].append(child)

    return dict(owner_to_dirs)


def collect_group_data_files(group_dir: Path) -> List[Path]:
    return sorted([p for p in group_dir.rglob("data.txt") if p.is_file()], key=lambda p: str(p).casefold())


def reorder_group(
    group_dir: Path,
    dry_run: bool = False,
    show_moves: bool = False,
) -> Tuple[int, int, int]:
    owner_index = build_owner_index(group_dir)
    data_files = collect_group_data_files(group_dir)
    if not data_files:
        return 0, 0, 0

    file_to_lines: Dict[Path, List[str]] = {p: _read_data_txt(p) for p in data_files}
    output_lines: Dict[Path, List[str]] = {p: [] for p in data_files}

    before_non_empty = sum(1 for lines in file_to_lines.values() for line in lines if line.strip())
    moved = 0

    for src_file in data_files:
        src_lines = file_to_lines[src_file]

        for line_num, line in enumerate(src_lines, start=1):
            stripped = line.strip()
            if not stripped:
                output_lines[src_file].append(line)
                continue

            first_token = stripped.split()[0].casefold()
            candidates = owner_index.get(first_token, [])

            if len(candidates) != 1:
                output_lines[src_file].append(line)
                continue

            target_child = candidates[0]
            target_file = target_child / "data.txt"

            if target_file not in output_lines:
                output_lines[src_file].append(line)
                continue

            if target_file == src_file:
                output_lines[src_file].append(line)
                continue

            output_lines[target_file].append(line)
            moved += 1
            if show_moves:
                rel_src = src_file.relative_to(group_dir)
                rel_dst = target_file.relative_to(group_dir)
                print(f"  MOVE {rel_src}:{line_num} -> {rel_dst} :: {stripped}")

    after_non_empty = sum(1 for lines in output_lines.values() for line in lines if line.strip())
    if after_non_empty != before_non_empty:
        raise RuntimeError(
            f"Line count mismatch in {group_dir}: before={before_non_empty}, after={after_non_empty}"
        )

    changed_files = sum(1 for p in data_files if output_lines[p] != file_to_lines[p])
    if not dry_run:
        for p in data_files:
            if output_lines[p] != file_to_lines[p]:
                _write_data_txt(p, output_lines[p])

    return moved, changed_files, before_non_empty


def main() -> int:
    args = parse_args()
    root = (
        Path(args.root).expanduser().resolve()
        if args.root
        else Path(__file__).resolve().parent.parent
    )
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    reverse_root = root / "data_reverse"
    if not reverse_root.exists() or not reverse_root.is_dir():
        raise SystemExit(f"Missing directory: {reverse_root}")

    groups = sorted(
        [p for p in reverse_root.iterdir() if p.is_dir() and p.name.startswith("3Class_")],
        key=lambda p: p.name.casefold(),
    )
    if not groups:
        raise SystemExit(f"No 3Class groups found under: {reverse_root}")

    total_moved = 0
    total_changed_files = 0
    total_instances = 0

    for group_dir in groups:
        moved, changed_files, instances = reorder_group(
            group_dir,
            dry_run=args.dry_run,
            show_moves=args.show_moves,
        )
        total_moved += moved
        total_changed_files += changed_files
        total_instances += instances
        prefix = "WOULD_REORDER" if args.dry_run else "REORDER"
        print(
            f"{prefix} {group_dir.name}  moved_lines={moved}  changed_files={changed_files}  total_instances={instances}"
        )

    print(
        f"Scanned_groups={len(groups)}  moved_lines_total={total_moved}  changed_files_total={total_changed_files}  total_instances={total_instances}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
