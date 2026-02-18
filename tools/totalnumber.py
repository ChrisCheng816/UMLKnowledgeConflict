from pathlib import Path
from collections import defaultdict
import argparse
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT_DIR = PROJECT_ROOT
DEFAULT_TARGET_NAME = "data.txt"

def count_nonempty_lines(path: Path) -> int:
    """Count samples: non-empty lines only (after strip)."""
    cnt = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for line in f:
            if line.strip():
                cnt += 1
    return cnt

def infer_group(root: Path, file_path: Path) -> str:
    """
    Prefer UML bucket names like 2Class_* / 3Class_* in the relative path.
    Fallback to top-level directory under root.
    """
    rel = file_path.relative_to(root)
    parts = rel.parts
    for part in parts:
        if re.match(r"^[23]Class_", part):
            return part
    return parts[0] if len(parts) >= 2 else "(root)"

def resolve_split_root(base_root: Path, split: str) -> Path:
    split_dir_name = f"data_{split}"
    if base_root.name == split_dir_name:
        return base_root
    return base_root / split_dir_name


def print_counts_for_root(root_dir: Path, target_name: str, split_label: str) -> None:
    if not root_dir.exists() or not root_dir.is_dir():
        print(f"[WARN] split root not found: {root_dir}")
        return

    files = sorted([p for p in root_dir.rglob(target_name) if p.is_file()])

    if not files:
        print(f"[WARN] No {target_name} files found under: {root_dir}")
        return

    per_group_total = defaultdict(int)
    per_file = []

    for p in files:
        n = count_nonempty_lines(p)
        rel = p.relative_to(root_dir)
        grp = infer_group(root_dir, p)
        per_group_total[grp] += n
        per_file.append((grp, n, rel))

    grand_total = sum(n for _, n, _ in per_file)

    print("=" * 100)
    print(f"Split: {split_label}")
    print(f"Root: {root_dir}")
    print(f"Found {len(files)} file(s) named {target_name}")
    print("Note: empty/blank lines are NOT counted as samples.")
    print("=" * 100)

    # -------- Per-file listing --------
    print("Per-file counts")
    print("-" * 100)
    print("Lines\tGroup\tFile")
    print("-" * 100)
    for grp, n, rel in per_file:
        print(f"{n}\t{grp}\t{rel}")

    # -------- Per-group totals --------
    print("=" * 100)
    print("Per-group totals")
    print("-" * 100)
    print("TOTAL\tGroup")
    print("-" * 100)
    for grp in sorted(per_group_total.keys()):
        print(f"{per_group_total[grp]}\t{grp}")

    # -------- Grand total --------
    print("=" * 100)
    print(f"GRAND_TOTAL\t{grand_total}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count non-empty lines in files named target (default: data.txt)."
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT_DIR),
        help="Project root or a split root directory.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET_NAME,
        help="Target filename to search for.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["forward", "reverse"],
        default=["reverse", "forward"],
        help="Dataset splits to print. Default: reverse forward",
    )
    args = parser.parse_args()

    base_root = Path(args.root).expanduser().resolve()
    target_name = args.target

    for split in args.splits:
        split_root = resolve_split_root(base_root, split)
        print_counts_for_root(split_root, target_name, split_label=split)

if __name__ == "__main__":
    main()
