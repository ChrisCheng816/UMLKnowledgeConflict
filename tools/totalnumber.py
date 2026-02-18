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


def count_files(root_dir: Path, pattern: str) -> list[Path]:
    return sorted([p for p in root_dir.rglob(pattern) if p.is_file()])


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


def print_counts_for_root(root_dir: Path, target_name: str, split_label: str, details: bool = False) -> None:
    if not root_dir.exists() or not root_dir.is_dir():
        print(f"[WARN] split root not found: {root_dir}")
        return

    files = count_files(root_dir, target_name)
    # Forward-side instances.jsonl exists at both:
    # 1) group root (merged total) and 2) subgroup folders (per-category).
    # Keep only subgroup files to avoid double-counting merged totals.
    if split_label == "forward" and target_name == "instances.jsonl":
        files = [
            p for p in files
            if not re.match(r"^[23]Class_", p.parent.name)
        ]

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

    if not details:
        print(f"[{split_label}]")
        for grp in sorted(per_group_total.keys()):
            print(f"{per_group_total[grp]}\t{grp}")
        return

    grand_total = sum(n for _, n, _ in per_file)
    print()
    print("#" * 40 + f" BEGIN {split_label.upper()} " + "#" * 40)
    print("=" * 100)
    print(f"Split: {split_label}")
    print(f"Root: {root_dir}")
    print(f"Found {len(files)} file(s) named {target_name}")
    print("Note: empty/blank lines are NOT counted as samples.")
    print("=" * 100)
    print("Per-file counts")
    print("-" * 100)
    print("Lines\tGroup\tFile")
    print("-" * 100)
    for grp, n, rel in per_file:
        print(f"{n}\t{grp}\t{rel}")
    print("=" * 100)
    print("Per-group totals")
    print("-" * 100)
    print("TOTAL\tGroup")
    print("-" * 100)
    for grp in sorted(per_group_total.keys()):
        print(f"{per_group_total[grp]}\t{grp}")
    print("=" * 100)
    print(f"GRAND_TOTAL\t{grand_total}")
    print("#" * 40 + f" END {split_label.upper()} " + "#" * 42)
    print()

def print_png_counts_for_root(root_dir: Path, split_label: str, details: bool = False) -> None:
    if not root_dir.exists() or not root_dir.is_dir():
        print(f"[WARN] split root not found: {root_dir}")
        return

    files = count_files(root_dir, "*.png")
    if not files:
        print(f"[WARN] No .png files found under: {root_dir}")
        return

    per_group_total = defaultdict(int)
    per_file = []
    for p in files:
        rel = p.relative_to(root_dir)
        grp = infer_group(root_dir, p)
        per_group_total[grp] += 1
        per_file.append((grp, 1, rel))

    if not details:
        print(f"[images_{split_label}]")
        for grp in sorted(per_group_total.keys()):
            print(f"{per_group_total[grp]}\t{grp}")
        return

    print()
    print("#" * 35 + f" BEGIN IMAGES_{split_label.upper()} " + "#" * 35)
    print("=" * 100)
    print(f"Split: images_{split_label}")
    print(f"Root: {root_dir}")
    print(f"Found {len(files)} file(s) named *.png")
    print("=" * 100)
    print("Per-group totals")
    print("-" * 100)
    print("TOTAL\tGroup")
    print("-" * 100)
    for grp in sorted(per_group_total.keys()):
        print(f"{per_group_total[grp]}\t{grp}")
    print("=" * 100)
    print(f"GRAND_TOTAL\t{len(files)}")
    print("#" * 35 + f" END IMAGES_{split_label.upper()} " + "#" * 37)
    print()


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
        help="Default target filename for non-forward splits.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["forward", "reverse"],
        default=["reverse", "forward"],
        help="Dataset splits to print. Default: reverse forward",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Enable detailed output (per-file list, headers, and grand total).",
    )
    args = parser.parse_args()

    base_root = Path(args.root).expanduser().resolve()
    images_root = base_root / "images" if base_root.name != "images" else base_root

    for split in args.splits:
        split_root = resolve_split_root(base_root, split)
        target_name = "instances.jsonl" if split == "forward" else args.target
        print_counts_for_root(split_root, target_name, split_label=split, details=args.details)
        image_split_root = resolve_split_root(images_root, split)
        print_png_counts_for_root(image_split_root, split_label=split, details=args.details)

if __name__ == "__main__":
    main()
