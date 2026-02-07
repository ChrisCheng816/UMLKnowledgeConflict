from pathlib import Path
from collections import defaultdict

# ====== CONFIG: set your root directory here ======
ROOT_DIR = Path(r"../")  # <-- change this
TARGET_NAME = "data.txt"
# ================================================

def count_nonempty_lines(path: Path) -> int:
    """Count samples: non-empty lines only (after strip)."""
    cnt = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for line in f:
            if line.strip():
                cnt += 1
    return cnt

def top_level_group(root: Path, file_path: Path) -> str:
    """
    Return the top-level directory name under root.
    Example: root/3Class_Inheritance/Food/data.txt -> '3Class_Inheritance'
    If data.txt is directly under root, return '(root)'.
    """
    rel = file_path.relative_to(root)
    parts = rel.parts
    return parts[0] if len(parts) >= 2 else "(root)"

def main() -> None:
    if not ROOT_DIR.exists() or not ROOT_DIR.is_dir():
        raise SystemExit(f"ROOT_DIR is not a valid directory: {ROOT_DIR}")

    files = sorted([p for p in ROOT_DIR.rglob(TARGET_NAME) if p.is_file()])

    if not files:
        print(f"No {TARGET_NAME} files found under: {ROOT_DIR}")
        return

    per_group_total = defaultdict(int)
    per_file = []

    for p in files:
        n = count_nonempty_lines(p)
        rel = p.relative_to(ROOT_DIR)
        grp = top_level_group(ROOT_DIR, p)
        per_group_total[grp] += n
        per_file.append((grp, n, rel))

    grand_total = sum(n for _, n, _ in per_file)

    print(f"Root: {ROOT_DIR}")
    print(f"Found {len(files)} file(s) named {TARGET_NAME}")
    print("Note: empty/blank lines are NOT counted as samples.")
    print("=" * 100)

    # -------- Per-file listing --------
    print("Per-file counts")
    print("-" * 100)
    print("Lines\tGroup\tFile")
    print("-" * 100)
    for grp, n, rel in per_file:
        print(f"{n}\t{grp}\t{rel}")

    # -------- Per-top-level-group totals --------
    print("=" * 100)
    print("Per-top-level directory totals")
    print("-" * 100)
    print("TOTAL\tGroup")
    print("-" * 100)
    for grp in sorted(per_group_total.keys()):
        print(f"{per_group_total[grp]}\t{grp}")

    # -------- Grand total --------
    print("=" * 100)
    print(f"GRAND_TOTAL\t{grand_total}")

if __name__ == "__main__":
    main()
