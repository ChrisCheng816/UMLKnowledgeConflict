from pathlib import Path

# ====== CONFIG: set your root directory here ======
ROOT_DIR = Path(r"./")  # <-- change this
TARGET_NAME = "data.txt"
# ================================================

def count_lines(path: Path) -> int:
    cnt = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for _ in f:
            cnt += 1
    return cnt

def main() -> None:
    if not ROOT_DIR.exists() or not ROOT_DIR.is_dir():
        raise SystemExit(f"ROOT_DIR is not a valid directory: {ROOT_DIR}")

    files = sorted([p for p in ROOT_DIR.rglob(TARGET_NAME) if p.is_file()])

    if not files:
        print(f"No {TARGET_NAME} files found under: {ROOT_DIR}")
        return

    total = 0
    print(f"Root: {ROOT_DIR}")
    print(f"Found {len(files)} file(s) named {TARGET_NAME}")
    print("-" * 90)
    print("Lines\tFile")
    print("-" * 90)

    for p in files:
        n = count_lines(p)
        total += n
        rel = p.relative_to(ROOT_DIR)
        print(f"{n}\t{rel}")

    print("-" * 90)
    print(f"TOTAL\t{total}")

if __name__ == "__main__":
    main()
