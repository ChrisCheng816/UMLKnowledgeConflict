# sync_2class_from_3class.py
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sync 2Class data.txt from 3Class data.txt inside forward/reverse separately."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root path. Default: parent directory of this script.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["reverse"],
        default=["reverse"],
        help="Which dataset splits to process. Default: reverse",
    )
    return parser.parse_args()


def build_required_pairs(lines, pair_mode: str = "both"):
    required = []
    seen = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        ab = f"{parts[0]} {parts[1]}"
        bc = f"{parts[1]} {parts[2]}"

        if pair_mode in ("ab", "both") and ab not in seen:
            seen.add(ab)
            required.append(ab)
        if pair_mode in ("bc", "both") and bc not in seen:
            seen.add(bc)
            required.append(bc)
    return required


def sync_split(split_root: Path):
    if not split_root.exists():
        print(f"[WARN] split root not found: {split_root}")
        return

    for three_dir in split_root.iterdir():
        if not (three_dir.is_dir() and three_dir.name.startswith("3Class_")):
            continue

        two_name = three_dir.name.replace("3Class_", "2Class_", 1)
        two_dir = split_root / two_name

        relation = three_dir.name.casefold()
        if "inheritance" in relation or "dependency" in relation:
            pair_mode = "ab"
        elif "composition" in relation or "aggregation" in relation:
            pair_mode = "bc"
        else:
            pair_mode = "both"

        # 1) Collect all required 2-class pairs from this 3Class group.
        required_all = set()
        required_by_file = {}
        for three_file in three_dir.rglob("data.txt"):
            lines = three_file.read_text(encoding="utf-8").splitlines()
            required = build_required_pairs(lines, pair_mode=pair_mode)
            required_by_file[three_file] = required
            required_all.update(required)

        # 2) Collect existing 2Class pairs under this group.
        existing_all = set()
        two_files = list(two_dir.rglob("data.txt")) if two_dir.exists() else []
        for two_file in two_files:
            for line in two_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                existing_all.add(line)

        # 3) Remove extra lines in 2Class that are not required by current 3Class group.
        for two_file in two_files:
            kept = []
            for line in two_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                if line in required_all:
                    kept.append(line)
            two_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

        # 4) Add missing required pairs into the corresponding 2Class subfolder.
        for three_file, required in required_by_file.items():
            rel = three_file.relative_to(three_dir)
            two_file = two_dir / rel
            two_file.parent.mkdir(parents=True, exist_ok=True)

            current = []
            if two_file.exists():
                current = two_file.read_text(encoding="utf-8").splitlines()

            updated = list(current)
            for pair in required:
                if pair not in existing_all:
                    updated.append(pair)
                    existing_all.add(pair)

            two_file.write_text("\n".join(updated) + ("\n" if updated else ""), encoding="utf-8")


def main():
    args = parse_args()
    root = args.root.resolve() if args.root else Path(__file__).resolve().parent.parent

    split_to_dir = {
        "reverse": root / "data_reverse",
    }

    for split in args.splits:
        split_root = split_to_dir[split]
        print(f"[INFO] syncing split={split}, root={split_root}")
        sync_split(split_root)

    print("Sync complete.")


if __name__ == "__main__":
    main()
