# sync_2class_from_3class.py
import argparse
import os
import shutil
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


def remove_empty_parents(start_dir: Path, stop_dir: Path):
    """
    Remove empty parents from start_dir upwards until stop_dir (exclusive).
    """
    cur = start_dir
    while cur != stop_dir and cur.exists():
        try:
            next(cur.iterdir())
            break
        except StopIteration:
            cur.rmdir()
            cur = cur.parent


def safe_unlink(path: Path):
    """
    Best-effort unlink for read-only files on Windows.
    """
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except PermissionError:
        try:
            os.chmod(path, 0o666)
            path.unlink()
            return True
        except OSError:
            return False
    return False


def _handle_remove_readonly(func, path, exc_info):
    """
    rmtree callback: retry after making path writable.
    """
    try:
        os.chmod(path, 0o777)
        func(path)
    except OSError:
        pass


def safe_rmtree(path: Path):
    """
    Best-effort recursive delete for read-only trees on Windows.
    """
    if not path.exists():
        return True
    shutil.rmtree(path, onerror=_handle_remove_readonly)
    return not path.exists()


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
        # Also track expected data.txt layout relative to 3Class root.
        required_all = set()
        required_by_file = {}
        expected_rel_data_files = set()
        expected_rel_dirs = {Path(".")}
        for three_file in sorted(three_dir.rglob("data.txt")):
            lines = three_file.read_text(encoding="utf-8").splitlines()
            required = build_required_pairs(lines, pair_mode=pair_mode)
            required_by_file[three_file] = required
            required_all.update(required)
            rel_data = three_file.relative_to(three_dir)
            expected_rel_data_files.add(rel_data)
            parent = rel_data.parent
            expected_rel_dirs.add(parent)
            while parent != Path("."):
                parent = parent.parent
                expected_rel_dirs.add(parent)

        # 2) Align 2Class directory/file layout to 3Class layout.
        # Delete any extra 2Class data.txt not present in the 3Class tree.
        if two_dir.exists():
            for two_file in sorted(two_dir.rglob("data.txt")):
                rel = two_file.relative_to(two_dir)
                if rel not in expected_rel_data_files:
                    if not safe_unlink(two_file):
                        print(f"[WARN] cannot remove extra file: {two_file}")
                    remove_empty_parents(two_file.parent, two_dir)
            # Remove extra directories that are not present in 3Class tree.
            for two_subdir in sorted(
                (p for p in two_dir.rglob("*") if p.is_dir()),
                key=lambda p: len(p.parts),
                reverse=True,
            ):
                rel_dir = two_subdir.relative_to(two_dir)
                if rel_dir not in expected_rel_dirs:
                    if not safe_rmtree(two_subdir):
                        print(f"[WARN] cannot remove extra directory: {two_subdir}")
                    remove_empty_parents(two_subdir.parent, two_dir)

        # 3) Rewrite each corresponding 2Class data.txt with ordered pairs.
        # Order is aligned to the source 3Class data.txt order.
        for three_file, required in required_by_file.items():
            rel = three_file.relative_to(three_dir)
            two_file = two_dir / rel
            two_file.parent.mkdir(parents=True, exist_ok=True)
            updated = list(required)
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
