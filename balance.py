# sync_2class_from_3class.py
from pathlib import Path

ROOT = Path(r"C:\project\UMLKnowledgeConflict")


def build_required_pairs(lines):
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
        if ab not in seen:
            seen.add(ab)
            required.append(ab)
        if bc not in seen:
            seen.add(bc)
            required.append(bc)
    return required


for three_dir in ROOT.iterdir():
    if not (three_dir.is_dir() and three_dir.name.startswith("3Class_")):
        continue

    two_name = three_dir.name.replace("3Class_", "2Class_", 1)
    two_dir = ROOT / two_name

    # 1) 收集该组所有 3Class 的要求二元组
    required_all = set()
    required_by_file = {}
    for three_file in three_dir.rglob("data.txt"):
        lines = three_file.read_text(encoding="utf-8").splitlines()
        required = build_required_pairs(lines)
        required_by_file[three_file] = required
        required_all.update(required)

    # 2) 收集该组 2Class 现有的二元组（跨子目录）
    existing_all = set()
    two_files = list(two_dir.rglob("data.txt")) if two_dir.exists() else []
    for two_file in two_files:
        for line in two_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            existing_all.add(line)

    # 3) 删除 2Class 中不在 required_all 的多余行（跨子目录删除）
    for two_file in two_files:
        kept = []
        for line in two_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line in required_all:
                kept.append(line)
        two_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    # 4) 将 3Class 中缺失的二元组添加到对应的 2Class 同名子目录
    for three_file, required in required_by_file.items():
        rel = three_file.relative_to(three_dir)
        two_file = two_dir / rel
        two_file.parent.mkdir(parents=True, exist_ok=True)

        # 读取当前（可能已清理过的）内容
        current = []
        if two_file.exists():
            current = two_file.read_text(encoding="utf-8").splitlines()

        updated = list(current)
        for pair in required:
            if pair not in existing_all:
                updated.append(pair)
                existing_all.add(pair)

        two_file.write_text("\n".join(updated) + ("\n" if updated else ""), encoding="utf-8")


print("Sync complete.")
