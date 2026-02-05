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

    for three_file in three_dir.rglob("data.txt"):
        rel = three_file.relative_to(three_dir)
        two_file = two_dir / rel
        two_file.parent.mkdir(parents=True, exist_ok=True)

        lines = three_file.read_text(encoding="utf-8").splitlines()
        required = build_required_pairs(lines)

        two_file.write_text("\n".join(required) + ("\n" if required else ""), encoding="utf-8")

print("Sync complete.")
