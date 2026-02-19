import re


# Maintain your commonly used models here.
# "name" is used in output filenames and JSONL metadata.
MODEL_SPECS = [
    {"name": "qwen3-8b", "path": "Qwen/Qwen3-VL-8B-Instruct"},
    {"name": "qwen3-4b", "path": "Qwen/Qwen3-VL-4B-Instruct"},
    {"name": "qwen3-2b", "path": "Qwen/Qwen3-VL-2B-Instruct"},
    {"name": "InternVL3.5-8B", "path": "OpenGVLab/InternVL3_5-8B-Instruct"},
    {"name": "InternVL3.5-4B", "path": "OpenGVLab/InternVL3_5-4B-Instruct"},
    {"name": "InternVL3.5-2B", "path": "OpenGVLab/InternVL3_5-2B-Instruct"},
    {"name": "InternVL3.5-30B", "path": "OpenGVLab/InternVL3_5-30B-A3B-Instruct"},
    {"name": "Qwen3-30B", "path": "Qwen/Qwen3-VL-30B-A3B-Instruct"},
]

def _slugify(text):
    value = (text or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "model"


def _name_from_model_path(model_path):
    if not model_path:
        return "model"
    tail = model_path.rstrip("/\\").split("/")[-1].split("\\")[-1]
    return _slugify(tail)


def resolve_models(model_names=None, model_paths=None):
    """
    Resolve models from registry names and/or direct paths.
    Returns a list of dicts: [{"name": "...", "path": "..."}, ...]
    """
    model_names = model_names or []
    model_paths = model_paths or []

    registry = {m["name"]: m for m in MODEL_SPECS}
    resolved = []

    if model_names:
        for name in model_names:
            if name not in registry:
                known = ", ".join(sorted(registry.keys())) or "(empty)"
                raise ValueError(f"Unknown model name '{name}'. Known models: {known}")
            resolved.append({"name": name, "path": registry[name]["path"]})
    elif not model_paths:
        resolved.extend({"name": m["name"], "path": m["path"]} for m in MODEL_SPECS)

    for path in model_paths:
        resolved.append({"name": _name_from_model_path(path), "path": path})

    dedup = []
    seen = set()
    for item in resolved:
        key = (item["name"], item["path"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup
