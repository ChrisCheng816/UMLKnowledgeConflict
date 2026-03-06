#!/usr/bin/env python3
"""Sanitize outputs_task2 for evaluation without touching original results.

Key guarantees:
- Never modifies files under ./Test/experiment_results.
- Writes a repaired mirror dataset and mapping reports under ./Test/eval.
- Converts all observed dirty values to canonical labels: True / False / Unknown.
- If a value is unresolved/ambiguous, it is repaired conservatively to Unknown.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

VALID_LABELS = {"True", "False", "Unknown"}

TRUE_TOKENS = {
    "true",
    "yes",
    "y",
    "correct",
    "affirmative",
    "1",
    "true.",
}
FALSE_TOKENS = {
    "false",
    "no",
    "n",
    "incorrect",
    "negative",
    "0",
    "false.",
}
UNKNOWN_TOKENS = {
    "unknown",
    "_unknown",
    "unk",
    "unknown.",
    "uncertain",
    "uncertain.",
    "unclear",
    "unclear.",
    "undetermined",
    "unsure",
    "not known",
    "not determined",
    "indeterminate",
    "cannot say",
    "can't say",
    "cant say",
    "cannot tell",
    "can't tell",
    "cant tell",
    "unable to determine",
    "insufficient information",
    "not enough information",
    "not sure",
    "cannot determine",
    "can't determine",
    "cant determine",
    "i don't know",
    "idk",
    "?",
    "n/a",
    "na",
    "none",
    "null",
    "nil",
}

TEXT_TOKEN_RE = re.compile(r"\b(true|false|unknown)\b", flags=re.IGNORECASE)
TEXT_SOFT_TOKEN_RE = re.compile(
    r"\b(yes|no|correct|incorrect|affirmative|negative)\b",
    flags=re.IGNORECASE,
)
TEXT_DECISION_ANCHOR_RE = re.compile(
    r"(?:final\s+answer|now\s+(?:i\s+)?answer\s+is|answer\s+is|revised\s+answer)\s*[:：\\-]?\s*"
    r"(true|false|unknown|yes|no|correct|incorrect|affirmative|negative)\b",
    flags=re.IGNORECASE,
)

SOFT_TOKEN_TO_LABEL = {
    "yes": "True",
    "correct": "True",
    "affirmative": "True",
    "no": "False",
    "incorrect": "False",
    "negative": "False",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair outputs_task2 into True/False/Unknown for eval only."
    )
    parser.add_argument(
        "--input-root",
        default="/scratch/zcheng06/Test/experiment_results",
        help="Path to original experiment_results root (read-only input).",
    )
    parser.add_argument(
        "--output-root",
        default="/scratch/zcheng06/Test/eval/repaired_eval_view",
        help="Output root for repaired mirror + reports.",
    )
    parser.add_argument(
        "--write-clean-jsonl",
        action="store_true",
        default=True,
        help="Write repaired mirror JSONL files (default: enabled).",
    )
    parser.add_argument(
        "--no-write-clean-jsonl",
        dest="write_clean_jsonl",
        action="store_false",
        help="Only write reports, no repaired JSONL mirror.",
    )
    parser.add_argument(
        "--max-examples-per-raw",
        type=int,
        default=10,
        help="Max location examples to keep per raw dirty value in summary report.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def write_csv(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _json_loads_or_none(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_candidate(value: Any, depth: int = 0) -> Any:
    """Extract likely scalar from nested objects/lists for label normalization."""
    if depth > 6:
        return value

    if value is None:
        return ""

    if isinstance(value, bool):
        return "True" if value else "False"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, list):
        if len(value) == 1:
            return _extract_candidate(value[0], depth + 1)
        return value

    if isinstance(value, dict):
        for k in ("output", "answer", "final_answer", "label", "response", "final"):
            if k in value:
                return _extract_candidate(value[k], depth + 1)
        if len(value) == 1:
            only_val = next(iter(value.values()))
            return _extract_candidate(only_val, depth + 1)
        return value

    return str(value)


def _normalize_token_text(text: str) -> str:
    t = text.strip()
    t = t.strip("`")
    t = t.strip()
    t = t.strip("\"'")
    t = t.strip()
    return t


def normalize_task2_value(raw_value: Any) -> Tuple[str, str]:
    """Return (canonical_label, rule).

    canonical_label in: True / False / Unknown
    """
    v = _extract_candidate(raw_value)

    # If list with multiple entries, try map each and keep unanimous mapping.
    if isinstance(v, list):
        mapped = [normalize_task2_value(x)[0] for x in v]
        mapped_valid = {m for m in mapped if m in VALID_LABELS}
        if len(mapped_valid) == 1:
            return next(iter(mapped_valid)), "list_unanimous"
        if len(mapped_valid) > 1:
            return "Unknown", "list_conflict_fallback_unknown"
        return "Unknown", "list_no_valid_fallback_unknown"

    # If dict remains here, stringify and continue with text heuristics.
    if isinstance(v, dict):
        v = json.dumps(v, ensure_ascii=True, sort_keys=True)

    text = str(v)
    text_norm = _normalize_token_text(text)
    if not text_norm:
        return "Unknown", "empty_fallback_unknown"

    # 1) direct exact
    if text_norm in VALID_LABELS:
        return text_norm, "exact"

    low = text_norm.lower()
    low_trim = low.rstrip(".!?;: ")

    # 2) strict dictionary mapping
    if low in TRUE_TOKENS or low_trim in TRUE_TOKENS:
        return "True", "dict_token"
    if low in FALSE_TOKENS or low_trim in FALSE_TOKENS:
        return "False", "dict_token"
    if low in UNKNOWN_TOKENS or low_trim in UNKNOWN_TOKENS:
        return "Unknown", "dict_token"

    # 3) JSON-encoded content (e.g. '"Unknown"', '["True"]', "{'output':'False'}")
    parsed = _json_loads_or_none(text_norm)
    if parsed is None and text_norm.startswith("[") and text_norm.endswith("]") and "'" in text_norm:
        parsed = _json_loads_or_none(text_norm.replace("'", "\""))
    if parsed is not None:
        mapped, _ = normalize_task2_value(parsed)
        if mapped in VALID_LABELS:
            return mapped, "json_unwrap"

    # 4) Anchored final-decision text (prefer the last explicit answer phrase).
    anchored = [m.group(1).lower() for m in TEXT_DECISION_ANCHOR_RE.finditer(low)]
    if anchored:
        last = anchored[-1]
        if last in {"true", "false", "unknown"}:
            return last.capitalize(), "text_decision_anchor"
        if last in SOFT_TOKEN_TO_LABEL:
            return SOFT_TOKEN_TO_LABEL[last], "text_decision_anchor_soft"

    # 4) long text fallback: look for explicit True/False/Unknown token
    tokens = [m.group(1).lower() for m in TEXT_TOKEN_RE.finditer(low)]
    if tokens:
        unique = set(tokens)
        if len(unique) == 1:
            only = next(iter(unique))
            return only.capitalize(), "text_single_token"
        # If mixed tokens appear, prefer last token only when it repeats at end.
        last = tokens[-1]
        if tokens.count(last) >= 2:
            return last.capitalize(), "text_last_token_majority"
        return "Unknown", "text_conflict_fallback_unknown"

    # 5) Secondary text tokens (yes/no/correct/incorrect/affirmative/negative).
    soft_tokens = [m.group(1).lower() for m in TEXT_SOFT_TOKEN_RE.finditer(low)]
    if soft_tokens:
        mapped = [SOFT_TOKEN_TO_LABEL[t] for t in soft_tokens if t in SOFT_TOKEN_TO_LABEL]
        mapped_unique = set(mapped)
        if len(mapped_unique) == 1:
            return mapped[0], "text_soft_token"
        if len(mapped_unique) > 1:
            last = mapped[-1]
            if mapped.count(last) >= 2:
                return last, "text_soft_last_token_majority"
            return "Unknown", "text_soft_conflict_fallback_unknown"

    return "Unknown", "unresolved_fallback_unknown"


def sanitize_outputs_task2(outputs: Any) -> Tuple[List[str], List[int], List[str]]:
    """Return (repaired_labels, fallback_indices_1based, rules_by_index)."""
    if not isinstance(outputs, list):
        outputs = [outputs]

    cleaned: List[str] = []
    fallback_idx: List[int] = []
    rules: List[str] = []
    for i, v in enumerate(outputs, start=1):
        label, rule = normalize_task2_value(v)
        cleaned.append(label)
        rules.append(rule)
        if "fallback" in rule:
            fallback_idx.append(i)
    return cleaned, fallback_idx, rules


def iter_jsonl_files(input_root: str) -> Iterable[Tuple[str, str]]:
    for model in sorted(os.listdir(input_root)):
        model_dir = os.path.join(input_root, model)
        if not os.path.isdir(model_dir):
            continue
        for name in sorted(os.listdir(model_dir)):
            if not name.endswith(".jsonl"):
                continue
            in_path = os.path.join(model_dir, name)
            rel = os.path.relpath(in_path, input_root)
            yield in_path, rel


def main() -> int:
    args = parse_args()
    input_root = args.input_root
    output_root = args.output_root

    if not os.path.isdir(input_root):
        print(f"Input root does not exist: {input_root}")
        return 2

    ensure_dir(output_root)
    clean_root = os.path.join(output_root, "repaired_jsonl")
    reports_root = os.path.join(output_root, "reports")
    ensure_dir(reports_root)

    total_files = 0
    total_rows = 0
    total_outputs = 0

    canonical_counts = Counter()
    rule_counts = Counter()
    raw_counts = Counter()
    raw_to_canonical = defaultdict(Counter)
    dirty_location_examples = defaultdict(list)
    dirty_counts_by_model = Counter()
    dirty_counts_by_file = Counter()

    for in_path, rel in iter_jsonl_files(input_root):
        total_files += 1
        model = rel.split("/", 1)[0]
        out_path = os.path.join(clean_root, rel)

        if args.write_clean_jsonl:
            ensure_dir(os.path.dirname(out_path))
            out_f = open(out_path, "w", encoding="utf-8")
        else:
            out_f = None

        try:
            with open(in_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    total_rows += 1
                    obj = json.loads(line)
                    outputs = obj.get("outputs_task2", [])
                    if not isinstance(outputs, list):
                        outputs = [outputs]

                    cleaned, fallback_idx, rules = sanitize_outputs_task2(outputs)
                    total_outputs += len(cleaned)

                    valid_cleaned = list(cleaned)
                    for lbl in cleaned:
                        canonical_counts[lbl] += 1
                    for rule in rules:
                        rule_counts[rule] += 1

                    for idx, raw_v in enumerate(outputs, start=1):
                        raw_text = str(raw_v)
                        raw_counts[raw_text] += 1
                        mapped_lbl = cleaned[idx - 1]
                        raw_to_canonical[raw_text][mapped_lbl] += 1

                        # Dirty means original raw not exact canonical.
                        if raw_text not in VALID_LABELS:
                            dirty_counts_by_model[model] += 1
                            dirty_counts_by_file[rel] += 1
                            if len(dirty_location_examples[raw_text]) < args.max_examples_per_raw:
                                sample_id = str(obj.get("id"))
                                dirty_location_examples[raw_text].append(
                                    {
                                        "file": rel,
                                        "line_no": line_no,
                                        "sample_id": sample_id,
                                        "output_idx": idx,
                                        "mapped_to": mapped_lbl,
                                        "rule": rules[idx - 1],
                                    }
                                )

                    if out_f is not None:
                        clean_obj = dict(obj)
                        clean_obj["outputs_task2_clean"] = cleaned
                        clean_obj["outputs_task2_repaired"] = cleaned
                        clean_obj["outputs_task2_valid"] = valid_cleaned
                        clean_obj["outputs_task2_fallback_count"] = len(fallback_idx)
                        clean_obj["outputs_task2_fallback_indices"] = fallback_idx
                        clean_obj["outputs_task2_clean_rules"] = rules
                        clean_obj["outputs_task2_repair_rules"] = rules
                        out_f.write(json.dumps(clean_obj, ensure_ascii=True) + "\n")
        finally:
            if out_f is not None:
                out_f.close()

    summary = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": input_root,
        "output_root": output_root,
        "write_clean_jsonl": bool(args.write_clean_jsonl),
        "guarantee": "Original files in input_root are never modified.",
        "total_files": total_files,
        "total_rows": total_rows,
        "total_outputs": total_outputs,
        "canonical_counts": dict(canonical_counts),
        "canonical_rates": {
            k: (canonical_counts[k] / total_outputs if total_outputs else None)
            for k in ["True", "False", "Unknown"]
        },
        "rule_counts": dict(rule_counts),
        "dirty_counts_by_model": dict(dirty_counts_by_model),
    }
    write_json(os.path.join(reports_root, "repair_summary.json"), summary)

    raw_rows: List[Dict[str, Any]] = []
    for raw_text, cnt in raw_counts.most_common():
        mapped = raw_to_canonical[raw_text]
        raw_rows.append(
            {
                "raw_value": raw_text,
                "count": cnt,
                "mapped_true": mapped.get("True", 0),
                "mapped_false": mapped.get("False", 0),
                "mapped_unknown": mapped.get("Unknown", 0),
                "example_locations": json.dumps(dirty_location_examples.get(raw_text, []), ensure_ascii=True),
            }
        )
    write_csv(
        os.path.join(reports_root, "repair_raw_value_mapping.csv"),
        raw_rows,
        [
            "raw_value",
            "count",
            "mapped_true",
            "mapped_false",
            "mapped_unknown",
            "example_locations",
        ],
    )

    by_file_rows = [
        {"file": file_rel, "dirty_count": cnt}
        for file_rel, cnt in dirty_counts_by_file.most_common()
    ]
    write_csv(
        os.path.join(reports_root, "repair_dirty_count_by_file.csv"),
        by_file_rows,
        ["file", "dirty_count"],
    )

    by_model_rows = [
        {"model": model, "dirty_count": cnt}
        for model, cnt in dirty_counts_by_model.most_common()
    ]
    write_csv(
        os.path.join(reports_root, "repair_dirty_count_by_model.csv"),
        by_model_rows,
        ["model", "dirty_count"],
    )

    print(f"Done. Repaired eval view written to: {output_root}")
    print("Original experiment_results were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
