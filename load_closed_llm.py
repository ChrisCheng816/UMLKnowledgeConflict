import base64
import json
import os
import random
import re
from typing import Any, Dict, List, Tuple

from apis import API_KEYS
from prompt_templates import (
    build_class_presence_prompt,
    build_relation_prompt,
    build_stage1_messages,
    build_stage2_messages_from_info,
    build_unified_system_prompt,
)

api_keys = API_KEYS()


def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at {image_path}. Please check the path.")


def get_info(model):
    if "claude" in model:
        api_key = api_keys.api_keys["ANTHROPIC_API_KEY"]
        base_url = "https://api.anthropic.com/v1/"
    elif "gpt" in model or "o" in model[0]:
        api_key = api_keys.api_keys["OPENAI_API_KEY"]
        base_url = "https://api.openai.com/v1"
    else:
        api_key = api_keys.api_keys["GEMINI_API_KEY"]
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    return api_key, base_url


def _format_class_desc(nodes):
    if not nodes:
        return ""
    if len(nodes) == 1:
        return nodes[0]
    if len(nodes) == 2:
        return f"{nodes[0]} and {nodes[1]}"
    return f"{', '.join(nodes[:-1])}, and {nodes[-1]}"


def _build_relation_prompt(nodes, relation="inheritance", query_pair=(0, 1)):
    return build_relation_prompt(nodes, relation=relation, query_pair=query_pair)


def _coerce_single_query_pair(query_pair):
    if query_pair is None:
        return (0, 1)
    if (
        isinstance(query_pair, (list, tuple))
        and len(query_pair) == 2
        and all(isinstance(item, (list, tuple)) for item in query_pair)
    ):
        return tuple(query_pair[0])
    return tuple(query_pair)


def _build_relation_prompts(nodes, relation="inheritance", query_pair=(0, 1)):
    query_pair = _coerce_single_query_pair(query_pair)
    return [_build_relation_prompt(nodes, relation=relation, query_pair=query_pair)]


def _detect_dataset_split(dataset_root):
    """
    Detect dataset split from path.

    Returns one of: "forward", "reverse", "mixed", "unknown".
    """
    if not dataset_root:
        return "unknown"
    normalized = os.path.normpath(os.path.abspath(dataset_root)).lower()
    parts = [p for p in normalized.split(os.sep) if p]
    if "data_reverse" in parts or "reverse" in parts:
        return "reverse"
    if "data_mixed" in parts or "mixed" in parts:
        return "mixed"
    if "data_forward" in parts or "forward" in parts:
        return "forward"
    return "unknown"


def _select_query_pair_for_task2(relation, dataset_split):
    """
    For 3-class rows, task2 always asks about the old 2-class node pair.

    Old-pair indices:
    - forward/mixed inheritance/dependency: (1, 2)
    - reverse inheritance/dependency: (0, 1)
    - forward/mixed aggregation/composition: (0, 1)
    - reverse aggregation/composition: (1, 2)
    """
    relation = (relation or "").strip().lower()
    if dataset_split == "reverse":
        if relation in {"inheritance", "dependency"}:
            return (0, 1)
        if relation in {"composition", "aggregation"}:
            return (1, 2)
    else:
        if relation in {"inheritance", "dependency"}:
            return (1, 2)
        if relation in {"composition", "aggregation"}:
            return (0, 1)
    return (0, 1)


def _build_class_presence_prompt(expected_count=None, relation="inheritance"):
    return build_class_presence_prompt(expected_count=expected_count, relation=relation)


def _build_unified_system_prompt():
    return build_unified_system_prompt()


def _build_stage2_chat_messages(info):
    return build_stage2_messages_from_info(info)


def _normalize_class_name(name):
    return re.sub(r"\s+", " ", str(name).strip()).lower()


def _strip_code_fence(text):
    text = (text or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text


def _dedupe_preserve(items):
    seen = set()
    out = []
    for item in items:
        key = _normalize_class_name(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(str(item).strip())
    return out


def _extract_class_names_from_text(text):
    raw = _strip_code_fence(text)
    if not raw:
        return []

    data = None
    try:
        data = json.loads(raw)
    except Exception:
        data = None

    items = []
    if isinstance(data, list):
        items = [x for x in data if isinstance(x, (str, int, float))]
    elif isinstance(data, dict):
        for key in ("classes", "class_names", "classNames", "names"):
            value = data.get(key)
            if isinstance(value, list):
                items = [x for x in value if isinstance(x, (str, int, float))]
                break
    else:
        parts = re.split(r"[\n,;]+", raw)
        for part in parts:
            cleaned = re.sub(r"^\s*[-*0-9.)\s]+", "", part).strip().strip("\"'`")
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in {"unknown", "none", "n/a", "true", "false", "[]"}:
                continue
            items.append(cleaned)

    return _dedupe_preserve([str(x).strip().strip("\"'`") for x in items if str(x).strip()])


def _validate_class_names_case_insensitive(expected_nodes, predicted_names):
    expected_set = {_normalize_class_name(n) for n in (expected_nodes or []) if str(n).strip()}
    predicted_set = {_normalize_class_name(n) for n in (predicted_names or []) if str(n).strip()}
    return bool(expected_set) and expected_set == predicted_set


def _collect_image_map(image_dir):
    image_map = {}
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    for name in os.listdir(image_dir):
        if not name.lower().endswith(".png"):
            continue
        m = re.match(r"^(\d+)_", name)
        if not m:
            continue
        image_id = str(int(m.group(1)))
        image_map[image_id] = os.path.join(image_dir, name)

    return image_map


def _resolve_case_insensitive_path(path):
    path = os.path.normpath(path)
    if os.path.exists(path):
        return path

    drive, tail = os.path.splitdrive(path)
    if os.path.isabs(path):
        cur = drive + os.sep if drive else os.sep
    else:
        cur = ""

    parts = [p for p in tail.split(os.sep) if p and p != "."]
    for part in parts:
        if not os.path.isdir(cur or "."):
            return None
        try:
            entries = os.listdir(cur or ".")
        except OSError:
            return None
        match = next((e for e in entries if e.lower() == part.lower()), None)
        if match is None:
            return None
        cur = os.path.join(cur, match) if cur else match
    return cur if os.path.exists(cur) else None


def _discover_dataset_dirs(root_dir):
    dataset_dirs = []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    for top in sorted(os.listdir(root_dir)):
        top_path = os.path.join(root_dir, top)
        if not os.path.isdir(top_path):
            continue
        top_lower = top.lower()
        if not (top_lower.startswith("2class") or top_lower.startswith("3class")):
            continue

        top_instances = os.path.join(top_path, "instances.jsonl")
        if os.path.isfile(top_instances):
            dataset_dirs.append(os.path.relpath(top_path, root_dir))
            continue

        for cur_root, _, files in os.walk(top_path):
            if "instances.jsonl" in files:
                dataset_dirs.append(os.path.relpath(cur_root, root_dir))

    if dataset_dirs:
        return sorted(dataset_dirs)

    reverse_only_dirs = []
    for top in sorted(os.listdir(root_dir)):
        top_path = os.path.join(root_dir, top)
        if not os.path.isdir(top_path):
            continue
        top_lower = top.lower()
        if not (top_lower.startswith("2class") or top_lower.startswith("3class")):
            continue
        if os.path.isdir(os.path.join(top_path, "out_wsd")):
            reverse_only_dirs.append(os.path.relpath(top_path, root_dir))

    if reverse_only_dirs:
        return sorted(reverse_only_dirs)

    return sorted(dataset_dirs)


def _resolve_instances_path(dataset_root, dataset_dir):
    candidates = [os.path.join(dataset_root, dataset_dir, "instances.jsonl")]

    dataset_root_abs = os.path.abspath(dataset_root)
    if os.path.basename(dataset_root_abs).lower() == "reverse":
        parent_root = os.path.dirname(dataset_root_abs)
        candidates.append(os.path.join(parent_root, dataset_dir, "instances.jsonl"))

    for p in candidates:
        resolved = _resolve_case_insensitive_path(p) or p
        if os.path.isfile(resolved):
            return resolved

    raise FileNotFoundError(
        f"instances.jsonl not found for dataset '{dataset_dir}'. Tried: {candidates}"
    )


def _extract_relation_arity_from_dataset_dir(dataset_dir):
    normalized = os.path.normpath(dataset_dir)
    parts = normalized.split(os.sep)
    if not parts:
        return "unknown", "unknown"
    top = parts[0]
    m = re.match(r"^([23])class_(.+)$", top, flags=re.IGNORECASE)
    if not m:
        return "unknown", "unknown"
    arity = m.group(1)
    relation = m.group(2).strip().lower() or "unknown"
    return relation, arity


def _build_group_out_path(out_path, relation, arity):
    root, ext = os.path.splitext(out_path)
    if not ext:
        ext = ".jsonl"
    return f"{root}_{relation}_{arity}{ext}"


def build_relation_prompt(
    nodes,
    relation="inheritance",
    query_pair=None,
    dataset_root=None,
):
    if query_pair is None:
        if len(nodes) >= 3:
            query_pair = _select_query_pair_for_task2(
                relation,
                _detect_dataset_split(dataset_root),
            )
        else:
            query_pair = (0, 1)

    return _build_relation_prompt(nodes, relation=relation, query_pair=query_pair)


def _build_chat_messages_for_client(prompt, image_path):
    base64_image = encode_image_to_base64(image_path)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        }
    ]


def _to_chat_completions_messages(messages):
    converted = []
    for msg in messages or []:
        role = msg.get("role", "user")
        content_items = []
        for item in msg.get("content", []):
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                content_items.append({"type": "text", "text": str(item.get("text", ""))})
            elif item_type == "image":
                image_path = item.get("image")
                if not image_path:
                    continue
                base64_image = encode_image_to_base64(str(image_path))
                content_items.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )
        converted.append({"role": role, "content": content_items})
    return converted


def _to_responses_input(messages):
    converted = []
    for msg in messages or []:
        role = msg.get("role", "user")
        content_items = []
        for item in msg.get("content", []):
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                content_items.append({"type": "input_text", "text": str(item.get("text", ""))})
            elif item_type == "image":
                image_path = item.get("image")
                if not image_path:
                    continue
                base64_image = encode_image_to_base64(str(image_path))
                content_items.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_image}",
                    }
                )
        converted.append({"role": role, "content": content_items})
    return converted


def run_vqa(model, prompt, image_path, messages=None):
    from openai import OpenAI

    api_key, base_url = get_info(model)
    if "o3" not in model:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    else:
        client = OpenAI(
            api_key=api_key,
        )

    if messages is None:
        messages = _build_chat_messages_for_client(prompt, image_path)

    if "codex" in model:
        response = client.responses.create(
            model=model,
            input=_to_responses_input(messages),
        )

        answer = response.output_text
    else:
        response = client.chat.completions.create(
            model=model,
            messages=_to_chat_completions_messages(messages),
        )

        answer = response.choices[0].message.content
    return answer


def run_relation_vqa(
    model,
    image_path,
    nodes,
    relation="inheritance",
    query_pair=None,
    dataset_root=None,
):
    prompt, triplet_slots = build_relation_prompt(
        nodes=nodes,
        relation=relation,
        query_pair=query_pair,
        dataset_root=dataset_root,
    )
    answer = run_vqa(model, prompt, image_path)
    return answer, prompt, triplet_slots


def load_prompt(
    dataset_dir,
    dataset_root=".",
    output_image_root="images",
    prepared_out_jsonl=None,
    relation="inheritance",
    query_pair=(0, 1),
):
    requests: List[Dict[str, str]] = []
    information: List[Dict[str, Any]] = []
    skipped_missing_image = 0

    instances_path = _resolve_instances_path(dataset_root, dataset_dir)
    image_map_cache: Dict[str, Dict[str, str]] = {}
    dataset_split = _detect_dataset_split(dataset_root)

    prepared_rows = []
    with open(instances_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sample_id = str(int(obj.get("id")))
            source_subset = obj.get("source_subset")
            source_id_raw = obj.get("source_id")
            source_id = sample_id if source_id_raw is None else str(int(source_id_raw))
            nodes = obj.get("nodes", [])
            relation_for_row = obj.get("template_id") or relation or "inheritance"
            image_candidates = []
            if source_subset:
                image_candidates.append(
                    (os.path.join(output_image_root, dataset_dir, source_subset, "out_wsd"), [source_id, sample_id])
                )
            image_candidates.append(
                (os.path.join(output_image_root, dataset_dir, "out_wsd"), [sample_id, source_id])
            )
            image_candidates.append(
                (os.path.join(output_image_root, "reverse", dataset_dir, "out_wsd"), [sample_id, source_id])
            )

            image_path = None
            tried = []
            for image_dir_candidate, id_candidates in image_candidates:
                image_dir = _resolve_case_insensitive_path(image_dir_candidate) or image_dir_candidate
                tried.append((image_dir, id_candidates))
                if not os.path.isdir(image_dir):
                    continue

                if image_dir not in image_map_cache:
                    image_map_cache[image_dir] = _collect_image_map(image_dir)
                image_map = image_map_cache[image_dir]

                for image_id in id_candidates:
                    image_path = image_map.get(image_id)
                    if image_path is not None:
                        break
                if image_path is not None:
                    break

            if image_path is None:
                skipped_missing_image += 1
                continue

            prompt_items = _build_relation_prompts(
                nodes,
                relation=relation_for_row,
                query_pair=(
                    _select_query_pair_for_task2(relation_for_row, dataset_split)
                    if len(nodes) >= 3
                    else query_pair
                ),
            )
            for q_idx, (user_text, triplet_slots) in enumerate(prompt_items, start=1):
                req = {"image_path": image_path, "prompt": user_text}
                requests.append(req)

                row = {
                    "id": sample_id,
                    "query_id": q_idx,
                    "template_id": obj.get("template_id"),
                    "nodes": nodes,
                    "triplet_slots": triplet_slots,
                    "image_path": image_path,
                    "prompt": user_text,
                }
                information.append(row)
                prepared_rows.append(row)

    if prepared_out_jsonl is not None:
        with open(prepared_out_jsonl, "w", encoding="utf8") as wf:
            for row in prepared_rows:
                json.dump(row, wf, ensure_ascii=False)
                wf.write("\n")

    if not requests:
        raise FileNotFoundError(
            f"No image-aligned samples found for dataset '{dataset_dir}'. "
            f"Skipped {skipped_missing_image} rows due to missing images."
        )

    return requests, information


def _build_gate_records(run_info):
    gate_records = {}
    for idx, info in enumerate(run_info):
        key = info.get("image_path")
        if not key:
            continue
        if key not in gate_records:
            nodes = info.get("nodes") or []
            relation = info.get("template_id") or "inheritance"
            gate_prompt = _build_class_presence_prompt(
                expected_count=len(nodes),
                relation=relation,
            )
            gate_records[key] = {
                "image_path": key,
                "nodes": nodes,
                "relation": relation,
                "prompt": gate_prompt,
                "request": {
                    "image_path": key,
                    "prompt": gate_prompt,
                    "messages": build_stage1_messages(
                        image_path=key,
                        user_text=gate_prompt,
                    ),
                },
                "indices": [],
            }
        gate_records[key]["indices"].append(idx)
    return gate_records


def _compute_gate_result(gate_records, gate_keys, gate_predictions):
    gate_result = {}
    for key, pred in zip(gate_keys, gate_predictions):
        first_output = pred[0] if pred else ""
        predicted_names = _extract_class_names_from_text(first_output)
        expected_nodes = gate_records[key]["nodes"]
        passed = _validate_class_names_case_insensitive(expected_nodes, predicted_names)
        gate_result[key] = {
            "passed": passed,
            "expected_class_names": expected_nodes,
            "predicted_class_names": predicted_names,
            "class_check_output": first_output,
            "class_check_prompt": gate_records[key].get("prompt", ""),
        }
    return gate_result


def _build_stage2_request_from_info(info):
    return {
        "image_path": info.get("image_path"),
        "prompt": info.get("prompt", ""),
        "messages": _build_stage2_chat_messages(info),
    }


def _run_requests(requests, model, progress_label="instances"):
    predictions = []
    for i, req in enumerate(requests):
        answer = run_vqa(
            model,
            req.get("prompt", ""),
            req.get("image_path"),
            messages=req.get("messages"),
        )
        predictions.append([str(answer).strip()])
        if i > 0 and (i % 200) == 0:
            print(f"\033[1;32m{i}\033[0m {progress_label} generated successfully")
    return predictions


def run_model(
    requests,
    information,
    model,
    num_runs=1,
    out_path="running_outputs.jsonl",
    model_name=None,
    model_path=None,
):
    run_count = max(1, int(num_runs))
    all_run_outputs = []
    all_run_checks = []

    for run_idx in range(run_count):
        random.seed(random.randint(1, 2**31 - 1))
        run_info = [dict(item) for item in information]

        gate_records = _build_gate_records(run_info)
        gate_keys = list(gate_records.keys())
        gate_requests = [gate_records[k]["request"] for k in gate_keys]
        print(f"[INFO][run {run_idx + 1}/{run_count}] Stage-1 class check: {len(gate_requests)} images")
        gate_predictions = _run_requests(
            gate_requests,
            model,
            progress_label=f"run-{run_idx + 1}-stage-1",
        )
        gate_result = _compute_gate_result(gate_records, gate_keys, gate_predictions)

        stage2_requests = []
        stage2_indices = []
        final_predictions = [None] * len(run_info)
        for idx, (_, info) in enumerate(zip(requests, run_info)):
            key = info.get("image_path")
            result = gate_result.get(key, {})
            passed = result.get("passed", False)
            info["class_check_passed"] = passed
            info["class_check_expected"] = result.get("expected_class_names", info.get("nodes", []))
            info["class_check_predicted"] = result.get("predicted_class_names", [])
            info["class_check_output"] = result.get("class_check_output", "")
            info["class_check_prompt"] = result.get("class_check_prompt", "")
            if passed:
                stage2_requests.append(_build_stage2_request_from_info(info))
                stage2_indices.append(idx)
            else:
                final_predictions[idx] = ["Unknown"]

        print(
            f"[INFO][run {run_idx + 1}/{run_count}] "
            f"Stage-2 relation QA: {len(stage2_requests)} prompts (after class check)"
        )
        stage2_predictions = _run_requests(
            stage2_requests,
            model,
            progress_label=f"run-{run_idx + 1}-stage-2",
        )
        for idx, pred in zip(stage2_indices, stage2_predictions):
            final_predictions[idx] = pred

        final_predictions = [pred if pred is not None else ["Unknown"] for pred in final_predictions]
        all_run_outputs.append([pred[0] if pred else "Unknown" for pred in final_predictions])
        all_run_checks.append(
            [
                {
                    "passed": info.get("class_check_passed"),
                    "expected_class_names": info.get("class_check_expected", []),
                    "predicted_class_names": info.get("class_check_predicted", []),
                    "class_check_output": info.get("class_check_output", ""),
                    "task1_prompt": info.get("class_check_prompt", ""),
                }
                for info in run_info
            ]
        )

    final_predictions = []
    for idx, info in enumerate(information):
        run_outputs = [all_run_outputs[r][idx] for r in range(run_count)]
        run_checks = [all_run_checks[r][idx] for r in range(run_count)]
        info["class_check_runs"] = run_checks
        info["class_check_passed"] = any(bool(check.get("passed", False)) for check in run_checks)
        info["class_check_expected"] = run_checks[0]["expected_class_names"] if run_checks else []
        info["class_check_predicted"] = run_checks[0]["predicted_class_names"] if run_checks else []
        info["class_check_output"] = run_checks[0]["class_check_output"] if run_checks else ""
        info["class_check_prompt"] = run_checks[0]["task1_prompt"] if run_checks else ""
        final_predictions.append(run_outputs)

    print(f"Starting to compute... (full-pipeline runs={run_count})")
    save_outputs(
        final_predictions,
        information,
        out_path,
        model_name=model_name,
        model_path=model_path,
    )
    return final_predictions


def save_outputs(predictions, information, out_path, model_name=None, model_path=None):
    def _to_bool_true(text):
        return str(text).strip().lower() == "true"

    def _pass_at_k_from_outputs(run_outputs, k):
        if not run_outputs:
            return None
        k = int(k)
        if len(run_outputs) < k:
            return None
        k = max(1, k)
        return any(_to_bool_true(x) for x in run_outputs[:k])

    def _pass_at_k_from_checks(run_checks, k):
        if not run_checks:
            return None
        k = int(k)
        if len(run_checks) < k:
            return None
        k = max(1, k)
        return any(bool((run_checks[i] or {}).get("passed", False)) for i in range(k))

    def _label(value):
        if value is None:
            return "Unknown"
        return "True" if value else "False"

    with open(out_path, "w", encoding="utf8") as f:
        for pred, info in zip(predictions, information):
            run_outputs = pred or []
            run_checks = info.get("class_check_runs") or []
            outputs_task1 = [bool((check or {}).get("passed", False)) for check in run_checks]
            predicted_class_names = [
                (check or {}).get("predicted_class_names", [])
                for check in run_checks
            ]
            expected_class_names = info.get("class_check_expected", info.get("nodes", []))
            task1_prompt = info.get("class_check_prompt")

            end2end_pass1 = _pass_at_k_from_outputs(run_outputs, 1)
            end2end_pass5 = _pass_at_k_from_outputs(run_outputs, 5)
            end2end_pass10 = _pass_at_k_from_outputs(run_outputs, 10)

            task1_pass1 = _pass_at_k_from_checks(run_checks, 1)
            task1_pass5 = _pass_at_k_from_checks(run_checks, 5)
            task1_pass10 = _pass_at_k_from_checks(run_checks, 10)

            obj = {
                "model_name": model_name,
                "model_path": model_path,
                "id": info["id"],
                "template_id": info.get("template_id"),
                "nodes": info.get("nodes"),
                "triplet_slots": info.get("triplet_slots"),
                "image_path": info.get("image_path"),
                "task2_prompt": info.get("prompt"),
                "task1_prompt": task1_prompt,
                "expected_class_names": expected_class_names,
                "predicted_class_names": predicted_class_names,
                "outputs_task1": outputs_task1,
                "outputs_task2": run_outputs,
                "pass@1": _label(end2end_pass1),
                "pass@5": _label(end2end_pass5),
                "pass@10": _label(end2end_pass10),
                "task1_pass@1": _label(task1_pass1),
                "task1_pass@5": _label(task1_pass5),
                "task1_pass@10": _label(task1_pass10),
            }
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")


def generate_outputs(
    model,
    out_path,
    model_name=None,
    dataset_dir=None,
    dataset_root=None,
    output_image_root=None,
    prepared_out_jsonl=None,
    relation="inheritance",
    query_pair=(0, 1),
    num_runs=1,
    include_relations=None,
    include_arities=None,
):
    if dataset_root is None:
        dataset_root = os.path.dirname(os.path.abspath(__file__))
    if output_image_root is None:
        output_image_root = os.path.join(dataset_root, "images")

    if dataset_dir is None:
        dataset_dirs = _discover_dataset_dirs(dataset_root)
        if not dataset_dirs:
            raise FileNotFoundError(
                f"No datasets found under {dataset_root}. Expected folders starting with '2class' or '3class'."
            )
    else:
        dataset_dirs = [dataset_dir]

    grouped_requests: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    grouped_information: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    valid_relations = {"inheritance", "aggregation", "composition", "dependency"}
    valid_arities = {"2", "3"}
    selected_relations = (
        {str(x).strip().lower() for x in include_relations}
        if include_relations is not None
        else set(valid_relations)
    )
    selected_arities = (
        {str(x).strip() for x in include_arities}
        if include_arities is not None
        else set(valid_arities)
    )
    selected_relations &= valid_relations
    selected_arities &= valid_arities
    if not selected_relations:
        raise ValueError("No valid relations selected. Allowed: inheritance, aggregation, composition, dependency.")
    if not selected_arities:
        raise ValueError("No valid arities selected. Allowed: 2, 3.")

    for ds in dataset_dirs:
        relation_name, arity = _extract_relation_arity_from_dataset_dir(ds)
        if relation_name not in valid_relations or arity not in valid_arities:
            print(f"[WARN] Skip dataset '{ds}': unknown group relation={relation_name}, arity={arity}")
            continue
        if relation_name not in selected_relations or arity not in selected_arities:
            continue
        group_key = (relation_name, arity)
        try:
            reqs, infos = load_prompt(
                dataset_dir=ds,
                dataset_root=dataset_root,
                output_image_root=output_image_root,
                prepared_out_jsonl=None,
                relation=relation,
                query_pair=query_pair,
            )
        except FileNotFoundError as e:
            print(f"[WARN] Skip dataset '{ds}': {e}")
            continue

        grouped_requests.setdefault(group_key, []).extend(reqs)
        grouped_information.setdefault(group_key, []).extend(infos)

    if not grouped_requests:
        raise FileNotFoundError(
            "No runnable datasets found. Please check images/*/<dataset>/out_wsd directories."
        )

    if prepared_out_jsonl is not None:
        with open(prepared_out_jsonl, "w", encoding="utf8") as wf:
            for group_key in sorted(grouped_information):
                for row in grouped_information[group_key]:
                    json.dump(row, wf, ensure_ascii=False)
                    wf.write("\n")

    ordered_relations = ["inheritance", "aggregation", "composition", "dependency"]
    ordered_arities = ["2", "3"]
    for relation_name in ordered_relations:
        if relation_name not in selected_relations:
            continue
        for arity in ordered_arities:
            if arity not in selected_arities:
                continue
            group_key = (relation_name, arity)
            group_requests = grouped_requests.get(group_key, [])
            group_information = grouped_information.get(group_key, [])
            group_out_path = _build_group_out_path(out_path, relation_name, arity)
            if not group_requests:
                print(f"[INFO] skip empty group {relation_name}_{arity} -> {group_out_path}")
                continue
            print(f"[INFO] Generating {relation_name}_{arity} -> {group_out_path}")
            run_model(
                group_requests,
                group_information,
                model,
                num_runs=num_runs,
                out_path=group_out_path,
                model_name=model_name,
                model_path=model,
            )
