import json
import math
import numpy as np
import os
import re
import random
import logging
from ultralytics.utils import LOGGER
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

BATCH_SIZE = 4
GPU_PER = 0.65
N = 1
LOGGER.setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def load_model(model_name):
    processor = AutoProcessor.from_pretrained(model_name)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=4,   # Use 4 GPUs
        dtype="auto",
        trust_remote_code=True,
        gpu_memory_utilization=GPU_PER      # Automatically choose FP16/BF16
    )
    return llm, processor

def fmt_mean(value):
    if value is None:
        return "No Data"
    return f"{value:.3f}"

def _format_class_desc(nodes):
    if not nodes:
        return ""
    if len(nodes) == 1:
        return nodes[0]
    if len(nodes) == 2:
        return f"{nodes[0]} and {nodes[1]}"
    return f"{', '.join(nodes[:-1])}, and {nodes[-1]}"

def _build_relation_prompt(nodes, relation="inheritance", query_pair=(0, 1)):
    # Supports 2-node and 3-node inputs.
    # query_pair controls which two nodes are asked about, defaulting to (node1, node2).
    if len(nodes) < 2:
        raise ValueError(f"nodes must contain at least 2 elements, got: {nodes}")

    if len(query_pair) != 2:
        raise ValueError(f"query_pair must contain exactly 2 indices, got: {query_pair}")
    src_idx, dst_idx = query_pair
    if src_idx < 0 or dst_idx < 0 or src_idx >= len(nodes) or dst_idx >= len(nodes):
        raise ValueError(
            f"query_pair indices out of range for nodes length {len(nodes)}: {query_pair}"
        )

    node1 = nodes[src_idx]
    node2 = nodes[dst_idx]
    node3 = nodes[2] if len(nodes) >= 3 else None

    relation = (relation or "inheritance").strip().lower()
    class_desc = _format_class_desc(nodes)
    class_prefix = "two" if len(nodes) == 2 else "three"

    relation_specs = {
        "inheritance": {
            "relation_sentence": "An arrow with a hollow triangle head points from the subclass to the superclass, indicating the inheritance relationship.",
            "ask_sentence": f"Does class {node2} inherits from class {node1}?",
        },
        "aggregation": {
            "relation_sentence": "A line with a hollow diamond at the whole side points from the part to the whole, indicating the aggregation relationship.",
            "ask_sentence": f"Does class {node2} aggregates class {node1}?",
        },
        "composition": {
            "relation_sentence": "A line with a filled diamond at the whole side points from the part to the whole, indicating the composition relationship.",
            "ask_sentence": f"Does class {node2} is composed of class {node1}?",
        },
        "dependency": {
            "relation_sentence": "A dashed arrow points from the dependent class to the class it depends on, indicating the dependency relationship.",
            "ask_sentence": f"Does class {node2} depends on class {node1}?",
        },
    }
    if relation not in relation_specs:
        raise ValueError(f"Unsupported relation type: {relation}")

    spec = relation_specs[relation]
    question = (
        f"This is a UML diagram showing {relation} relationships. "
        f"The diagram contains {class_prefix} classes: {class_desc}. "
        "Each class is represented as a box, with the class name at the top. "
        f"{spec['relation_sentence']} "
        "You must treat the diagram as the only source of truth. "
        "Answer the question solely from the relationships explicitly depicted in the image. "
        "Do not use background knowledge, learned associations, or any “reasonable” assumptions about what the entities usually mean. "
        "Do not answer based on model priors or what seems likely. "
        f"The question is:\n {spec['ask_sentence']} "
        "You may reason privately, but do not reveal any reasoning or intermediate steps. Output only the final binary answer: True or False."
    )

    return question, {
        "node1": node1,
        "node2": node2,
        "node3": node3,
        "query_src_idx": src_idx,
        "query_dst_idx": dst_idx,
    }

def _build_relation_prompts(nodes, relation="inheritance", query_pair=(0, 1)):
    # If nodes form a chain with 3+ classes, generate one prompt for each adjacent pair.
    if len(nodes) >= 3:
        pairs = [(i, i + 1) for i in range(len(nodes) - 1)]
    else:
        pairs = [query_pair]
    return [_build_relation_prompt(nodes, relation=relation, query_pair=pair) for pair in pairs]


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

    # Reverse datasets generated by txt2wsd keep only out_wsd under reverse/<Dataset>/out_wsd.
    # In that case instances.jsonl is still maintained in the forward dataset root.
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
    # Standard location: <dataset_root>/<dataset_dir>/instances.jsonl
    candidates = [os.path.join(dataset_root, dataset_dir, "instances.jsonl")]

    # txt2wsd reverse layout: reverse/<dataset>/out_wsd, while instances.jsonl stays in ../<dataset>/.
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
    # Example: 2Class_Inheritance/Animal -> ("inheritance", "2")
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

def load_prompt(
    processor,
    dataset_dir,
    dataset_root=".",
    output_image_root="output_image",
    prepared_out_jsonl=None,
    relation="inheritance",
    query_pair=(0, 1)
):
    requests = []
    information = []

    instances_path = _resolve_instances_path(dataset_root, dataset_dir)
    image_map_cache = {}

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
            relation_for_row = (obj.get("template_id") or relation or "inheritance")
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
                tried_msg = "; ".join([f"{d} ids={ids}" for d, ids in tried])
                raise FileNotFoundError(
                    f"No image found for sample_id={sample_id}, source_id={source_id}. Tried: {tried_msg}"
                )
            prompt_items = _build_relation_prompts(
                nodes,
                relation=relation_for_row,
                query_pair=query_pair,
            )
            for q_idx, (user_text, triplet_slots) in enumerate(prompt_items, start=1):
                q_pair = (
                    triplet_slots.get("query_src_idx"),
                    triplet_slots.get("query_dst_idx"),
                )

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": user_text},
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Extract visual features from the same messages payload and wrap into a vLLM request.
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, image_patch_size=processor.image_processor.patch_size, return_video_kwargs=True, return_video_metadata=True)
                mm = {}
                if image_inputs is not None:
                    mm["image"] = image_inputs
                if video_inputs is not None:
                    mm["video"] = video_inputs

                req = {
                    "prompt": prompt,
                    "multi_modal_data": mm,
                    "mm_processor_kwargs": video_kwargs
                }
                requests.append(req)

                row = {
                    "id": sample_id,
                    "query_id": q_idx,
                    # "query_pair": list(q_pair),
                    "template_id": obj.get("template_id"),
                    "nodes": nodes,
                    "triplet_slots": triplet_slots,
                    "image_path": image_path,
                    "prompt": user_text
                }
                information.append(row)
                prepared_rows.append(row)

    if prepared_out_jsonl is not None:
        with open(prepared_out_jsonl, "w", encoding="utf8") as wf:
            for row in prepared_rows:
                json.dump(row, wf, ensure_ascii=False)
                wf.write("\n")

    return requests, information

def run_model(requests, information, batch_size, model, out_path = "running_outputs.jsonl"):
    predictions = []
    counter = 1
    for i in range(0, len(requests), batch_size):
        batch_prompts = requests[i:i+batch_size]
        batch_predictions = run_batch(batch_prompts, model)
        predictions.extend(batch_predictions)

        if (i // 200) == counter:
            print(f"\033[1;32m{i}\033[0m instances generated successfully")
            counter += 1

    print("Starting to compute...")
    save_outputs(predictions, information, out_path)
    return predictions

def run_batch(batch_prompts, model):
    predictions = []

    sampling_params = SamplingParams(
        max_tokens=1024,
        n=N,
        stop_token_ids=[]
    )

    outputs = model.generate(batch_prompts, sampling_params)
    # Keep multiple candidates per prompt for pass@k style metrics.
    predictions = [[o.text.strip() for o in output.outputs] for output in outputs]

    return predictions

def save_outputs(predictions, information, out_path):
    with open(out_path, "w", encoding="utf8") as f:
        for pred, info in zip(predictions, information):
            first_output = pred[0] if pred else ""
            obj = {
                "id": info["id"],
                "query_id": info.get("query_id"),
                "query_pair": info.get("query_pair"),
                "template_id": info.get("template_id"),
                "nodes": info.get("nodes"),
                "triplet_slots": info.get("triplet_slots"),
                "image_path": info.get("image_path"),
                "prompt": info.get("prompt"),
                "outputs": pred,
                "output": first_output
            }
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

def generate_outputs(
    model_path,
    out_path,
    dataset_dir=None,
    dataset_root=None,
    output_image_root=None,
    prepared_out_jsonl=None,
    relation="inheritance",
    query_pair=(0, 1),
):
    if dataset_root is None:
        dataset_root = os.path.dirname(os.path.abspath(__file__))
    if output_image_root is None:
        output_image_root = os.path.join(dataset_root, "output_image")

    if dataset_dir is None:
        dataset_dirs = _discover_dataset_dirs(dataset_root)
        if not dataset_dirs:
            raise FileNotFoundError(
                f"No datasets found under {dataset_root}. Expected folders starting with '2class' or '3class'."
            )
    else:
        dataset_dirs = [dataset_dir]

    llm, processor = load_model(model_path)
    grouped_requests = {}
    grouped_information = {}
    valid_relations = {"inheritance", "aggregation", "composition", "dependency"}

    for ds in dataset_dirs:
        relation_name, arity = _extract_relation_arity_from_dataset_dir(ds)
        if relation_name not in valid_relations or arity not in {"2", "3"}:
            print(f"[WARN] Skip dataset '{ds}': unknown group relation={relation_name}, arity={arity}")
            continue
        group_key = (relation_name, arity)
        try:
            reqs, infos = load_prompt(
                processor=processor,
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
            "No runnable datasets found. Please check output_image/*/<dataset>/out_wsd directories."
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
        for arity in ordered_arities:
            group_key = (relation_name, arity)
            group_requests = grouped_requests.get(group_key, [])
            group_information = grouped_information.get(group_key, [])
            group_out_path = _build_group_out_path(out_path, relation_name, arity)
            if not group_requests:
                print(f"[INFO] skip empty group {relation_name}_{arity} -> {group_out_path}")
                continue
            print(f"[INFO] Generating {relation_name}_{arity} -> {group_out_path}")
            run_model(group_requests, group_information, BATCH_SIZE, llm, group_out_path)
