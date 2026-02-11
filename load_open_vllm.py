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
GPU_PER = 0.85
LOGGER.setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)


def to_png(n, width=6, ext="png", dataset=0):
    return f"{MAPPING_DATASET[dataset]}/{int(n):0{width}d}.{ext}"

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
    class_desc = ", ".join(nodes)
    class_prefix = "two" if len(nodes) == 2 else "three"
    triplet_context = f" The third class is {node3}." if node3 is not None else ""

    if relation == "inheritance":
        question = f"This is a UML diagram showing inheritance relationships. The diagram contains {class_prefix} classes: {class_desc}. Each class is represented as a box, with the class name at the top. An arrow with a hollow triangle head points from the subclass to the superclass, indicating the inheritance relationship.{triplet_context} Please analyze the diagram and determine whether class {node2} inherits from class {node1}. Do not output any reasoning or thought steps; output only the final binary answer: True or False."
    elif relation == "aggregation":
        question = f"This is a UML diagram showing aggregation relationships. The diagram contains {class_prefix} classes: {class_desc}. Each class is represented as a box, with the class name at the top. A line with a hollow diamond at the whole side indicates aggregation (whole-part relationship).{triplet_context} Please analyze the diagram and determine whether class {node2} aggregates class {node1}. Do not output any reasoning or thought steps; output only the final binary answer: True or False."
    elif relation == "composition":
        question = f"This is a UML diagram showing composition relationships. The diagram contains {class_prefix} classes: {class_desc}. Each class is represented as a box, with the class name at the top. A line with a filled diamond at the whole side indicates composition (strong whole-part ownership).{triplet_context} Please analyze the diagram and determine whether class {node2} is composed of class {node1}. Do not output any reasoning or thought steps; output only the final binary answer: True or False."
    elif relation == "dependency":
        question = f"This is a UML diagram showing dependency relationships. The diagram contains {class_prefix} classes: {class_desc}. Each class is represented as a box, with the class name at the top. A dashed arrow points from the dependent class to the class it depends on.{triplet_context} Please analyze the diagram and determine whether class {node2} depends on class {node1}. Do not output any reasoning or thought steps; output only the final binary answer: True or False."
    else:
        raise ValueError(f"Unsupported relation type: {relation}")

    return question, {
        "node1": node1,
        "node2": node2,
        "node3": node3,
        "query_src_idx": src_idx,
        "query_dst_idx": dst_idx,
    }


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

        for cur_root, _, files in os.walk(top_path):
            if "instances.jsonl" in files:
                dataset_dirs.append(os.path.relpath(cur_root, root_dir))

    return sorted(dataset_dirs)

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

    instances_path = os.path.join(dataset_root, dataset_dir, "instances.jsonl")
    image_dir_candidate = os.path.join(output_image_root, dataset_dir, "out_wsd")
    image_dir = _resolve_case_insensitive_path(image_dir_candidate) or image_dir_candidate
    image_map = _collect_image_map(image_dir)

    prepared_rows = []
    with open(instances_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sample_id = str(int(obj.get("id")))
            nodes = obj.get("nodes", [])
            relation_for_row = (obj.get("template_id") or relation or "inheritance")
            image_path = image_map.get(sample_id)
            if image_path is None:
                raise FileNotFoundError(
                    f"No image found for id={sample_id}. Expected a file like '{sample_id}_*.png' in {image_dir}."
                )
            if len(nodes) == 3:
                query_pairs = [(0, 1), (1, 2)]
            else:
                query_pairs = [query_pair]

            for q_idx, q_pair in enumerate(query_pairs, start=1):
                user_text, triplet_slots = _build_relation_prompt(
                    nodes,
                    relation=relation_for_row,
                    query_pair=q_pair,
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
                    "query_pair": list(q_pair),
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
        temperature=0,
        stop_token_ids=[]
    )

    outputs = model.generate(batch_prompts, sampling_params)
    # Decode input and output to strings

    predictions = [output.outputs[0].text.strip() for output in outputs]

    return predictions

def save_outputs(predictions, information, out_path):
    with open(out_path, "w", encoding="utf8") as f:
        for pred, info in zip(predictions, information):
            obj = {
                "id": info["id"],
                "query_id": info.get("query_id"),
                "query_pair": info.get("query_pair"),
                "template_id": info.get("template_id"),
                "nodes": info.get("nodes"),
                "triplet_slots": info.get("triplet_slots"),
                "image_path": info.get("image_path"),
                "prompt": info.get("prompt"),
                "output": pred
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

    requests = []
    information = []
    for ds in dataset_dirs:
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
        requests.extend(reqs)
        information.extend(infos)

    if not requests:
        raise FileNotFoundError(
            "No runnable datasets found. Please check output_image/*/<dataset>/out_wsd directories."
        )

    if prepared_out_jsonl is not None:
        with open(prepared_out_jsonl, "w", encoding="utf8") as wf:
            for row in information:
                json.dump(row, wf, ensure_ascii=False)
                wf.write("\n")

    run_model(requests, information, BATCH_SIZE, llm, out_path)
