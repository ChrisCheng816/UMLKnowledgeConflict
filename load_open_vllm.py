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

def _build_inheritance_prompt(nodes, relation="inheritance"):
    # Reserved triplet interface for future extension, e.g. nodes=[node1, node2, node3].
    if len(nodes) < 2:
        raise ValueError(f"nodes must contain at least 2 elements, got: {nodes}")

    node1 = nodes[0]
    node2 = nodes[1]
    node3 = nodes[2] if len(nodes) >= 3 else None

    if relation == "inheritance":
        question = f"In this UML diagram, does class {node2} inherit from class {node1}? Please answer True or False."
    else:
        raise ValueError(f"Unsupported relation type: {relation}")

    return question, {"node1": node1, "node2": node2, "node3": node3}

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

def load_prompt(
    processor,
    dataset_dir,
    output_image_root="output_image",
    prepared_out_jsonl=None,
    relation="inheritance"
):
    requests = []
    information = []

    instances_path = os.path.join(dataset_dir, "instances.jsonl")
    image_dir = os.path.join(output_image_root, dataset_dir, "out_wsd")
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
            user_text, triplet_slots = _build_inheritance_prompt(nodes, relation=relation)
            image_path = image_map.get(sample_id)
            if image_path is None:
                raise FileNotFoundError(
                    f"No image found for id={sample_id}. Expected a file like '{sample_id}_*.png' in {image_dir}."
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
        max_tokens=512,
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
                "template_id": info.get("template_id"),
                "nodes": info.get("nodes"),
                "triplet_slots": info.get("triplet_slots"),
                "image_path": info.get("image_path"),
                "prompt": info.get("prompt"),
                "output": pred
            }
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

def generate_outputs(model_path, dataset_dir, out_path, prepared_out_jsonl=None):
    llm, processor = load_model(model_path)
    requests, information = load_prompt(
        processor=processor,
        dataset_dir=dataset_dir,
        prepared_out_jsonl=prepared_out_jsonl
    )
    run_model(requests, information, BATCH_SIZE, llm, out_path)
