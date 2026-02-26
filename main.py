import argparse
import os
import re

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["NCCL_P2P_DISABLE"] = "1"

CLOSED_DEFAULT_MODELS = [
    {"name": "gemini_2_5_pro", "path": "gemini-2.5-pro"},
    {"name": "o3", "path": "o3"},
    {"name": "o4_mini", "path": "o4-mini"},
    {"name": "claude_3_7_sonnet", "path": "claude-3-7-sonnet-20250219"},
    {"name": "gpt_4_1", "path": "gpt-4.1"},
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["open_vllm", "closed_llm"],
        default="open_vllm",
        help="Inference backend. Default is open_vllm (open-source vLLM).",
    )
    parser.add_argument(
        "--mode",
        choices=["forward", "reverse", "mixed", "both"],
        default="both",
        help="Run forward dataset, reverse dataset, mixed dataset, or all available splits.",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help="Registered model name in model_registry.py. Repeatable. For open_vllm this is required.",
    )
    parser.add_argument(
        "--out-prefix",
        default="",
        help="Optional output prefix. Files are written as [<out-prefix>_] <model>_<mode>_<relation>_<arity>.jsonl",
    )
    parser.add_argument(
        "--arity",
        choices=["2", "3", "both"],
        default="both",
        help="Run only 2-class, only 3-class, or both.",
    )
    parser.add_argument(
        "--relation",
        action="append",
        nargs="+",
        choices=["inheritance", "aggregation", "composition", "dependency"],
        default=None,
        help=(
            "Relation types to run. You can pass multiple at once, e.g. "
            "--relation inheritance dependency. Repeatable. Default is all relations."
        ),
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="vLLM tensor parallel size. Default: auto (from CUDA_VISIBLE_DEVICES, else 1).",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=None,
        help="vLLM gpu_memory_utilization in (0,1]. Default uses loader setting.",
    )
    return parser.parse_args()


def build_runs(mode, root):
    all_runs = {
        "forward": (os.path.join(root, "data_forward"), os.path.join(root, "images", "data_forward")),
        "reverse": (os.path.join(root, "data_reverse"), os.path.join(root, "images", "data_reverse")),
        "mixed": (os.path.join(root, "data_mixed"), os.path.join(root, "images", "data_mixed")),
    }
    if mode == "both":
        return [
            ("forward", *all_runs["forward"]),
            ("reverse", *all_runs["reverse"]),
            ("mixed", *all_runs["mixed"]),
        ]
    return [(mode, *all_runs[mode])]


def _slugify(text):
    value = (text or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "model"


def _dedup_models(models):
    out = []
    seen = set()
    for m in models:
        key = (m["name"], m["path"])
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def _resolve_closed_models(model_names):
    # Closed backend accepts API model ids directly.
    # - If no models provided, use curated defaults for one-click runs.
    # - model_name/model_path are both treated as model ids.
    if not model_names:
        return list(CLOSED_DEFAULT_MODELS)

    resolved = []
    for name in model_names or []:
        resolved.append({"name": _slugify(name), "path": name})
    return _dedup_models(resolved)


def main():
    args = parse_args()

    backend = args.backend
    root = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(root, "experiment_results")
    os.makedirs(results_dir, exist_ok=True)
    runs = build_runs(args.mode, root)
    arity_filter = {"2", "3"} if args.arity == "both" else {args.arity}
    if args.relation:
        relation_filter = {
            relation
            for relation_group in args.relation
            for relation in relation_group
        }
    else:
        relation_filter = {
            "inheritance",
            "aggregation",
            "composition",
            "dependency",
        }

    if backend == "closed_llm":
        models = _resolve_closed_models(args.model_name)
    else:
        # Delay heavy import so '--help' works even if runtime deps are not ready.
        from model_registry import resolve_models

        if not args.model_name:
            raise ValueError("open_vllm requires --model-name, and it must match model_registry.py entries.")
        models = resolve_models(model_names=args.model_name, model_paths=None)
    if not models:
        raise ValueError("No model selected. Configure MODEL_SPECS or pass --model-name/--model-path.")
    print(f"[INFO] selected models: {[m['name'] for m in models]}")

    for model_item in models:
        model_name = model_item["name"]
        model_path = model_item["path"]
        model_results_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_results_dir, exist_ok=True)
        valid_runs = []
        for tag, dataset_root, output_image_root in runs:
            if not os.path.isdir(dataset_root):
                print(f"[WARN] skip {tag}: dataset_root not found -> {dataset_root}")
                continue
            if not os.path.isdir(output_image_root):
                print(f"[WARN] skip {tag}: output_image_root not found -> {output_image_root}")
                continue
            valid_runs.append((tag, dataset_root, output_image_root))

        if not valid_runs:
            print(f"[WARN] no runnable splits for model={model_name}, backend={backend}; skip model")
            continue

        if backend == "open_vllm":
            from load_open_vllm import generate_outputs, load_model

            llm = None
            processor = None
            try:
                # Load vLLM once per model and reuse across forward/reverse.
                llm, processor = load_model(
                    model_path,
                    tensor_parallel_size=args.tp_size,
                    gpu_memory_utilization=args.gpu_mem_util,
                )
                print(f"[INFO] model initialized: name={model_name}, path={model_path}, backend={backend}")

                for tag, dataset_root, output_image_root in valid_runs:
                    print(
                        f"[INFO] run model={model_name}, split={tag}, backend={backend}: "
                        f"dataset_root={dataset_root}, output_image_root={output_image_root}"
                    )
                    out_stem = f"{model_name}_{tag}" if not args.out_prefix else f"{args.out_prefix}_{model_name}_{tag}"
                    generate_outputs(
                        model_path=model_path,
                        model_name=model_name,
                        out_path=os.path.join(model_results_dir, f"{out_stem}.jsonl"),
                        dataset_root=dataset_root,
                        output_image_root=output_image_root,
                        include_arities=arity_filter,
                        include_relations=relation_filter,
                        llm=llm,
                        processor=processor,
                    )
            finally:
                if llm is not None:
                    del llm
                if processor is not None:
                    del processor
                try:
                    import gc
                    gc.collect()
                except Exception:
                    pass
        else:
            from load_closed_llm import generate_outputs

            print(f"[INFO] model initialized: name={model_name}, path={model_path}, backend={backend}")
            for tag, dataset_root, output_image_root in valid_runs:
                print(
                    f"[INFO] run model={model_name}, split={tag}, backend={backend}: "
                    f"dataset_root={dataset_root}, output_image_root={output_image_root}"
                )
                out_stem = f"{model_name}_{tag}" if not args.out_prefix else f"{args.out_prefix}_{model_name}_{tag}"
                generate_outputs(
                    model=model_path,
                    model_name=model_name,
                    out_path=os.path.join(model_results_dir, f"{out_stem}.jsonl"),
                    dataset_root=dataset_root,
                    output_image_root=output_image_root,
                    include_arities=arity_filter,
                    include_relations=relation_filter,
                )


if __name__ == "__main__":
    main()
