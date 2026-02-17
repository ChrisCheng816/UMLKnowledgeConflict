import argparse
import gc
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from model_registry import resolve_models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run reverse-set UML VLM tests with matched images."
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help="Registered model name in model_registry.py. Repeatable.",
    )
    parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        help="Direct model path. Repeatable.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for vLLM generation.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="How many full runs to execute for each sample.",
    )
    parser.add_argument(
        "--dataset-root",
        default="data_reverse",
        help="Reverse dataset root.",
    )
    parser.add_argument(
        "--image-root",
        default="images/data_reverse",
        help="Image root that corresponds to reverse dataset.",
    )
    parser.add_argument(
        "--results-root",
        default="experiment_results",
        help="Directory to save test outputs.",
    )
    parser.add_argument(
        "--limit-datasets",
        type=int,
        default=0,
        help="Only run first N discovered datasets (0 means all).",
    )
    return parser.parse_args()


def discover_reverse_datasets(dataset_root):
    dataset_dirs = []
    for top in sorted(os.listdir(dataset_root)):
        top_path = os.path.join(dataset_root, top)
        if not os.path.isdir(top_path):
            continue
        top_lower = top.lower()
        if not (top_lower.startswith("2class_") or top_lower.startswith("3class_")):
            continue

        top_instances = os.path.join(top_path, "instances.jsonl")
        if os.path.isfile(top_instances):
            dataset_dirs.append(os.path.relpath(top_path, dataset_root))
            continue

        for cur_root, _, files in os.walk(top_path):
            if "instances.jsonl" in files:
                dataset_dirs.append(os.path.relpath(cur_root, dataset_root))

    return sorted(dataset_dirs)


def main():
    args = parse_args()
    import load_open_vllm as open_vllm

    root = os.path.dirname(os.path.abspath(__file__))

    dataset_root = os.path.join(root, args.dataset_root)
    image_root = os.path.join(root, args.image_root)
    results_root = os.path.join(root, args.results_root)

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"image_root not found: {image_root}")

    models = resolve_models(model_names=args.model_name, model_paths=args.model_path)
    if not models:
        raise ValueError("No model selected.")

    dataset_dirs = discover_reverse_datasets(dataset_root)
    if not dataset_dirs:
        raise FileNotFoundError(f"No datasets found under: {dataset_root}")
    if args.limit_datasets > 0:
        dataset_dirs = dataset_dirs[: args.limit_datasets]

    open_vllm.BATCH_SIZE = max(1, int(args.batch_size))
    print(f"[INFO] batch_size={open_vllm.BATCH_SIZE}, num_runs={args.num_runs}")
    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] image_root={image_root}")
    print(f"[INFO] total datasets={len(dataset_dirs)}")

    os.makedirs(results_root, exist_ok=True)

    for model_item in models:
        model_name = model_item["name"]
        model_path = model_item["path"]
        model_results_dir = os.path.join(results_root, model_name)
        os.makedirs(model_results_dir, exist_ok=True)

        llm = None
        processor = None
        try:
            llm, processor = open_vllm.load_model(model_path)
            print(f"[INFO] model initialized: {model_name} -> {model_path}")

            # Loop over every reverse dataset and run end-to-end test.
            for ds in dataset_dirs:
                out_base = os.path.join(model_results_dir, f"{model_name}_reverse_{ds.replace(os.sep, '_')}.jsonl")
                print(f"[INFO] running dataset={ds} -> {out_base}")
                open_vllm.generate_outputs(
                    model_path=model_path,
                    model_name=model_name,
                    out_path=out_base,
                    dataset_dir=ds,
                    dataset_root=dataset_root,
                    output_image_root=image_root,
                    llm=llm,
                    processor=processor,
                    num_runs=args.num_runs,
                )
        finally:
            if llm is not None:
                del llm
            if processor is not None:
                del processor
            gc.collect()


if __name__ == "__main__":
    main()
