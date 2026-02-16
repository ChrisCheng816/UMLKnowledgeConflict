import argparse
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["NCCL_P2P_DISABLE"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["forward", "reverse", "both"],
        default="both",
        help="Run forward dataset, reverse dataset, or both.",
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
        help="Direct model path. Repeatable. Can be used with --model-name.",
    )
    parser.add_argument(
        "--out-prefix",
        default="",
        help="Optional output prefix. Files are written as [<out-prefix>_] <model>_<mode>_<relation>_<arity>.jsonl",
    )
    return parser.parse_args()


def build_runs(mode, root):
    all_runs = {
        "forward": (os.path.join(root, "data_forward"), os.path.join(root, "images", "data_forward")),
        "reverse": (os.path.join(root, "data_reverse"), os.path.join(root, "images", "data_reverse")),
    }
    if mode == "both":
        return [("forward", *all_runs["forward"]), ("reverse", *all_runs["reverse"])]
    return [(mode, *all_runs[mode])]


def main():
    args = parse_args()
    # Delay heavy import so '--help' works even if runtime deps are not ready.
    from load_open_vllm import generate_outputs, load_model
    from model_registry import resolve_models

    root = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(root, "experiment_results")
    os.makedirs(results_dir, exist_ok=True)
    runs = build_runs(args.mode, root)
    models = resolve_models(model_names=args.model_name, model_paths=args.model_path)
    if not models:
        raise ValueError("No model selected. Configure MODEL_SPECS or pass --model-name/--model-path.")
    print(f"[INFO] selected models: {[m['name'] for m in models]}")

    for model_item in models:
        model_name = model_item["name"]
        model_path = model_item["path"]
        model_results_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_results_dir, exist_ok=True)
        llm = None
        processor = None
        try:
            # Load vLLM once per model and reuse across forward/reverse.
            llm, processor = load_model(model_path)
            print(f"[INFO] model initialized: name={model_name}, path={model_path}")

            for tag, dataset_root, output_image_root in runs:
                if not os.path.isdir(dataset_root):
                    print(f"[WARN] skip {tag}: dataset_root not found -> {dataset_root}")
                    continue
                if not os.path.isdir(output_image_root):
                    print(f"[WARN] skip {tag}: output_image_root not found -> {output_image_root}")
                    continue

                print(
                    f"[INFO] run model={model_name}, split={tag}: "
                    f"dataset_root={dataset_root}, output_image_root={output_image_root}"
                )
                out_stem = f"{model_name}_{tag}" if not args.out_prefix else f"{args.out_prefix}_{model_name}_{tag}"
                generate_outputs(
                    model_path=model_path,
                    model_name=model_name,
                    out_path=os.path.join(model_results_dir, f"{out_stem}.jsonl"),
                    dataset_root=dataset_root,
                    output_image_root=output_image_root,
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


if __name__ == "__main__":
    main()
