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
        "--model-path",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model path passed to generate_outputs.",
    )
    parser.add_argument(
        "--out-prefix",
        default="result",
        help="Output prefix. Files are written as <out-prefix>_<mode>_<relation>_<arity>.jsonl",
    )
    return parser.parse_args()


def build_runs(mode, root):
    all_runs = {
        "forward": (os.path.join(root, "data_forward"), os.path.join(root, "image_forward")),
        "reverse": (os.path.join(root, "data_reverse"), os.path.join(root, "image_reverse")),
    }
    if mode == "both":
        return [("forward", *all_runs["forward"]), ("reverse", *all_runs["reverse"])]
    return [(mode, *all_runs[mode])]


def main():
    args = parse_args()
    # Delay heavy import so '--help' works even if runtime deps are not ready.
    from load_open_vllm import generate_outputs, load_model

    root = os.path.dirname(os.path.abspath(__file__))
    runs = build_runs(args.mode, root)
    llm = None
    processor = None

    try:
        # Load vLLM once and reuse across forward/reverse to avoid repeated
        # worker/ray initialization and duplicated GPU reservation.
        llm, processor = load_model(args.model_path)
        print("[INFO] model initialized once and will be reused")

        for tag, dataset_root, output_image_root in runs:
            if not os.path.isdir(dataset_root):
                print(f"[WARN] skip {tag}: dataset_root not found -> {dataset_root}")
                continue
            if not os.path.isdir(output_image_root):
                print(f"[WARN] skip {tag}: output_image_root not found -> {output_image_root}")
                continue

            print(f"[INFO] run {tag}: dataset_root={dataset_root}, output_image_root={output_image_root}")
            generate_outputs(
                model_path=args.model_path,
                out_path=f"{args.out_prefix}_{tag}.jsonl",
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
