from load_open_vllm import generate_outputs
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

if __name__ == "__main__":
    model_path = "Qwen/Qwen3-VL-8B-Instruct"
    root = os.path.dirname(os.path.abspath(__file__))

    runs = [
        ("forward", os.path.join(root, "data_forward"), os.path.join(root, "image_forward")),
        ("reverse", os.path.join(root, "data_reverse"), os.path.join(root, "image_reverse")),
    ]

    for tag, dataset_root, output_image_root in runs:
        if not os.path.isdir(dataset_root):
            print(f"[WARN] skip {tag}: dataset_root not found -> {dataset_root}")
            continue
        if not os.path.isdir(output_image_root):
            print(f"[WARN] skip {tag}: output_image_root not found -> {output_image_root}")
            continue

        print(f"[INFO] run {tag}: dataset_root={dataset_root}, output_image_root={output_image_root}")
        generate_outputs(
            model_path=model_path,
            out_path=f"result_{tag}.jsonl",
            dataset_root=dataset_root,
            output_image_root=output_image_root,
        )
