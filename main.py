from load_open_vllm import generate_outputs
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

if __name__ == "__main__":

    generate_outputs("Qwen/Qwen3-VL-8B-Instruct", "result.jsonl")