"""
Offline trace generation using SGLang Engine (offline/batch inference).

Generates reasoning traces for AIME 2025 and HMMT Feb 2025 datasets using
Qwen3-32B. No logprobs are collected — only raw traces are saved. These
traces will later be used to generate logprobs for the DeepConf offline
confidence pipeline.

Config:
    Model:       Qwen/Qwen3-32B
    Temperature: 0.6
    Top-p:       0.95
    Top-k:       20
    Max tokens:  32768
"""

import json
import os
import pickle
import time
import argparse
from datetime import datetime

import sglang as sgl
from transformers import AutoTokenizer

from helper import extract_answer, equal_func

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = "Qwen/Qwen3-32B"
MAX_TOKENS = 32768
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20

DATASETS = {
    "aime_2025": "aime_2025.jsonl",
    "hmmt_feb_2025": "hmmt_feb_2025.jsonl",
}

DEFAULT_BUDGET = 64  # traces per question
OUTPUT_DIR = "outputs-traces"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]


def prepare_prompt(question_data, tokenizer):
    """Format a question into a chat-template prompt."""
    messages = [
        {
            "role": "system",
            "content": (
                "该助手为DeepSeek-R1，由深度求索公司创造。\n"
                "今天是2025年5月28日，星期一。\n"
            ),
        },
        {"role": "user", "content": question_data["question"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ground_truth = str(question_data.get("answer", "")).strip()
    return prompt, ground_truth


def process_trace(text, ground_truth):
    """Lightweight processing: extract answer and check correctness."""
    extracted = extract_answer(text)
    is_correct = False
    if extracted and ground_truth:
        try:
            is_correct = equal_func(extracted, ground_truth)
        except Exception:
            is_correct = str(extracted) == str(ground_truth)
    return {
        "text": text,
        "extracted_answer": extracted,
        "is_correct": is_correct,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def generate_dataset(engine, tokenizer, dataset_name, dataset_file, budget, rid, output_dir):
    """Generate traces for every question in a single dataset."""
    data = load_dataset(dataset_file)
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}  ({len(data)} questions, {budget} traces each)")
    print(f"{'='*60}")

    model_name = MODEL_PATH.split("/")[-1]
    base_dir = os.path.join(output_dir, model_name, dataset_name)

    for qid, question_data in enumerate(data):
        trace_dir = os.path.join(base_dir, f"traces/qid{qid}_rid{rid}")
        os.makedirs(trace_dir, exist_ok=True)

        # ── Resumption: check existing traces ────────────────────────────
        existing = set()
        for fname in os.listdir(trace_dir):
            if fname.startswith("trace_") and fname.endswith(".pkl"):
                idx = int(fname.replace("trace_", "").replace(".pkl", ""))
                existing.add(idx)
        missing = sorted(set(range(budget)) - existing)

        if not missing:
            print(f"  [qid={qid}] Already have {budget}/{budget} traces — skipping")
            continue

        if existing:
            print(f"  [qid={qid}] Resuming: {len(existing)} exist, {len(missing)} to generate")
        else:
            print(f"  [qid={qid}] Generating {budget} traces...")

        prompt, ground_truth = prepare_prompt(question_data, tokenizer)

        # ── Batch generation via SGLang Engine ────────────────────────────
        # Duplicate the prompt for each missing trace so the engine can
        # batch them efficiently with its internal scheduler.
        prompts = [prompt] * len(missing)
        sampling_params = {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_new_tokens": MAX_TOKENS,
        }

        t0 = time.time()
        outputs = engine.generate(
            prompt=prompts,
            sampling_params=sampling_params,
        )
        gen_time = time.time() - t0

        # ── Save each trace individually (fault-tolerant) ────────────────
        correct = 0
        total_tokens = 0
        for i, trace_idx in enumerate(missing):
            text = outputs[i]["text"]
            num_tokens = outputs[i]["meta_info"]["completion_tokens"]
            total_tokens += num_tokens

            trace_data = process_trace(text, ground_truth)
            trace_data["num_tokens"] = num_tokens
            trace_data["trace_id"] = trace_idx

            trace_path = os.path.join(trace_dir, f"trace_{trace_idx:04d}.pkl")
            with open(trace_path, "wb") as f:
                pickle.dump(trace_data, f)

            if trace_data["is_correct"]:
                correct += 1

        # ── Save question-level metadata ─────────────────────────────────
        meta = {
            "question_id": qid,
            "run_id": rid,
            "dataset": dataset_name,
            "question": question_data["question"],
            "ground_truth": ground_truth,
            "num_traces_generated": len(missing),
            "num_traces_total": budget,
            "correct_traces": correct,
            "accuracy": correct / len(missing) if missing else 0,
            "total_tokens": total_tokens,
            "generation_time_s": gen_time,
            "config": {
                "model": MODEL_PATH,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "max_tokens": MAX_TOKENS,
            },
            "timestamp": datetime.now().isoformat(),
        }
        meta_path = os.path.join(base_dir, f"qid{qid}_rid{rid}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        acc = correct / len(missing) * 100 if missing else 0
        tps = total_tokens / gen_time if gen_time > 0 else 0
        print(
            f"  [qid={qid}] Done in {gen_time:.1f}s — "
            f"acc={correct}/{len(missing)} ({acc:.1f}%), "
            f"tokens={total_tokens}, {tps:.0f} tok/s"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate reasoning traces with SGLang offline inference"
    )
    parser.add_argument(
        "--datasets", nargs="*", default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Which datasets to process (default: all)",
    )
    parser.add_argument(
        "--budget", type=int, default=DEFAULT_BUDGET,
        help=f"Number of traces per question (default: {DEFAULT_BUDGET})",
    )
    parser.add_argument(
        "--rid", type=str, default="run0",
        help="Run ID for file naming (default: run0)",
    )
    parser.add_argument(
        "--tp", type=int, default=2,
        help="Tensor parallelism degree (default: 2)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    output_dir = args.output_dir

    print(f"Model:       {MODEL_PATH}")
    print(f"Datasets:    {args.datasets}")
    print(f"Budget:      {args.budget} traces/question")
    print(f"Run ID:      {args.rid}")
    print(f"TP:          {args.tp}")
    print(f"Output:      {output_dir}")
    print()

    # ── Initialize engine & tokenizer ────────────────────────────────────
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Initializing SGLang Engine (this may take a while)...")
    t0 = time.time()
    engine = sgl.Engine(
        model_path=MODEL_PATH,
        tp_size=args.tp,
    )
    print(f"Engine ready in {time.time() - t0:.1f}s\n")

    # ── Generate for each dataset ────────────────────────────────────────
    overall_start = time.time()
    for ds_name in args.datasets:
        ds_file = DATASETS[ds_name]
        if not os.path.exists(ds_file):
            print(f"WARNING: {ds_file} not found, skipping {ds_name}")
            continue
        generate_dataset(engine, tokenizer, ds_name, ds_file, args.budget, args.rid, output_dir)

    total_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"All done in {total_time:.1f}s")
    print(f"{'='*60}")

    engine.shutdown()


if __name__ == "__main__":
    main()
