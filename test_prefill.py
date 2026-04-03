"""Quick test: verify prompt_logprobs works across multiple generate() calls."""
import os
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json, pickle, gc
import numpy as np

# Load data
with open("aime_2024.jsonl") as f:
    data = [json.loads(l) for l in f]

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)
from helper import prepare_prompt
prompt_text, gt = prepare_prompt(data[0], tokenizer)
prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
prompt_len = len(prompt_ids)
print(f"Prompt len: {prompt_len} tokens")

with open("outputs-bedrock/aime24/deepconf_simple_qid0_ridbedrock_20260402_070120.pkl", "rb") as f:
    pdata = pickle.load(f)

# Filter traces that fit
valid = []
for t in pdata["all_traces"][:100]:
    ft = prompt_text + t["text"]
    ids = tokenizer.encode(ft, add_special_tokens=False)
    if len(ids) <= 40958:
        valid.append(ft)
    if len(valid) >= 50:
        break
print(f"Valid traces: {len(valid)}")

llm = LLM(model="Qwen/Qwen3-32B", tensor_parallel_size=2, trust_remote_code=True,
          max_model_len=40960, gpu_memory_utilization=0.90)

sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=5)

# Test multiple calls of increasing size
for call_num in range(5):
    batch = valid[call_num*10:(call_num+1)*10]
    if not batch:
        break
    print(f"\nCall {call_num+1}: {len(batch)} traces...")
    outputs = llm.generate(batch, sp)
    print(f"  Got {len(outputs)} outputs")
    # Check first output has logprobs
    o = outputs[0]
    if o.prompt_logprobs:
        gen = o.prompt_logprobs[prompt_len:]
        non_none = sum(1 for x in gen if x is not None)
        print(f"  prompt_logprobs: total={len(o.prompt_logprobs)}, gen_portion={len(gen)}, non_none={non_none}")
    del outputs
    gc.collect()

print("\nAll calls succeeded!")
