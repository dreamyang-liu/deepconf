from collections import defaultdict
from typing import List, Tuple, Dict

import pickle
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

INJECT_PROMPT_VERSION_1 = ". Now I feel confident about my existing analysis, I should directly and only give the short answer without being verbose and explaining details.</think>\n\\boxed"
INJECT_PROMPT_VERSION_2 = "</think>\n\\boxed"

PROMPT_VERSION_MAP = {
    1: INJECT_PROMPT_VERSION_1,
    2: INJECT_PROMPT_VERSION_2
}

PROMPT_VERSION = 1

MODEL_PATH = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


ENGINE = None
def get_vllm():
    global ENGINE
    if ENGINE is None:
        ENGINE = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=8,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
    return ENGINE

SAMPLING_PARAMETERS = None
MAX_TOKENS = 640000
def get_sampling_params():
    final_params = SamplingParams(
        n=1,
        temperature=0.6,
        top_p=0.95,
        max_tokens=20,
        logprobs=20,
        extra_args={'enable_conf': False,
        'window_size': None,
        'threshold': -1.0}  # Use individual confidence bar as threshold
    )
    return final_params

TOKENIZER = None
def get_tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return TOKENIZER

def prepare_prompt(question, dropped_thinking_trace, tokens_to_consider):
    
    # Format prompt using chat template
    messages = [
        {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
        {"role": "user", "content": question},
    ]

    tokenizer = get_tokenizer()
    
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    full_prompt += dropped_thinking_trace + PROMPT_VERSION_MAP[PROMPT_VERSION]
    # Tokenize the prompt and truncate to the specified token limit
    tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
    if len(tokens) > tokens_to_consider:
        tokens = tokens[:tokens_to_consider]
        full_prompt = tokenizer.decode(tokens)
    return full_prompt

def load_outputs(outputs_path):
    outputs = []
    for pkl in os.listdir(outputs_path):
        outputs.append(_load_output(os.path.join(outputs_path, pkl)))
    return outputs


def _load_output(path):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            return data
    except Exception:
        return None

def get_stats(results: List[Tuple[str, float]]) -> Dict[str, float]:
    stats = defaultdict(float)
    for (answer, conf) in results:
        stats[answer] += conf
    return stats

def prepare_batch_messages(question, traces, tokens_to_consider):
    batch_messages = []
    for trace in traces:
        batch_messages.append(prepare_prompt(question, trace['text'], tokens_to_consider))
    return batch_messages

def process(output):
    question = output['question']
    warmup_traces = output['warmup_traces']
    final_traces = output['final_traces']
    traces = warmup_traces + final_traces
    batch_messages = prepare_batch_messages(question, traces, 2048)
    

outputs = load_outputs("./outputs/128")

process(outputs[0])

# _load_output("/home/ubuntu/projects/deepconf/deepconf/outputs/deepconf_qid6_rid0_20251010_014300.pkl")