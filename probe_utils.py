from collections import defaultdict
from typing import List, Tuple, Dict

import pickle
import os
import json
import numpy as np

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from math_verify import parse

INJECT_PROMPT_VERSION_1 = ". Now I feel confident about my existing analysis, I should directly and only give the short answer without being verbose or explaining details.</think>\n\\boxed"
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
            tensor_parallel_size=4,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
    return ENGINE

SAMPLING_PARAMETERS = None
MAX_TOKENS = 100000
def get_sampling_params():
    final_params = SamplingParams(
        n=1,
        temperature=0.6,
        top_p=0.95,
        max_tokens=MAX_TOKENS,
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


def process_output_offline(output, ground_truth, window_size):
    """Process a single vLLM output"""
    text = output.text
    token_ids = output.token_ids
    logprobs = output.logprobs
    
    # # Calculate confidence
    # confs = compute_confidence(logprobs) if logprobs else []
    
    # extracted_answer = extract_answer(text)
    
    # is_correct = False
    # if extracted_answer and ground_truth:
    #     try:
    #         is_correct = equal_func(extracted_answer, ground_truth)
    #     except:
    #         is_correct = str(extracted_answer) == str(ground_truth)
    
    return {
        "stop_reason": output.finish_reason,
        "text": text,
        "token_ids": token_ids,
        "num_tokens": len(token_ids) if token_ids else 0,
        # "confs": confs,
        # "extracted_answer": extracted_answer,
        # "is_correct": is_correct,
    }


def process_batch_results_offline(batch_outputs, ground_truth, window_size):
    processed_results = []
    for output in batch_outputs:
        """Process batch results from vLLM for a single question"""
        question_outputs = output.outputs
        
        # Process all traces for this question
        traces = []
        total_tokens = 0
        
        for output in question_outputs:
            trace_data = process_output_offline(output, ground_truth, window_size)
            traces.append(trace_data)
            total_tokens += trace_data["num_tokens"]
        
        processed_results.append({
            'traces': traces,
            'ground_truth': ground_truth,
            'total_tokens': total_tokens,
            'num_traces': len(traces)
        })
    return processed_results

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
    full_prompt += dropped_thinking_trace
    # Tokenize the prompt and truncate to the specified token limit
    tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
    if len(tokens) > tokens_to_consider:
        tokens = tokens[:tokens_to_consider]
        full_prompt = tokenizer.decode(tokens)
    return full_prompt + PROMPT_VERSION_MAP[PROMPT_VERSION]

def load_outputs(outputs_path):
    outputs = []
    for pkl in os.listdir(outputs_path):
        outputs.append((_load_output(os.path.join(outputs_path, pkl)), pkl.split("_")[2]))
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

def probe_answers(messages):
    engine = get_vllm()
    sampling_params = get_sampling_params()
    outputs = engine.generate(messages, sampling_params)
    return post_process_vllm_generation_outputs(outputs)

def post_process_vllm_generation_outputs(outputs):
    answers = []
    processed_results = process_batch_results_offline(outputs, "X", 2048)
    for processed_result in processed_results:
        answers.append(parse(f"${processed_result['traces'][0]['text'][1:-1]}$"))
    return answers

def conf_metric_min_window_conf(confs, window_size):
    window = []
    for i in range(len(confs) - window_size):
        window.append(np.mean(confs[i:i+window_size]))
    return np.min(window)

def conf_metric_max_window_conf(confs, window_size):
    window = []
    for i in range(len(confs) - window_size):
        window.append(np.mean(confs[i:i+window_size]))
    return np.max(window)

def conf_metric_mean_window_conf(confs, window_size):
    window = []
    for i in range(len(confs) - window_size):
        window.append(np.mean(confs[i:i+window_size]))
    return np.mean(window)

def conf_metric_last_window_conf(confs, window_size):
    return np.mean(confs[-window_size:])


def calculate_macro_entropy(answer_dict):
    """
    Calculate the entropy of an answer distribution.
    Higher entropy means higher uncertainty.
    
    Args:
        answer_dict: dict mapping answers to their occurrence counts
    
    Returns:
        float: entropy value (0 to log(n))
    """
    if not answer_dict:
        return 0
        
    total = sum(answer_dict.values())
    if total == 0:
        return 0
        
    probabilities = [count / total for count in answer_dict.values()]
    entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
    
    return entropy

def compute_confidence_from_metric(answer_dict):
    """
    Compute confidence based on entropy of the answer distribution.
    Lower entropy means higher confidence.
    
    Args:
        answer_dict: dict mapping answers to their occurrence counts
    
    Returns:
        float: confidence score between 0 and 1
    """
    if not answer_dict:
        return 0.0
        
    total = sum(answer_dict.values())
    if total == 0:
        return 0.0
        
    probabilities = [count / total for count in answer_dict.values()]
    
    entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
    
    if len(answer_dict) <= 1:
        return 1.0
        
    max_entropy = np.log(len(answer_dict))
    
    confidence = 1 - (entropy / max_entropy)
    
    return round(confidence, 3)

def process(output, qid, tokens_to_consider):
    if os.path.exists(f"./prob_analysis/analysis_results_{qid}_probtoken_{tokens_to_consider}.json"):
        print(f"Alreay processed, skipping for {qid} {tokens_to_consider} ...")
        return

    question = output['question']
    traces = output['all_traces']
    batch_messages = prepare_batch_messages(question, traces, tokens_to_consider)
    batch_confs = [trace['confs'] for trace in traces]

    # Calculate confidence metrics for each trace
    conf_stats = {}
    window_size = 5  # Default window size for confidence metrics

    for i, confs in enumerate(batch_confs):
        if len(confs) >= window_size:
            conf_stats[f"trace_{i}"] = {
                "min_window_conf": conf_metric_min_window_conf(confs, window_size),
                "max_window_conf": conf_metric_max_window_conf(confs, window_size),
                "mean_window_conf": conf_metric_mean_window_conf(confs, window_size),
                "last_window_conf": conf_metric_last_window_conf(confs, window_size)
            }
        else:
            # For traces with fewer tokens than window_size
            conf_stats[f"trace_{i}"] = {
                "min_window_conf": np.mean(confs) if confs else 0,
                "max_window_conf": np.mean(confs) if confs else 0,
                "mean_window_conf": np.mean(confs) if confs else 0,
                "last_window_conf": np.mean(confs) if confs else 0
            }

    answers = probe_answers(batch_messages)
    answer_dict = defaultdict(int)
    for answer in answers:
        answer_dict[str(answer[0])] += 1
    answer_dict = dict(sorted(answer_dict.items(), key=lambda x: x[1], reverse=True))
    # Save results to JSON file
    results = {
        "answers": answer_dict,
        "ground_truth": output["ground_truth"],
        "deepconf_token_stats": output['token_stats']
    }

    with open(f"./prob_analysis/analysis_results_{qid}_probtoken_{tokens_to_consider}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to analysis_results.json")


if __name__ == "__main__":
    outputs = load_outputs("./outputs")
    for prob_token in [2048 * i for i in range(1, 65)]:
        for output in outputs:
            try:
                process(output[0], output[1], prob_token)
            except Exception as e:
                print(e)

# _load_output("/home/ubuntu/projects/deepconf/deepconf/outputs/deepconf_qid6_rid0_20251010_014300.pkl")