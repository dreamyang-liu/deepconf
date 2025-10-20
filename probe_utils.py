from collections import defaultdict
from typing import List, Tuple, Dict

import pickle
import os
import json
import numpy as np

from functools import cache
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from helper import extract_answer

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
            tensor_parallel_size=1,
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


@cache
def get_template_length(question):
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
    existing_prompt_token_count = len(tokenizer.encode(full_prompt, add_special_tokens=False))
    return existing_prompt_token_count

def process_output_offline(output, ground_truth, window_size):
    """Process a single vLLM output"""
    text = output.text
    token_ids = output.token_ids
    logprobs = output.logprobs
    
    extracted_answer = extract_answer("\\boxed" + text)
    
    return {
        "stop_reason": output.finish_reason,
        "text": text,
        "token_ids": token_ids,
        "num_tokens": len(token_ids) if token_ids else 0,
        # "confs": confs,
        "extracted_answer": extracted_answer,
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

def prepare_prompt(question, dropped_thinking_trace, prob_token):
    
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
    token_usage = 0
    if len(tokens) > get_template_length(question) + prob_token:
        tokens = tokens[:get_template_length(question) + prob_token]
        full_prompt = tokenizer.decode(tokens)
        token_usage = len(tokens)
    return full_prompt + PROMPT_VERSION_MAP[PROMPT_VERSION], token_usage

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

def prepare_batch_messages(question, traces, prob_token):
    batch_messages = []
    token_usages = []
    for trace in traces:
        message, token_usage = prepare_prompt(question, trace['text'], prob_token)
        batch_messages.append(message)
        token_usages.append(token_usage)
    return batch_messages, token_usages

def probe_answers(messages):
    engine = get_vllm()
    sampling_params = get_sampling_params()
    outputs = engine.generate(messages, sampling_params)
    return post_process_vllm_generation_outputs(outputs)

def post_process_vllm_generation_outputs(outputs):
    answers = []
    processed_results = process_batch_results_offline(outputs, "X", 2048)
    for processed_result in processed_results:
        answers.append(processed_result['traces'][0]['extracted_answer'])
    return answers

def conf_metric_min_window_conf(confs, window_size):
    if len(confs) <= window_size:
        return np.mean(confs) if confs else 0
    # Use numpy's rolling window mean for better performance
    return np.min(np.convolve(confs, np.ones(window_size)/window_size, mode='valid'))

def conf_metric_max_window_conf(confs, window_size):
    if len(confs) <= window_size:
        return np.mean(confs) if confs else 0
    # Use numpy's rolling window mean for better performance
    return np.max(np.convolve(confs, np.ones(window_size)/window_size, mode='valid'))

def conf_metric_mean_window_conf(confs, window_size):
    if len(confs) <= window_size:
        return np.mean(confs) if confs else 0
    # Use numpy's rolling window mean for better performance
    return np.mean(np.convolve(confs, np.ones(window_size)/window_size, mode='valid'))

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

def calculate_traces_accuracy(answer_dict, ground_truth):
    if not answer_dict:
        return 0.0

    total = sum(answer_dict.values())
    if total == 0:
        return 0.0

    # Fallback to string comparison
    for answer, count in answer_dict.items():
        if str(answer) == str(ground_truth):
            return count / total
    return 0.0

def process(output, qid, prob_token, window_size):
    if os.path.exists(f"./prob_analysis/analysis_results_{qid}_probtoken_{prob_token}_windowsize_{window_size}.json"):
        print(f"Alreay processed, skipping for {qid} {prob_token} {window_size} ...")
        return
    if prob_token > max(trace['num_tokens'] for trace in output['all_traces']):
        print(f"Reached max token, skipping for {qid} {prob_token} {window_size} ...")
        return 

    question = output['question']
    traces = output['all_traces']
    batch_messages, _ = prepare_batch_messages(question, traces, prob_token)
    batch_confs = [trace['confs'][:prob_token] for trace in traces]
    assert all(len(conf) <= prob_token for conf in batch_confs)
    answers = probe_answers(batch_messages)

    # Calculate confidence metrics for each trace
    conf_stats = []

    for i, confs in enumerate(batch_confs):
        if len(confs) >= window_size:
            conf_stats.append({
                "min_window_conf": conf_metric_min_window_conf(confs, window_size),
                "max_window_conf": conf_metric_max_window_conf(confs, window_size),
                "mean_window_conf": conf_metric_mean_window_conf(confs, window_size),
                "last_window_conf": conf_metric_last_window_conf(confs, window_size)
            })
        else:
            # For traces with fewer tokens than window_size
            conf_stats.append({
                "min_window_conf": np.mean(confs) if confs else 0,
                "max_window_conf": np.mean(confs) if confs else 0,
                "mean_window_conf": np.mean(confs) if confs else 0,
                "last_window_conf": np.mean(confs) if confs else 0
            })
    
    answer_dict = defaultdict(lambda: {"count": 0, "min_window_conf": 0, "max_window_conf": 0, "mean_window_conf": 0, "last_window_conf": 0})
    for i, answer in enumerate(answers):
        answer_key = answer
        answer_dict[answer_key]["count"] += 1
        answer_dict[answer_key]["min_window_conf"] += conf_stats[i]["min_window_conf"]
        answer_dict[answer_key]["max_window_conf"] += conf_stats[i]["max_window_conf"]
        answer_dict[answer_key]["mean_window_conf"] += conf_stats[i]["mean_window_conf"]
        answer_dict[answer_key]["last_window_conf"] += conf_stats[i]["last_window_conf"]
    answer_dict = dict(sorted(answer_dict.items(), key=lambda x: x[1]["count"], reverse=True))

    # Calculate confidence metrics based on answer distribution
    count_dist = {k: v["count"] for k, v in answer_dict.items()}
    min_window_conf_dist = {k: v["min_window_conf"] for k, v in answer_dict.items()}
    max_window_conf_dist = {k: v["max_window_conf"] for k, v in answer_dict.items()}
    mean_window_conf_dist = {k: v["mean_window_conf"] for k, v in answer_dict.items()}
    last_window_conf_dist = {k: v["last_window_conf"] for k, v in answer_dict.items()}

    # Calculate entropy and confidence metrics
    entropy_count = calculate_macro_entropy(count_dist)
    entropy_min_window = calculate_macro_entropy(min_window_conf_dist)
    entropy_max_window = calculate_macro_entropy(max_window_conf_dist)
    entropy_mean_window = calculate_macro_entropy(mean_window_conf_dist)
    entropy_last_window = calculate_macro_entropy(last_window_conf_dist)

    # Compute confidence from metrics
    conf_count = compute_confidence_from_metric(count_dist)
    conf_min_window = compute_confidence_from_metric(min_window_conf_dist)
    conf_max_window = compute_confidence_from_metric(max_window_conf_dist)
    conf_mean_window = compute_confidence_from_metric(mean_window_conf_dist)
    conf_last_window = compute_confidence_from_metric(last_window_conf_dist)

    # Save results to JSON file
    results = {
        "answers": answer_dict,
        "ground_truth": output["ground_truth"],
        "deepconf_token_stats": output['token_stats'],
        "entropy_metrics": {
            "entropy_count": entropy_count,
            "entropy_min_window": entropy_min_window,
            "entropy_max_window": entropy_max_window,
            "entropy_mean_window": entropy_mean_window,
            "entropy_last_window": entropy_last_window
        },
        "confidence_metrics": {
            "conf_count": conf_count,
            "conf_min_window": conf_min_window,
            "conf_max_window": conf_max_window,
            "conf_mean_window": conf_mean_window,
            "conf_last_window": conf_last_window
        },
        "window_size": window_size,
        "prob_token": prob_token,
        "count_accuracy": calculate_traces_accuracy(count_dist, output["ground_truth"])
    }

    with open(f"./prob_analysis/analysis_results_{qid}_probtoken_{prob_token}_windowsize_{window_size}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to analysis_results_{qid}_probtoken_{prob_token}_windowsize_{window_size}.json")


if __name__ == "__main__":
    outputs = load_outputs("./outputs")
    for window_size in [1024, 2048]:
        for output in outputs:
            for prob_token in [2048 * i for i in range(1, 17)]:
                if window_size > prob_token:
                    continue
                try:
                    process(output[0], output[1], prob_token, window_size)
                except Exception as e:
                    print(e)