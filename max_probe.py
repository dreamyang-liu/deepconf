import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob

from functools import cache
from tqdm import tqdm, trange
from collections import Counter
from probe_utils import get_template_length, probe_answers, get_tokenizer
import math

@cache
def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data


def prepare_prompt(question, trace, prob_token, candidate_answers):
    
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
    full_prompt += trace['text']
    # Tokenize the prompt and truncate to the specified token limit
    tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
    if len(tokens) > get_template_length(question) + prob_token:
        tokens = tokens[:get_template_length(question) + prob_token]
        full_prompt = tokenizer.decode(tokens)
    return full_prompt +f"I need to choose one from the candidate answers [{', '.join(candidate_answers)}]</think>\nThe answer is \\boxed"

def prepare_batch_messages_online(question, traces, prob_token, candidate_answers):
    batch_messages = []
    answers = []
    for trace in traces:
        if trace['stop_reason'] == 'gconf_threshold':
            continue
        message = prepare_prompt(question, trace, prob_token, candidate_answers)
        batch_messages.append(message)
        answer = trace['extracted_answer'] if trace['extracted_answer'] is not None else trace['text'][-200:]
        answers.append(answer)
    return batch_messages, answers

def collect_candidate_answers(output):
    traces = output['final_traces']
    candidate_answers = set()
    for trace in traces:
        if trace['extracted_answer'] is not None:
            candidate_answers.add(trace['extracted_answer'])
    return candidate_answers

def run_online(output, trace_budget, trace_key='final_traces', sample_count=1):
    question = output['question']
    traces = output[trace_key]

    # Sample traces given trace_budget
    if len(traces) >= trace_budget:
        selected_indices = np.random.choice(len(traces), trace_budget, replace=False)
        traces = [traces[i] for i in selected_indices]
    prob_token = 100000000

    candidate_answers = collect_candidate_answers(output)
    batch_messages, extracted_answers = prepare_batch_messages_online(question, traces, prob_token, candidate_answers)
    answers = probe_answers(batch_messages, sample_count=sample_count)

    # Write question, extracted answers and answers to a single json file
    import json
    import time
    debug_data = {
        "question": question,
        "ground_truth": output["ground_truth"],
        "extracted_answers": extracted_answers,
        "probe_answers": answers
    }
    # with open(f"./probe/debug_output_{prob_token}_{time.time()}.json", "w") as f:
    #     json.dump(debug_data, f, indent=2)
    # return max(answers, key=answers.count), sum(token_usages), prob_token

def should_stop(entropy_trace, entropy_threshold, last_n):
    # Need at least last_n entropy values to check the trend
    if len(entropy_trace) < last_n:
        return False
    
    # Check if latest entropy is below threshold
    # if abs(entropy_trace[-1] - entropy_threshold) >= 0.5:
        # return False
    
    # Check if last n steps are decreasing
    last_three = entropy_trace[-last_n:]
    is_decreasing = all(last_three[i] > last_three[i+1] for i in range(last_n-1))
    
    return is_decreasing


if __name__ == "__main__":
    import wandb

    outputs_dir = "outputs/512"
    pkl_files = glob.glob(os.path.join(outputs_dir, "*rid0_*.pkl"))
    # Sort pkl files by qid number
    def extract_qid(filename):
        # Extract qid number from filename like "deepconf_simple_qid15_rid0_20251011_145958.pkl"
        import re
        match = re.search(r'qid(\d+)', filename)
        return int(match.group(1)) if match else 0

    pkl_files.sort(key=extract_qid)
    print(pkl_files)

    for sample_count in [2, 4, 8, 16, 32]:
        for last_n in [4, 6]:
            warmup_budget = 16
            percentile = 90
            total_tokens = 0
            correct_count = 0
            num_warmup_sample_runs = 1
            entropy_threshold = 0.4
            prob_interval = 1000000
            trace_budget = 512

            wandb.init(project="probe-analysis-wdr", name=f"trace_budget-{trace_budget}-last_n-{last_n}" ,config={
                "warmup_budget": warmup_budget,
                "num_warmup_sample_runs": num_warmup_sample_runs,
                "entropy_threshold": entropy_threshold,
                "percentile": percentile,
                "trace_budget": trace_budget,
                "prob_interval": prob_interval,
                "last_n": last_n,
                "sample_count": sample_count,
            })

            table = wandb.Table(columns=["qid","warmup_trace_budget", "entropy_threshold", "probe_token", "running_token_usage", "answer", "ground truth"])

            for file_path in tqdm(pkl_files):
                try:
                    entropies = []
                    warmup_tokens = 0
                    # for run in range(num_warmup_sample_runs):
                    #     entropy, answer, num_tokens = warmup_SVP(load_data(file_path))
                    #     entropies.append(entropy)
                    #     warmup_tokens += num_tokens

                    total_tokens += warmup_tokens / num_warmup_sample_runs

                    mean_entropy = np.mean(entropies)
                    var_entropy = np.var(entropies)
                    token_usage = 0
                    prob_token = 0
                    # print(f"File: {file_path}, Budget: {warmup_budget} Mean entropy: {mean_entropy}, Variance: {var_entropy}")
                    # if mean_entropy < entropy_threshold:
                    #     correct_count += (answer == load_data(file_path)["ground_truth"])
                    # else:
                    answer, token_usage, prob_token = run_online(load_data(file_path), trace_budget, prob_interval, 555, last_n)
                    correct_count += (answer == load_data(file_path)["ground_truth"])
                    total_tokens += token_usage
                    qid = file_path.split("_")[1]
                    table.add_data(qid, warmup_budget, entropy_threshold, prob_token, warmup_tokens / num_warmup_sample_runs + token_usage, answer, load_data(file_path)["ground_truth"])
                    wandb.log({
                        "mean_entropy": mean_entropy,
                        "var_entropy": var_entropy,
                        "prob_token": prob_token,
                        "is_correct": int(answer == load_data(file_path)["ground_truth"]),
                        "running_token_usage": warmup_tokens / num_warmup_sample_runs + token_usage,
                        "qid": qid
                    }, step=extract_qid(file_path), commit=True)
                except Exception as e:
                    print(f"Error processing {file_path}: {e.with_traceback()}")
            wandb.log({"inference_reults": table})
            final_accuracy = correct_count / 30
            wandb.log({
                "final_accuracy": final_accuracy,
                "final_token_usage": total_tokens,
                "warmup_budget": warmup_budget,
                "trace_budget": trace_budget

            })
            print(f"Budget {warmup_budget} | Accuracy: {final_accuracy} | Token Usage: {total_tokens}")
            wandb.finish()
    
