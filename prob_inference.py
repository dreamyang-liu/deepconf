import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob

from functools import cache
from tqdm import tqdm, trange
from collections import Counter
from probe_utils import prepare_batch_messages, probe_answers


import math

def shift1_geometric_mean(values):
    """
    Calculate the shift-1 geometric mean of a list of values.
    
    The shift-1 geometric mean is calculated as:
    GM = (∏(xi + 1))^(1/n) - 1
    
    This is useful for data that may contain zeros or negative values close to zero,
    as adding 1 to each value ensures all inputs to the geometric mean are positive.
    
    Args:
        values: List or iterable of numeric values
        
    Returns:
        float: The shift-1 geometric mean
        
    Raises:
        ValueError: If any value is <= -1 (which would make xi + 1 <= 0)
        ValueError: If the input is empty
    """
    if not values:
        raise ValueError("Cannot calculate geometric mean of empty sequence")
    
    values_list = list(values)
    
    # Check that all shifted values are positive
    for val in values_list:
        if val <= -1:
            raise ValueError(f"Value {val} would result in non-positive shifted value")
    
    n = len(values_list)
    
    # Calculate using logarithms to avoid overflow with large numbers
    log_sum = sum(math.log(val + 1) for val in values_list)
    
    # Calculate geometric mean and subtract 1
    geometric_mean = math.exp(log_sum / n)
    
    return geometric_mean - 1

@cache
def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data

def compute_answer_entropy(answers):
    # Count different answers
    answer_counts = Counter(answers)

    # Calculate entropy from answer counts
    total_count = sum(answer_counts.values())
    probabilities = [count / total_count for count in answer_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy

def warmup_from_file(path, warmup_budget):
    data = load_data(path)
    # Extract traces from the loaded data
    traces = data['all_traces']

    # Select warmup_budget number of traces using random sampling
    if len(traces) >= warmup_budget:
        selected_indices = np.random.choice(len(traces), warmup_budget, replace=False)
        selected_traces = [traces[i] for i in selected_indices]
    else:
        selected_traces = traces

    answers = [trace['extracted_answer'] for trace in selected_traces]
    num_tokens = sum(trace['num_tokens'] for trace in selected_traces)

    return compute_answer_entropy(answers), max(answers, key=answers.count), num_tokens


def run_offline(output, trace_budget, prob_interval, entropy_threshold, last_n):
    question = output['question']
    traces = output['all_traces']
    max_len = max(trace['num_tokens'] for trace in traces)
    # Sample traces given trace_budget
    if len(traces) >= trace_budget:
        selected_indices = np.random.choice(len(traces), trace_budget, replace=False)
        traces = [traces[i] for i in selected_indices]
    entropy_trace = []
    for prob_token in trange(prob_interval, max_len + prob_interval + 1, prob_interval):
        batch_messages, token_usages = prepare_batch_messages(question, traces, prob_token)
        answers = probe_answers(batch_messages)
        answer_count_entropy = compute_answer_entropy(answers)
        entropy_trace.append(answer_count_entropy)
        if should_stop(entropy_trace, entropy_threshold, last_n) or prob_token >= max_len:
            return max(answers, key=answers.count), sum(token_usages), prob_token
    

def run_online(output, trace_budget, prob_interval, entropy_threshold, last_n):
    question = output['question']
    traces = output['all_traces']
    max_len = max(trace['num_tokens'] for trace in traces)
    # Sample traces given trace_budget
    if len(traces) >= trace_budget:
        selected_indices = np.random.choice(len(traces), trace_budget, replace=False)
        traces = [traces[i] for i in selected_indices]
    entropy_trace = []
    for prob_token in trange(prob_interval, max_len + prob_interval + 1, prob_interval):
        batch_messages, token_usages = prepare_batch_messages(question, traces, prob_token)
        answers = probe_answers(batch_messages)
        answer_count_entropy = compute_answer_entropy(answers)
        entropy_trace.append(answer_count_entropy)
        if should_stop(entropy_trace, entropy_threshold, last_n) or prob_token >= max_len:
            return max(answers, key=answers.count), sum(token_usages), prob_token


def should_stop(entropy_trace, entropy_threshold, last_n):
    # Need at least last_n entropy values to check the trend
    if len(entropy_trace) < last_n:
        return False
    
    # Check if latest entropy is below threshold
    if abs(entropy_trace[-1] - entropy_threshold) < 0.2:
        return False
    
    # Check if last n steps are decreasing
    last_three = entropy_trace[-last_n:]
    is_decreasing = all(last_three[i] > last_three[i+1] for i in range(last_n-1))
    
    return is_decreasing


if __name__ == "__main__":
    tokens = [
        137385, 
        144087,
        1450341,
        1504699,
        284677,
        1144367,
        1565496,
        2065822,
        612344,
        2700486,
        523632,
        1815648,
        1788745,
        2331374,
        1510140,
        2292651,
        1090267,
        2334702,
        702245,
        2526475,
        202982,
        1216488,
        278574,
        487872,
        879154,
        377053,
        335622,
        962769,
        2459933,
        717667
    ]
    print(f"Count Entropy Descend Shift-1 Geometric Mean: {shift1_geometric_mean(tokens):,.2f}")
    print(f"Count Entropy Descend Arithmetic Mean: {np.mean(tokens):,.2f}")
    print(f"Count Entropy Descend Sum: {np.sum(tokens):,.0f}")
    print(f"Count Entropy Descend Accuracy: {19 / 30:,.2f}")

    tokens2 = [
         351053.0,
        368911.0,
        820809.0,
        873716.0,
        523486.0,
        934970.0,
        913049.0,
        967309.0,
        717357.0,
        2277428.0,
        595799.0,
        946596.0,
        2210159.0,
        1181400.0,
        1292459.0,
        585636.0,
        1230007.0,
        1659906.0,
        2342401.0,
        1081763.0,
        380878.0,
        447381.0,
        567731.0,
        987352.0,
        1673567.0,
        691482.0,
        652586.0,
        523512.0,
        2367190.0,
        2033033.0,
    ]
    print(f"Deep Conf Online 64 Shift-1 Geometric Mean: {shift1_geometric_mean(tokens2):,.2f}")
    print(f"Deep Conf Online 64 Arithmetic Mean: {np.mean(tokens2):,.2f}")
    print(f"Deep Conf Online 64 Sum: {np.sum(tokens2):,.0f}")
    print(f"Deep Conf Online 64 Accuracy: {23 / 30:,.2f}")

    # import wandb

    # outputs_dir = "outputs"
    # pkl_files = glob.glob(os.path.join(outputs_dir, "*rid0_*.pkl"))
    # # Sort pkl files by qid number
    # def extract_qid(filename):
    #     # Extract qid number from filename like "deepconf_simple_qid15_rid0_20251011_145958.pkl"
    #     import re
    #     match = re.search(r'qid(\d+)', filename)
    #     return int(match.group(1)) if match else 0

    # pkl_files.sort(key=extract_qid)
    # print(pkl_files)

    # for trace_budget in [64, 128, 256]:
    #     for last_n in [3, 5]:
    #         warmup_budget = 16
    #         percentile = 90
    #         total_tokens = 0
    #         correct_count = 0
    #         num_warmup_sample_runs = 8
    #         entropy_threshold = 0.4
    #         prob_interval = 4096
    #         # trace_budget = 128

    #         wandb.init(project="probe-analysis-wdr", name=f"trace_budget-{trace_budget}-last_n-{last_n}" ,config={
    #             "warmup_budget": warmup_budget,
    #             "num_warmup_sample_runs": num_warmup_sample_runs,
    #             "entropy_threshold": entropy_threshold,
    #             "percentile": percentile,
    #             "trace_budget": trace_budget,
    #             "prob_interval": prob_interval,
    #             "last_n": last_n
    #         })

    #         table = wandb.Table(columns=["qid","warmup_trace_budget", "entropy_threshold", "probe_token", "answer", "ground truth"], log_mode="INCREMENTAL")

    #         for file_path in tqdm(pkl_files):
    #             try:
    #                 entropies = []
    #                 warmup_tokens = 0
    #                 for run in range(num_warmup_sample_runs):
    #                     entropy, answer, num_tokens = warmup_from_file(file_path, warmup_budget)
    #                     entropies.append(entropy)
    #                     warmup_tokens += num_tokens

    #                 total_tokens += warmup_tokens / num_warmup_sample_runs

    #                 mean_entropy = np.mean(entropies)
    #                 var_entropy = np.var(entropies)
    #                 token_usage = 0
    #                 prob_token = 0
    #                 print(f"File: {file_path}, Budget: {warmup_budget} Mean entropy: {mean_entropy}, Variance: {var_entropy}")
    #                 if mean_entropy < entropy_threshold:
    #                     correct_count += (answer == load_data(file_path)["ground_truth"])
    #                 else:
    #                     answer, token_usage, prob_token = run_offline(load_data(file_path), trace_budget, prob_interval, np.percentile(entropies, percentile), last_n)
    #                     correct_count += (answer == load_data(file_path)["ground_truth"])
    #                     total_tokens += token_usage
    #                 qid = file_path.split("_")[2]
    #                 table.add_data(qid, warmup_budget, entropy_threshold, prob_token, answer, load_data(file_path)["ground_truth"])
    #                 wandb.log({
    #                     "inference_reults": table,
    #                     "mean_entropy": mean_entropy,
    #                     "var_entropy": var_entropy,
    #                     "prob_token": prob_token,
    #                     "is_correct": int(answer == load_data(file_path)["ground_truth"]),
    #                     "running_token_usage": warmup_tokens / num_warmup_sample_runs + token_usage,
    #                     "qid": qid
    #                 }, step=extract_qid(file_path), commit=True)
    #             except Exception as e:
    #                 print(f"Error processing {file_path}: {e}")

    #         final_accuracy = correct_count / 30
    #         wandb.log({
    #             "final_accuracy": final_accuracy,
    #             "final_token_usage": total_tokens,
    #             "warmup_budget": warmup_budget,
    #             "trace_budget": trace_budget

    #         })
    #         print(f"Budget {warmup_budget} | Accuracy: {final_accuracy} | Token Usage: {total_tokens}")
    #         wandb.finish()
    
