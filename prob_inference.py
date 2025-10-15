import os
from probe_utils import prepare_batch_messages, probe_answers
import pickle
from collections import Counter
import numpy as np
from functools import cache


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

    return compute_answer_entropy(answers)


def run_offline(output, trace_budget, prob_interval, entropy_threshold, last_n):
    question = output['question']
    traces = output['all_traces']
    # Sample traces given trace_budget
    if len(traces) >= trace_budget:
        selected_indices = np.random.choice(len(traces), trace_budget, replace=False)
        traces = [traces[i] for i in selected_indices]
    entropy_trace = []
    for prob_token in range(prob_interval, 65536):
        batch_messages = prepare_batch_messages(question, traces, prob_token)
        answers = probe_answers(batch_messages)
        answer_count_entropy = compute_answer_entropy(answers)
        entropy_trace.append(answer_count_entropy)
        if should_stop(entropy_trace, entropy_threshold, last_n):
            return max(answers, key=answers.count)


def should_stop(entropy_trace, entropy_threshold, last_n):
    # Need at least last_n entropy values to check the trend
    if len(entropy_trace) < last_n:
        return False
    
    # Check if latest entropy is below threshold
    if entropy_trace[-1] >= entropy_threshold:
        return False
    
    # Check if last n steps are decreasing
    last_three = entropy_trace[-last_n:]
    is_decreasing = all(last_three[i] > last_three[i+1] for i in range(last_n-1))
    
    return is_decreasing

if __name__ == "__main__":
    import glob
    from tqdm import trange

    import matplotlib.pyplot as plt

    outputs_dir = "outputs"
    pkl_files = glob.glob(os.path.join(outputs_dir, "*.pkl"))
    warmup_budget = 16
    percentile = 0.95
    num_runs = 64

    for file_path in pkl_files:
        try:
            # plt.figure(figsize=(15, 5))
            for i, budget in enumerate([16, 32, 64]):
                entropies = []
                for run in trange(num_runs):
                    entropy = warmup_from_file(file_path, budget)
                    entropies.append(entropy)

                mean_entropy = np.mean(entropies)
                var_entropy = np.var(entropies)

                print(f"File: {file_path}, Budget: {budget} Mean entropy: {mean_entropy}, Variance: {var_entropy}", end='')
                if mean_entropy < 0.5:
                    print(load_data(file_path)['is_voted_correct'])
                else:
                    print()
                # Plot histogram for each budget
            #     plt.subplot(1, 3, i+1)
            #     plt.hist(entropies, bins=20, alpha=0.7, density=True)
            #     plt.title(f'Budget: {budget}\nMean: {mean_entropy:.3f}, Var: {var_entropy:.3f}')
            #     plt.xlabel('Entropy')
            #     plt.ylabel('Density')

            # plt.suptitle(f'Entropy Distributions - {os.path.basename(file_path)}')
            # plt.tight_layout()

            answer = run_offline(load_data(file_path), 128, 1024, np.percentile(entropies, 90), 3)
            print(answer, load_data(file_path)["ground_truth"])
            # Save to local file
            # filename = os.path.splitext(os.path.basename(file_path))[0]
            # save_path = f'entropy_distributions_{filename}.png'
            # plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # print(f"Saved plot to {save_path}")

            # plt.show()

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
