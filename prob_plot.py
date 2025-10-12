import os
import json
import argparse

folder_path = "/home/ubuntu/projects/deepconf/deepconf/prob_analysis"
window_size = 1024
import matplotlib.pyplot as plt
prob_tokens = [2048, 4096, 8192, 16384, 32768]
qid = list(range(16))
count_accuracies_by_qid = {q: [] for q in qid}

# Load data for all qids
for prob_token in prob_tokens:
    for q in qid:
        filename = os.path.join(folder_path, f"analysis_results_qid{q}_probtoken_{prob_token}_windowsize_{window_size}.json")
        try:
            with open(filename, "r") as f:
                analysis_result = json.load(f)
                count_accuracies_by_qid[q].append(analysis_result["entropy_metrics"]["entropy_last_window"])
        except FileNotFoundError:
            # Handle missing files by appending None
            count_accuracies_by_qid[q].append(None)
        
# Calculate average across all questions
avg_count_accuracies = []
for i in range(len(prob_tokens)):
    values = [count_accuracies_by_qid[q][i] for q in qid if i < len(count_accuracies_by_qid[q]) and count_accuracies_by_qid[q][i] is not None]
    if values:
        avg_count_accuracies.append(sum(values) / len(values))
    else:
        avg_count_accuracies.append(None)
plt.figure(figsize=(12, 8))
# Plot each question
for q in qid:
    plt.plot(prob_tokens, count_accuracies_by_qid[q], marker='.', linestyle='--', alpha=0.3, label=f'QID {q}')

# Plot the average with a thicker line
plt.plot(prob_tokens, avg_count_accuracies, marker='o', linewidth=3, color='black', label='Average')
plt.xlabel('Probability Token')
plt.ylabel('Count Accuracy')
plt.title(f'Count Accuracy vs Probability Token (Window Size: {window_size})')
plt.grid(True)
plt.legend(loc='best', ncol=4)
plt.savefig(os.path.join(folder_path, f'../prob_plots/count_accuracy_plot_windowsize_{window_size}.png'))
plt.show()
