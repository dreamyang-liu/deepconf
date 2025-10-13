import os
import json

folder_path = "/home/ubuntu/projects/deepconf/deepconf/prob_analysis"
window_size = 1024
import matplotlib.pyplot as plt

import glob

prob_files = glob.glob("./prob_analysis/analysis_results_qid29_probtoken_*_windowsize_1024.json")

prob_tokens = sorted([int(prob_file.split("_")[5]) for prob_file in prob_files])

qid = list(range(30))
count_accuracies_by_qid = {q: [] for q in qid}

METRIC_TITLE = "Mean Window Conf Entropy VS Accuracy"

# Load data for all qids
for prob_token in prob_tokens:
    for q in qid:
        filename = os.path.join(folder_path, f"analysis_results_qid{q}_probtoken_{prob_token}_windowsize_{window_size}.json")
        try:
            with open(filename, "r") as f:
                analysis_result = json.load(f)
                # Calculate normalized accuracy by dividing mean confidence of correct answer
                # by the sum of mean confidences of all answers
                correct_answer = analysis_result["ground_truth"]
                correct_mean_conf = analysis_result["answers"][correct_answer]["mean_window_conf"] if correct_answer in analysis_result["answers"] else 0
                total_mean_conf = sum(answer_data["mean_window_conf"] for answer_data in analysis_result["answers"].values())

                # Calculate normalized accuracy (0 if correct answer isn't in answers)
                normalized_accuracy = correct_mean_conf / total_mean_conf if total_mean_conf > 0 else 0
                analysis_result["normalized_accuracy"] = normalized_accuracy
                # metric = analysis_result["count_accuracy"]
                metric = analysis_result["normalized_accuracy"]

                count_accuracies_by_qid[q].append((metric, analysis_result["entropy_metrics"]["entropy_mean_window"], correct_answer==list(analysis_result["answers"].keys())[0]))
        except FileNotFoundError:
            # Handle missing files by appending None
            # print(f"analysis_results_qid{q}_probtoken_{prob_token}_windowsize_{window_size}.json")
            count_accuracies_by_qid[q].append(count_accuracies_by_qid[q][-1])
            # pass
        


# Plot each question in a separate figure
for q in qid:
    fig, ax1 = plt.subplots(figsize=(12, 8))
    # Create second y-axis
    ax2 = ax1.twinx()

    # Determine colors based on count_accuracies_by_qid[q][2]
    colors = ['green' if c[2] else 'red' for c in count_accuracies_by_qid[q]]

    # Plot c[0] on left y-axis with conditional colors
    ax1.scatter(prob_tokens, [c[0] for c in count_accuracies_by_qid[q]], c=colors, marker='o', alpha=0.7, label='Accuracy')
    ax1.plot(prob_tokens, [c[0] for c in count_accuracies_by_qid[q]], linestyle='--', alpha=0.7, color='blue')

    # Plot c[1] on right y-axis
    ax2.plot(prob_tokens, [c[1] for c in count_accuracies_by_qid[q]], marker='o', linestyle='-', alpha=0.7, color='red', label='Entropy')

    # Set labels and titles
    ax1.set_xlabel('Prob Token')
    ax1.set_ylabel('Accuracy values', color='blue')
    ax2.set_ylabel('Entropy values', color='red')
    ax1.set_title(f'{METRIC_TITLE} vs Prob Token - QID {q} (Window Size: {window_size})')

    # Color the y-axis labels to match the data
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    ax1.grid(True)
    plt.savefig(os.path.join(folder_path, f'../prob_plots/{"_".join(METRIC_TITLE.lower().split(" "))}_plot_qid_{q}_windowsize_{window_size}.png'))
    plt.show()