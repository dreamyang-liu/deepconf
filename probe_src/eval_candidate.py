import os
import json
from collections import defaultdict

def analyze_dp_probe_results():
    directory = "/home/ec2-user/projects/deepconf/dp_probe_results"

    # Dictionary to store results by question_id
    results_by_qid = defaultdict(list)

    # Read all JSON files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('probe.json'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    qid = data['question_id']
                    results_by_qid[qid].append(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading {filename}: {e}")

    # Analyze results
    total_correct = 0
    total_samples = 0

    print("Analysis Results:")
    print("================")

    for qid in sorted(results_by_qid.keys()):
        samples = results_by_qid[qid]
        correct_count = sum(1 for sample in samples if sample['aggregation']['is_correct'])

        print(f"\nQuestion ID: {qid}")
        print(f"Number of bootstrap samples: {len(samples)}")
        print(f"Correct samples: {correct_count}/{len(samples)}")
        print(f"Ground truth: {samples[0]['ground_truth']}")
        print(f"Question: {samples[0]['question'][:100]}...")

        # Show majority answers from each sample
        majority_answers = [sample['aggregation']['majority_answer'] for sample in samples]
        # print(len(majority_answers))
        unique_majority_answers = list(set(majority_answers))
        print(f"Majority answers across samples: {unique_majority_answers}")

        total_correct += correct_count
        total_samples += len(samples)

    # Calculate overall accuracy
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    print(f"\n" + "="*50)
    print(f"OVERALL RESULTS:")
    print(f"Total samples: {total_samples}")
    print(f"Total correct: {total_correct}")
    print(f"Overall accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"Number of unique questions: {len(results_by_qid)}")

analyze_dp_probe_results()