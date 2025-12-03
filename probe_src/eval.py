import glob
import pickle
import os
from functools import cache
from collections import Counter
from helper import equal_func
from probe_utils import get_tokenizer
@cache
def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data


dirs = glob.glob("probe_results/deepseek-8b/*")
token_usages = []
accuracies = []

tokenizer = get_tokenizer()

def process_directory(args):
    d, token = args
    available_tokens = list(map(int, os.listdir(f"{d}/")))
    if token not in available_tokens:
        token = max(available_tokens)
    file_name = os.listdir(f"{d}/{token}")[0]
    data = load_data(f"{d}/{token}/{file_name}")
    counter = Counter(data['probe_answers_with_candidates'])
    # Aggregate keys that represent the same formula
    aggregated_counter = Counter()
    for key in counter:
        # Find if this key is equivalent to any existing aggregated key
        found_equivalent = False
        for agg_key in aggregated_counter:
            if equal_func(key, agg_key):
                aggregated_counter[agg_key] += counter[key]
                found_equivalent = True
                break

        if not found_equivalent:
            aggregated_counter[key] = counter[key]

    counter = aggregated_counter
    max_key = max(counter, key=counter.get) if counter else None
    is_correct = equal_func(max_key, data['ground_truth'])
    if not is_correct:
        print("-------------------------------")
        print(f"Answer: {max_key}")
        print(f"Truth : {data['ground_truth']}")
    token_usgs = [len(tokenizer.encode(prompt)) for prompt in data['prompts']]
    # print(token_usgs)
    return is_correct, sum(token_usgs)

def run_analysis(token):
    correct = 0
    print(f"==================== Token {token} ====================")
    token_usage = 0
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(process_directory, [(d, token) for d in dirs]))

    for is_correct, usage in results:
        if is_correct:
            correct += 1
        token_usage += usage

    accuracy = correct / 30
    print(f"Accuracy: {accuracy} Token Usage: {token_usage}")
    accuracies.append(accuracy)
    token_usages.append(token_usage)


if __name__ == "__main__":
    run_analysis(9999999)

# for token in range(1 * 2048, 2048 * 30, 2048):
    # run_analysis(token)