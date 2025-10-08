
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

import numpy as np
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from  inspect_helper import compute_voting_answer, process_batch_results, extract_structured_conf, compute_instance_accuracy, recompute_traces

MODEL_PATH = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
WINDOW_SIZE = 2048

import pickle
from helper import compute_confidence


INJECT_PROMPT_VERSION_1 = ". Now I feel confident about my existing analysis, I should directly and only give the short answer without being verbose and explaining details.</think>\n\\boxed"
INJECT_PROMPT_VERSION_2 = "</think>\n\\boxed"

PROMPT_VERSION_MAP = {
    1: INJECT_PROMPT_VERSION_1,
    2: INJECT_PROMPT_VERSION_2
}

PROMPT_VERSION = 1

def prepare_prompt(question, dropped_thinking_trace, tokenizer):
    
    # Format prompt using chat template
    messages = [
        {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
        {"role": "user", "content": question},
    ]
    
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    full_prompt += dropped_thinking_trace + PROMPT_VERSION_MAP[PROMPT_VERSION]
    return full_prompt


def get_cache_uuid(filename):
    return f"{filename.split(".")[0]}-prompt-{PROMPT_VERSION}"

def analysis(filename, processed_results, final_outputs):
    correctness = []
    os.makedirs(f"./analysis/{get_cache_uuid(filename)}", exist_ok=True)
    first_token_confs = defaultdict(list)
    avg_token_confs = defaultdict(list)
    max_token_confs = defaultdict(list)
    generated_token_length = defaultdict(list)
    output_texts = []
    for _, (processed_result, output) in enumerate(zip(processed_results, final_outputs)):
        trace = processed_result['traces'][0]
        correctness.append(int(trace['is_correct']))

        conf = compute_confidence(output.outputs[0].logprobs)
        if trace['is_correct']:
            first_token_confs['correct'].append(conf[0])
            avg_token_confs['correct'].append(np.mean(conf))
            max_token_confs['correct'].append(max(conf) if conf else 0)
            generated_token_length['correct'].append(len(conf))
        else:
            first_token_confs['incorrect'].append(conf[0])
            avg_token_confs['incorrect'].append(np.mean(conf))
            max_token_confs['incorrect'].append(max(conf) if conf else 0)
            generated_token_length['incorrect'].append(len(conf))
        output_texts.append(processed_result['traces'][0]['text'])

    # Draw scatter plot for first token confidence
    plt.figure(figsize=(10, 6))
    if first_token_confs['correct']:
        plt.scatter(range(len(first_token_confs['correct'])), first_token_confs['correct'],
                    color='green', label='Correct', alpha=0.7)
    if first_token_confs['incorrect']:
        plt.scatter(range(len(first_token_confs['incorrect'])), first_token_confs['incorrect'],
                    color='red', label='Incorrect', alpha=0.7)
    plt.title(f'First Token Confidence ({filename})')
    plt.xlabel('Sample Index')
    plt.ylabel('Confidence')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./analysis/{get_cache_uuid(filename)}/first_token_conf.png")

    # Draw scatter plot for average token confidence
    plt.figure(figsize=(10, 6))
    if avg_token_confs['correct']:
        plt.scatter(range(len(avg_token_confs['correct'])), avg_token_confs['correct'],
                    color='green', label='Correct', alpha=0.7)
    if avg_token_confs['incorrect']:
        plt.scatter(range(len(avg_token_confs['incorrect'])), avg_token_confs['incorrect'],
                    color='red', label='Incorrect', alpha=0.7)
    plt.title(f'Average Token Confidence ({filename})')
    plt.xlabel('Sample Index')
    plt.ylabel('Confidence')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./analysis/{get_cache_uuid(filename)}/avg_token_conf.png")

    # Draw scatter plot for max token confidence
    plt.figure(figsize=(10, 6))
    if max_token_confs['correct']:
        plt.scatter(range(len(max_token_confs['correct'])), max_token_confs['correct'],
                    color='green', label='Correct', alpha=0.7)
    if max_token_confs['incorrect']:
        plt.scatter(range(len(max_token_confs['incorrect'])), max_token_confs['incorrect'],
                    color='red', label='Incorrect', alpha=0.7)
    plt.title(f'Max Token Confidence ({filename})')
    plt.xlabel('Sample Index')
    plt.ylabel('Confidence')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./analysis/{get_cache_uuid(filename)}/max_token_conf.png")

    # Draw violin plot for first token confidence
    plt.figure(figsize=(10, 6))
    data_to_plot = []
    labels = []
    if first_token_confs['correct']:
        data_to_plot.append(first_token_confs['correct'])
        labels.append('Correct')
    if first_token_confs['incorrect']:
        data_to_plot.append(first_token_confs['incorrect'])
        labels.append('Incorrect')
    if data_to_plot:
        plt.violinplot(data_to_plot, showmeans=True, showmedians=True)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.title(f'First Token Confidence Distribution ({filename})')
        plt.ylabel('Confidence')
        plt.grid(True)
        plt.savefig(f"./analysis/{get_cache_uuid(filename)}/first_token_conf_violin.png")

    # Draw violin plot for average token confidence
    plt.figure(figsize=(10, 6))
    data_to_plot = []
    labels = []
    if avg_token_confs['correct']:
        data_to_plot.append(avg_token_confs['correct'])
        labels.append('Correct')
    if avg_token_confs['incorrect']:
        data_to_plot.append(avg_token_confs['incorrect'])
        labels.append('Incorrect')
    if data_to_plot:
        plt.violinplot(data_to_plot, showmeans=True, showmedians=True)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.title(f'Average Token Confidence Distribution ({filename})')
        plt.ylabel('Confidence')
        plt.grid(True)
        plt.savefig(f"./analysis/{get_cache_uuid(filename)}/avg_token_conf_violin.png")

    # Draw violin plot for max token confidence
    plt.figure(figsize=(10, 6))
    data_to_plot = []
    labels = []
    if max_token_confs['correct']:
        data_to_plot.append(max_token_confs['correct'])
        labels.append('Correct')
    if max_token_confs['incorrect']:
        data_to_plot.append(max_token_confs['incorrect'])
        labels.append('Incorrect')
    if data_to_plot:
        plt.violinplot(data_to_plot, showmeans=True, showmedians=True)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.title(f'Max Token Confidence Distribution ({filename})')
        plt.ylabel('Confidence')
        plt.grid(True)
        plt.savefig(f"./analysis/{get_cache_uuid(filename)}/max_token_conf_violin.png")

    # Calculate Pearson coefficient between correctness and average token confidence
    correct_confs = avg_token_confs['correct']
    incorrect_confs = avg_token_confs['incorrect']

    # Prepare data for correlation
    all_confs = correct_confs + incorrect_confs
    all_correctness = [1] * len(correct_confs) + [0] * len(incorrect_confs)

    # Prepare results for JSON
    results_data = {}
    if len(all_confs) > 1 and len(set(all_correctness)) > 1:  # Ensure we have variation in both variables
        from scipy.stats import pearsonr
        corr, p_value = pearsonr(all_correctness, all_confs)
        corr_line = f"Pearson correlation coefficient for avg token conf ({filename}): {corr:.4f} (p-value: {p_value:.4f})"
        print(corr_line)
        results_data["avg_token_conf_correlation"] = {
            "correlation": round(corr, 4),
            "p_value": round(p_value, 4)
        }
    else:
        corr_line = f"Cannot calculate Pearson correlation for avg token conf ({filename}): insufficient data or no variation"
        print(corr_line)
        results_data["avg_token_conf_correlation"] = {
            "error": "insufficient data or no variation"
        }

    # Calculate Pearson coefficient for first token confidence
    correct_first_confs = first_token_confs['correct']
    incorrect_first_confs = first_token_confs['incorrect']

    # Prepare data for correlation
    all_first_confs = correct_first_confs + incorrect_first_confs
    all_first_correctness = [1] * len(correct_first_confs) + [0] * len(incorrect_first_confs)

    if len(all_first_confs) > 1 and len(set(all_first_correctness)) > 1:  # Ensure we have variation in both variables
        from scipy.stats import pearsonr
        first_corr, first_p_value = pearsonr(all_first_correctness, all_first_confs)
        first_corr_line = f"Pearson correlation coefficient for first token conf ({filename}): {first_corr:.4f} (p-value: {first_p_value:.4f})"
        print(first_corr_line)
        results_data["first_token_conf_correlation"] = {
            "correlation": round(first_corr, 4),
            "p_value": round(first_p_value, 4)
        }
    else:
        first_corr_line = f"Cannot calculate Pearson correlation for first token conf ({filename}): insufficient data or no variation"
        print(first_corr_line)
        results_data["first_token_conf_correlation"] = {
            "error": "insufficient data or no variation"
        }

    # Calculate Pearson coefficient for max token confidence
    correct_max_confs = max_token_confs['correct']
    incorrect_max_confs = max_token_confs['incorrect']

    # Prepare data for correlation
    all_max_confs = correct_max_confs + incorrect_max_confs
    all_max_correctness = [1] * len(correct_max_confs) + [0] * len(incorrect_max_confs)

    if len(all_max_confs) > 1 and len(set(all_max_correctness)) > 1:  # Ensure we have variation in both variables
        from scipy.stats import pearsonr
        max_corr, max_p_value = pearsonr(all_max_correctness, all_max_confs)
        max_corr_line = f"Pearson correlation coefficient for max token conf ({filename}): {max_corr:.4f} (p-value: {max_p_value:.4f})"
        print(max_corr_line)
        results_data["max_token_conf_correlation"] = {
            "correlation": round(max_corr, 4),
            "p_value": round(max_p_value, 4)
        }
    else:
        max_corr_line = f"Cannot calculate Pearson correlation for max token conf ({filename}): insufficient data or no variation"
        print(max_corr_line)
        results_data["max_token_conf_correlation"] = {
            "error": "insufficient data or no variation"
        }

    correct_count = sum(correctness)
    correct_line = f"Revivied Trace Correctness ({filename}): {correct_count}/{len(final_outputs)}"
    results_data["correctness"] = {
        "correct_count": correct_count,
        "total_count": len(final_outputs),
        "accuracy": round(correct_count / len(final_outputs), 4) if len(final_outputs) > 0 else 0
    }
    results_data["generation"] = {
        "macro_max_length": max(generated_token_length['correct'] + generated_token_length['incorrect']),
        "macro_min_length": min(generated_token_length['correct'] + generated_token_length['incorrect']),
        "macro_average_length": np.mean(max(generated_token_length['correct'] + generated_token_length['incorrect'])),
        "correct_average_length": np.mean(generated_token_length['correct']) if generated_token_length['correct'] else 0,
        "incorrect_average_length": np.mean(generated_token_length['incorrect']) if generated_token_length['incorrect'] else 0,
        "correct_min_length": min(generated_token_length['correct']) if generated_token_length['correct'] else 0,
        "incorrect_min_length": min(generated_token_length['incorrect']) if generated_token_length['incorrect'] else 0,
        "correct_max_length": max(generated_token_length['correct']) if generated_token_length['correct'] else 0,
        "incorrect_max_length": max(generated_token_length['incorrect']) if generated_token_length['incorrect'] else 0
    }
    print(correct_line)

    # Write results to JSON file
    import json
    with open(f"./analysis/{get_cache_uuid(filename)}/correlation_results.json", 'w') as f:
        json.dump(results_data, f, indent=4)
    
    with open(f"./analysis/{get_cache_uuid(filename)}/outputs_text.json", 'w') as f:
        json.dump(output_texts, f, indent=4)

    print(f"Revivied Trace Correctness ({filename}): {correct_count}/{len(final_outputs)}")

try:
    import argparse
    parser = argparse.ArgumentParser(description="Analysis script")
    parser.add_argument('--enable-llm', action='store_true', help="Enable LLM engine initialization")
    parser.add_argument('--budget', type=int, required=True)
    args = parser.parse_args()
    tokenizer_init_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer_init_time = time.time() - tokenizer_init_start
    print(f"Tokenizer initialized in {tokenizer_init_time:.2f} seconds")

    # Initialize vLLM engine
    if args.enable_llm:
        print("Initializing vLLM engine...")
        llm_init_start = time.time()
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        llm_init_time = time.time() - llm_init_start

        continue_params = SamplingParams(
            n=1,
            temperature=0.6,
            top_p=0.95,
            max_tokens=64000,
            logprobs=20,
            extra_args={'enable_conf': False,
            'window_size': 999999,
            'threshold': -1} 
        )
    BASE_PATH = "/home/ubuntu/projects/deepconf/deepconf/outputs"
    macro_accuracies = defaultdict(lambda : defaultdict(list))

    # Draw 3x3 scatter plots for structured_conf vs instance_accuracy
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    trace_types = ['completed_traces', 'truncated_traces', 'combined_traces']
    conf_types = ['min_conf', 'max_conf', 'avg_conf']
    deepconf_correctness = []
    for filename in os.listdir(BASE_PATH):
        # Only iterate all question in one round, remove it when finish all trj
        if 'rid1' in filename or f'budget{args.budget}' not in filename:
            continue
        with open(os.path.join(BASE_PATH, filename), 'rb') as file:
            data = pickle.load(file)
            deepconf_correctness.append(data['is_voted_correct'])
            print(data['is_voted_correct'], filename)
            group_confs = []
            kept_group_confs = []
            prepared_prompts = []
            completed_traces = []
            for trace in data['final_traces']:
                if trace['stop_reason'] == 'gconf_threshold':
                    group_confs.append(np.array(trace['group_confs']))
                    if args.enable_llm:
                        prepared_prompts.append(prepare_prompt(data['question'], trace['text'], tokenizer))
                else:
                    kept_group_confs.append(np.array(trace['group_confs']))
                    completed_traces.append(trace)
            
            cache_file = f'./cache/generation_cache_{get_cache_uuid(filename)}.pkl'
            if os.path.exists(cache_file):
                print(f"Loading cached outputs from {cache_file}")
                with open(cache_file, 'rb') as f:
                    truncated_trace_generation_outputs = pickle.load(f)
            else:
                #
                print("Generating outputs with LLM...")
                if args.enable_llm:
                    truncated_trace_generation_outputs = llm.generate(prepared_prompts, continue_params)
                # Cache the results
                with open(cache_file, 'wb') as f:
                    pickle.dump(truncated_trace_generation_outputs, f)
            os.makedirs(f"./analysis/{get_cache_uuid(filename)}", exist_ok=True)
            completed_traces = recompute_traces(completed_traces, tokenizer)
            processed_results = list(process_batch_results(truncated_trace_generation_outputs, data['ground_truth'], WINDOW_SIZE, '\\boxed', tokenizer))

            # Write completed traces and truncated traces
            truncated_traces_token_confs = []
            for trace in processed_results:
                truncated_traces_token_confs.append({
                    "answer_token_conf": trace['traces'][0]['answer_token_conf'],
                    "extracted_answer": trace['traces'][0]['extracted_answer']
                })
            completed_traces_token_confs = []
            for trace in completed_traces:
                completed_traces_token_confs.append({
                    "answer_token_conf": trace['answer_token_conf'],
                    "extracted_answer": trace['extracted_answer']
                })
            with open(f"./analysis/{get_cache_uuid(filename)}/answer_token_conf_budget{args.budget}.json", "w") as f:
                json.dump({
                    "completed": completed_traces_token_confs,
                    "truncated": truncated_traces_token_confs,
                    "ground_truth": data["ground_truth"]
                }, f, indent=2)

            voting_answer = compute_voting_answer(completed_traces, processed_results)
            voting_answer['ground_truth'] = data['ground_truth']
            with open(f"./analysis/{get_cache_uuid(filename)}/conf_exp_budget{args.budget}.json", "w") as f:
                json.dump(voting_answer, f)
            structured_conf = extract_structured_conf(completed_traces, processed_results)
            instance_accuracy = compute_instance_accuracy(completed_traces, processed_results, data['ground_truth'])

            for i, conf_type in enumerate(conf_types):
                for j, trace_type in enumerate(trace_types):
                    ax = axes[i, j]
                    conf_value = structured_conf[conf_type][trace_type]
                    acc_value = instance_accuracy[trace_type]

                    ax.scatter(conf_value, acc_value, color='blue')
                    ax.set_title(f"{conf_type} vs. Accuracy ({trace_type})")
                    ax.set_xlabel(f"{conf_type}")
                    ax.set_ylabel("Accuracy")
                    ax.grid(True)

            plt.tight_layout()

            for trace_type in ["completed_traces", "truncated_traces", "combined_traces"]:
                for conf_type in ["max_conf", "min_conf", "avg_conf"]:
                    macro_accuracies[conf_type][trace_type].append(
                        voting_answer[conf_type][trace_type][0] == voting_answer['ground_truth'])
            
    plt.savefig(f"./analysis/conf_accuracy_scatter_budget{args.budget}.png")
    plt.close(fig)
    # Calculate average accuracy for all combinations
    for conf_type in ["max_conf", "min_conf", "avg_conf"]:
        for trace_type in ["completed_traces", "truncated_traces", "combined_traces"]:
            macro_accuracies[conf_type][f"{trace_type}_accuracy"] = np.mean(macro_accuracies[conf_type][trace_type])

    macro_accuracies["combined"]["metadata"] = [MODEL_PATH, WINDOW_SIZE, PROMPT_VERSION_MAP[PROMPT_VERSION]]
    macro_accuracies["deepconf_accuracy"] = np.mean(deepconf_correctness)
    with open(f"./analysis/accuracy_results_budget{args.budget}.json", "w") as f:
        json.dump(macro_accuracies, f)


except FileNotFoundError as e:
    print(e)
    print("File not found")
except Exception as e:
    print(f"Error loading pickle file: {e.with_traceback()}")