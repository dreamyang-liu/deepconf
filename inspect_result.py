
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

import numpy as np
import os
from collections import defaultdict

MODEL_PATH = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
WINDOW_SIZE = 2048

import pickle
from helper import process_batch_revive_results


def prepare_prompt(question, dropped_thinking_trace, tokenizer):
    
    # Format prompt using chat template
    messages = [
        {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
        {"role": "user", "content": question},
        {"role": "assistant", "content": "Below is my thinking on the question.\n" + dropped_thinking_trace[7:]},
        {"role": "user", "content": "Based on your current thinking, directly give your short answer, don't make any reasoning or explaination."}
    ]
    
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_prompt += "<think>\n</think>"
    return full_prompt

import matplotlib.pyplot as plt

def plot_conf(group_confs):
    plt.figure(figsize=(10, 6))
    for conf in group_confs:
        plt.plot(conf)
    plt.title('Group Confidence Over Token Generation')
    plt.xlabel('Token Position')
    plt.ylabel('Group Confidence')
    plt.grid(True)
    plt.savefig('group_confs_chart.png')


def compute_confidence(logprobs):
    """Compute confidence score from logprobs"""
    confs = []
    for token_logprobs in logprobs:
        if token_logprobs:
            # vLLM returns a dict of {token_id: Logprob object}
            # Get the selected token's logprob (the one with highest probability)
            mean_logprob = np.mean([lp.logprob for lp in token_logprobs.values()])
            confs.append(round(-mean_logprob, 3))
    return confs

def compute_least_grouped(confs, group_size):
    """Compute sliding window mean confidence"""
    if len(confs) < group_size:
        return [sum(confs) / len(confs)] if confs else [0]
    
    sliding_means = []
    for i in range(len(confs) - group_size + 1):
        window = confs[i:i + group_size]
        sliding_means.append(round(sum(window) / len(window), 3))
    return sliding_means


try:

    tokenizer_init_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer_init_time = time.time() - tokenizer_init_start
    print(f"Tokenizer initialized in {tokenizer_init_time:.2f} seconds")

    # Initialize vLLM engine
    print("Initializing vLLM engine...")
    llm_init_start = time.time()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=8,
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
        extra_args={'enable_conf': True,
        'window_size': 999999,
        'threshold': -1}  # Use individual confidence bar as threshold
    )
    BASE_PATH = "/home/ubuntu/projects/deepconf/deepconf/outputs"
    for filename in os.listdir(BASE_PATH):
        with open(os.path.join(BASE_PATH, filename), 'rb') as file:
            data = pickle.load(file)
            print("Fields in the pickle file:")
            print(data['final_traces'][0].keys())
            group_confs = []
            prepared_prompts = []
            for trace in data['final_traces']:
                if trace['stop_reason'] == 'gconf_threshold':
                    group_confs.append(np.array(trace['group_confs']))
                    # prepared_prompts.append(prepare_prompt(data['question'], trace['text'], tokenizer))
            
            cache_file = f'./generation_cache_{filename.split(".")[0]}.pkl'
            if os.path.exists(cache_file):
                print(f"Loading cached outputs from {cache_file}")
                with open(cache_file, 'rb') as f:
                    final_outputs = pickle.load(f)
            else:
                #
                print("Generating outputs with LLM...")
                # final_outputs = llm.generate(prepared_prompts, continue_params)
                # Cache the results
                with open(cache_file, 'wb') as f:
                    pickle.dump(final_outputs, f)
            
            processed_results = list(process_batch_revive_results(final_outputs, data['ground_truth'], WINDOW_SIZE))
            correctness = []
            os.makedirs(f"./analysis/{filename.split(".")[0]}", exist_ok=True)
            first_token_confs = defaultdict(list)
            avg_token_confs = defaultdict(list)
            for idx, (processed_result, output) in enumerate(zip(processed_results, final_outputs)):
                trace = processed_result['traces'][0]
                prompt_conf = group_confs[idx]
                response_conf = np.array(trace['group_confs'])
                correctness.append(int(trace['is_correct']))

                conf = compute_confidence(output.outputs[0].logprobs)
                if trace['is_correct']:
                    first_token_confs['correct'].append(conf[0])
                    avg_token_confs['correct'].append(np.mean(conf))
                else:
                    first_token_confs['incorrect'].append(conf[0])
                    avg_token_confs['incorrect'].append(np.mean(conf))
            
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
            plt.savefig(f"./analysis/{filename.split('.')[0]}/first_token_conf.png")

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
            plt.savefig(f"./analysis/{filename.split('.')[0]}/avg_token_conf.png")

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
                plt.savefig(f"./analysis/{filename.split('.')[0]}/first_token_conf_violin.png")

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
                plt.savefig(f"./analysis/{filename.split('.')[0]}/avg_token_conf_violin.png")

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
                results_data["avg_token_correlation"] = {
                    "correlation": round(corr, 4),
                    "p_value": round(p_value, 4)
                }
            else:
                corr_line = f"Cannot calculate Pearson correlation for avg token conf ({filename}): insufficient data or no variation"
                print(corr_line)
                results_data["avg_token_correlation"] = {
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
                results_data["first_token_correlation"] = {
                    "correlation": round(first_corr, 4),
                    "p_value": round(first_p_value, 4)
                }
            else:
                first_corr_line = f"Cannot calculate Pearson correlation for first token conf ({filename}): insufficient data or no variation"
                print(first_corr_line)
                results_data["first_token_correlation"] = {
                    "error": "insufficient data or no variation"
                }

            correct_count = sum(correctness)
            correct_line = f"Revivied Trace Correctness ({filename}): {correct_count}/{len(final_outputs)}"
            results_data["correctness"] = {
                "correct_count": correct_count,
                "total_count": len(final_outputs),
                "accuracy": round(correct_count / len(final_outputs), 4) if len(final_outputs) > 0 else 0
            }
            print(correct_line)

            # Write results to JSON file
            import json
            with open(f"./analysis/{filename.split('.')[0]}/correlation_results.json", 'w') as f:
                json.dump(results_data, f, indent=4)

            print(f"Revivied Trace Correctness ({filename}): {correct_count}/{len(final_outputs)}")

except FileNotFoundError as e:
    print(e)
    print("File not found")
except Exception as e:
    print(f"Error loading pickle file: {e}")