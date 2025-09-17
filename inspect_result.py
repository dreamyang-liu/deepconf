
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

import numpy as np

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
        {"role": "user", "content": "Based on your current thinking, give your answer"}
    ]
    
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
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

    with open('/home/ubuntu/projects/deepconf/deepconf/outputs/deepconf_qid0_rid0_20250917_032354.pkl', 'rb') as file:
        data = pickle.load(file)
        print("Fields in the pickle file:")
        print(data['final_traces'][0].keys())
        group_confs = []
        prepared_prompts = []
        for trace in data['final_traces']:
            if trace['stop_reason'] == 'gconf_threshold':
                group_confs.append(np.array(trace['group_confs']))
                prepared_prompts.append(prepare_prompt(data['question'], trace['text'], tokenizer))
        
        import os
        cache_file = 'final_outputs_cache.pkl'

        if os.path.exists(cache_file):
            print(f"Loading cached outputs from {cache_file}")
            with open(cache_file, 'rb') as f:
                final_outputs = pickle.load(f)
        else:
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
            print("Generating outputs with LLM...")
            final_outputs = llm.generate(prepared_prompts, continue_params)
            # Cache the results
            with open(cache_file, 'wb') as f:
                pickle.dump(final_outputs, f)
        
        processed_results = list(process_batch_revive_results(final_outputs, data['ground_truth'], WINDOW_SIZE))
        # breakpoint()
        correct_count = 0
        for idx, result in enumerate(processed_results):
            trace = result['traces'][0]
            prompt_conf = group_confs[idx]
            response_conf = np.array(trace['group_confs'])

            correct_count += trace['is_correct']

            # Plot confidence scores
            plt.figure(figsize=(12, 6))
            # Create x-axis values for both arrays
            x_prompt = np.arange(len(prompt_conf))
            x_response = np.arange(len(prompt_conf), len(prompt_conf) + len(response_conf))

            # Plot with different colors
            plt.plot(x_prompt, prompt_conf, color='blue', label='Prompt Confidence')
            plt.plot(x_response, response_conf, color='red', label='Response Confidence')

            # Add a vertical line to mark the transition
            plt.axvline(x=len(prompt_conf), color='black', linestyle='--', alpha=0.7)

            plt.title('Confidence Scores for Prompt and Response')
            plt.xlabel('Token Position')
            plt.ylabel('Confidence Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'./fig/prompt_response_confidence-{idx}.png')
        print(f"Revivied Trace Correctness: {correct_count}/{len(final_outputs)}")

except FileNotFoundError:
    print("File not found")
except Exception as e:
    print(f"Error loading pickle file: {e}")