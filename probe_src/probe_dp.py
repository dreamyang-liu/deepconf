"""
Dynamic Programming Probe - Uses candidate answers from traces to guide new inferences.

This script:
1. Loads offline DeepConf data
2. Extracts candidate answers from existing traces
3. Injects candidates into prompts using get_inject_prompt_v2
4. Generates new answers with vLLM
5. Aggregates final results
"""

import os
import sys
import pickle
import json
import argparse
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from helper import extract_answer, equal_func
from functools import cache
import glob

# Model configuration
MODEL_PATH = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MAX_TOKENS = 64000

# Global instances
ENGINE = None
TOKENIZER = None


def get_inject_prompt_v2(candidates=None):
    """Version with clearer override instructions"""
    if not candidates:
        return ". Given my reasoning above, I can now provide the direct answer.</think>\n\\boxed"

    elif len(candidates) == 1:
        return f". Candidate: {candidates[0]}. My analysis leads me to either confirm this or give my own stronger answer:</think>\n\\boxed"

    else:
        candidates_list = ", ".join([f"{i+1}. {cand}" for i, cand in enumerate(candidates)])
        return f". Candidates: {candidates_list}. I'll select from these OR give my own answer if my reasoning strongly supports a different conclusion:</think>\n\\boxed"

@cache
def get_vllm():
    """Initialize and return vLLM engine (singleton)"""
    global ENGINE
    if ENGINE is None:
        ENGINE = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=8,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
    return ENGINE

@cache
def get_tokenizer():
    """Initialize and return tokenizer (singleton)"""
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return TOKENIZER


def load_data(input_path):
    """Load offline DeepConf data from pickle file"""
    print(f"Loading data from {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded question {data.get('question_id')}, {len(data.get('all_traces', []))} traces")
    return data


def extract_candidates_from_traces(traces, min_count=0, prob_token=None):
    """
    Extract all candidate answers from traces, considering token budget.
    
    Args:
        traces: List of trace dictionaries with 'extracted_answer' field
        min_count: Minimum occurrences for a candidate to be included
        prob_token: Token budget limit - only count candidates from traces with enough tokens
        
    Returns:
        List of all candidate answers sorted by frequency
    """
    answer_counts = Counter()
    
    for trace in traces:
        answer = trace.get('extracted_answer')
        if not answer:
            continue
        
        # If prob_token is specified, only count answers from traces with enough tokens
        if prob_token is not None:
            num_tokens = trace.get('num_tokens', 0)
            # Skip traces that don't have enough tokens to generate the answer
            if num_tokens > prob_token:
                continue
        
        # Count this answer
        answer_counts[answer] += 1
    
    # Get all candidates with at least min_count occurrences, sorted by frequency
    candidates = [
        answer for answer, count in answer_counts.most_common()
        if count >= min_count
    ]
    
    return candidates


def prepare_prompt_with_candidates(question, thinking_trace, candidates, prob_token=None):
    """
    Prepare prompt with truncated thinking and candidate injection.
    
    Args:
        question: The question text
        thinking_trace: The thinking trace text to truncate
        candidates: List of candidate answers
        prob_token: Token limit for the thinking trace
        
    Returns:
        Tuple of (full_prompt, token_usage)
    """
    tokenizer = get_tokenizer()
    
    # Format base prompt
    messages = [
        {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
        {"role": "user", "content": question},
    ]
    
    base_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # Add thinking trace
    full_prompt = base_prompt + thinking_trace
    
    # Truncate to token limit if specified
    token_usage = 0
    if prob_token:
        tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
        base_tokens = tokenizer.encode(base_prompt, add_special_tokens=False)
        
        if len(tokens) > len(base_tokens) + prob_token:
            tokens = tokens[:len(base_tokens) + prob_token]
            full_prompt = tokenizer.decode(tokens)
        token_usage = len(tokens)
    
    # Add candidate injection
    inject_prompt = get_inject_prompt_v2(candidates)
    full_prompt += inject_prompt
    
    return full_prompt, token_usage


def prepare_batch_prompts(question, traces, candidates, prob_token=None):
    """
    Prepare batch of prompts for all traces.
    
    Args:
        question: The question text
        traces: List of trace dictionaries
        candidates: List of candidate answers to inject
        prob_token: Token limit for thinking traces
        
    Returns:
        Tuple of (batch_prompts, token_usages)
    """
    batch_prompts = []
    token_usages = []
    
    for trace in traces:
        # Skip prompting if trace stopped due to conf bar and our probe token limit is greater than trace length
        if trace.get('stop_reason') == 'gconf_threshold' and trace.get('num_tokens', 0) < prob_token:
            # Still include in token usage but not in prompts
            token_usages.append(trace.get('num_tokens', 0))
            continue

        prompt, token_usage = prepare_prompt_with_candidates(
            question,
            trace['text'],
            candidates,
            prob_token
        )
        batch_prompts.append(prompt)
        token_usages.append(token_usage)
    
    return batch_prompts, token_usages


def probe_with_vllm(prompts):
    """
    Run vLLM inference on batch of prompts.
    
    Args:
        prompts: List of prompt strings
        
    Returns:
        List of extracted answers
    """
    engine = get_vllm()
    
    sampling_params = SamplingParams(
        n=1,
        temperature=0.6,
        top_p=0.95,
        max_tokens=200,
        logprobs=20,
    )
    
    print(f"Running inference on {len(prompts)} prompts...")
    outputs = engine.generate(prompts, sampling_params)
    
    # Extract answers
    answers = []
    for output in outputs:
        text = output.outputs[0].text
        # Extract answer after \boxed
        answer = extract_answer("\\boxed" + text)
        answers.append(answer)
    
    return answers


def aggregate_answers(answers, ground_truth=None):
    """
    Aggregate answers and compute statistics.
    
    Args:
        answers: List of answer strings
        ground_truth: Ground truth answer (optional)
        
    Returns:
        Dictionary with aggregation results
    """
    # Count occurrences
    answer_counts = Counter(answers)
    
    # Calculate statistics
    total = len(answers)
    most_common = answer_counts.most_common(1)[0] if answer_counts else (None, 0)
    
    # Calculate entropy
    if answer_counts:
        probs = [count / total for count in answer_counts.values()]
        entropy = -sum(p * np.log(p) for p in probs if p > 0)
        confidence = 1 - (entropy / np.log(len(answer_counts))) if len(answer_counts) > 1 else 1.0
    else:
        entropy = 0
        confidence = 0
    
    # Check correctness if ground truth provided
    is_correct = False
    accuracy = 0.0
    if ground_truth and most_common[0]:
        try:
            is_correct = equal_func(most_common[0], ground_truth)
        except:
            is_correct = str(most_common[0]) == str(ground_truth)
        
        # Calculate accuracy (proportion of correct answers)
        correct_count = sum(
            count for answer, count in answer_counts.items()
            if str(answer) == str(ground_truth)
        )
        accuracy = correct_count / total if total > 0 else 0
    
    return {
        "answer_distribution": dict(answer_counts.most_common()),
        "majority_answer": most_common[0],
        "majority_count": most_common[1],
        "majority_ratio": most_common[1] / total if total > 0 else 0,
        "total_answers": total,
        "unique_answers": len(answer_counts),
        "entropy": entropy,
        "confidence": confidence,
        "is_correct": is_correct,
        "accuracy": accuracy,
    }


def process_data(data, prob_token=None, output_path=None):
    """
    Main processing function.
    
    Args:
        data: Loaded data dictionary
        prob_token: Token limit for thinking traces
        output_path: Path to save results (optional)
        
    Returns:
        Results dictionary
    """
    question = data['question']
    ground_truth = data.get('ground_truth')
    traces = data['final_traces']
    
    print(f"\nProcessing question: {question[:100]}...")
    print(f"Ground truth: {ground_truth}")
    print(f"Number of traces: {len(traces)}")
    
    # Step 1: Extract all candidates from traces (considering token budget)
    print(f"\nExtracting all candidates from traces (token budget: {prob_token})...")
    candidates = extract_candidates_from_traces(traces, prob_token=prob_token)
    print(f"Found {len(candidates)} candidates within budget: {candidates}")
    
    # Step 2: Prepare prompts with candidates
    print(f"\nPreparing prompts with candidates (prob_token={prob_token})...")
    batch_prompts, token_usages = prepare_batch_prompts(
        question, traces, candidates, prob_token
    )
    print(batch_prompts[0][:500] + "\n...")
    print(f"Prepared {len(batch_prompts)} prompts")
    
    # Step 3: Run inference
    answers = probe_with_vllm(batch_prompts)
    print(f"\nGenerated {len(answers)} answers")
    
    # Step 4: Aggregate results
    print("\nAggregating results...")
    aggregation = aggregate_answers(answers, ground_truth)
    
    # Prepare final results
    results = {
        "question_id": data.get('question_id'),
        "question": question,
        "ground_truth": ground_truth,
        "candidates": candidates,
        "num_candidates": len(candidates),
        "prob_token": prob_token,
        "aggregation": aggregation,
        "raw_answers": answers,
    }
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Candidates used: {candidates}")
    print(f"Majority answer: {aggregation['majority_answer']}")
    print(f"Majority ratio: {aggregation['majority_ratio']:.2%}")
    print(f"Confidence: {aggregation['confidence']:.3f}")
    print(f"Unique answers: {aggregation['unique_answers']}")
    print(f"Is correct: {aggregation['is_correct']}")
    print(f"Accuracy: {aggregation['accuracy']:.2%}")
    print("=" * 70)
    
    return results


def main():
    # parser = argparse.ArgumentParser(description='Dynamic Programming Probe for DeepConf')
    # parser.add_argument('--input', type=str, required=True,
    #                     help='Input pickle file path')
    # parser.add_argument('--output', type=str,
    #                     help='Output JSON file path (optional)')
    # parser.add_argument('--prob_token', type=int, default=64000,
    #                     help='Token limit for thinking traces (default: 64000)')
    
    # args = parser.parse_args()
    # Process all data files in the outputs-online folder
    input_dir = "/home/ec2-user/projects/deepconf/outputs-online"
    output_dir = "/home/ec2-user/projects/deepconf/dp_probe_results"
    os.makedirs(output_dir, exist_ok=True)
    prob_token = 64000

    # Find all pkl files in subdirectories
    pkl_files = glob.glob(os.path.join(input_dir, "*/*.pkl"))
    print(f"Found {len(pkl_files)} pkl files to process")

    all_results = []

    for pkl_file in pkl_files:
        try:
            print(f"\n{'='*80}")
            print(f"Processing: {pkl_file}")
            print(f"{'='*80}")

            # Extract question ID from path for output naming
            qid = os.path.basename(os.path.dirname(pkl_file))
            # Extract original filename without extension for output naming
            original_filename = os.path.splitext(os.path.basename(pkl_file))[0]
            output_file = os.path.join(output_dir, f"{original_filename}_dp_probe.json")

            # Load and process data
            data = load_data(pkl_file)
            results = process_data(
                data,
                prob_token=prob_token,
                output_path=output_file
            )
            all_results.append(results)

        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
            continue

    # Save summary results
    summary_file = os.path.join(output_dir, "summary_results.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nProcessed {len(all_results)} files successfully")
    print(f"Summary saved to {summary_file}")
    return results


if __name__ == "__main__":
    main()
