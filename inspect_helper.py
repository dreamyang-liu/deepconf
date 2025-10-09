from dynasor.core.evaluator import math_equal
import numpy as np
import copy

def extract_answer(text):
    """Extract boxed answer from text"""
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        return a.strip()
    return None

def quick_parse(text):
    if '\\text{' in text and '}' in text:
        # Find all occurrences of \text{...} and remove them
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            # Replace \text{content} with just content
            content = text[start + 6:end]  # 6 is length of '\text{'
            text = text[:start] + content + text[end + 1:]
    return text

def equal_func(answer, ground_truth):
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)

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


def locate_answer_conf(token_ids, confs, extracted_answer, text, tokenizer):
    """Locate the answer in the text and return the corresponding confidence scores"""
    # If no answer was extracted, return an empty list
    assert len(token_ids) == len(confs), "confs length should be equal to token ids"
    answer_tokens_with_conf = []
    if not extracted_answer:
        return [0.0], answer_tokens_with_conf

    # If no confidence scores available, return an empty list
    if not confs or len(confs) != len(token_ids):
        return [0.0], answer_tokens_with_conf

    # Find the extracted answer in the text
    # Start from the end since the answer is likely at the end of the text
    pos = text.rfind(extracted_answer)
    if pos == -1:
        # The extracted_answer comes directly from text, so we can find it directly
        pos = text.rfind(extracted_answer)
        if pos == -1:
            return [0.0], answer_tokens_with_conf

        start_pos = pos
        end_pos = pos + len(extracted_answer)
    else:
        start_pos = pos
        end_pos = pos + len(extracted_answer)

    # Count tokens up to the start and end positions
    start_token_idx = 0
    end_token_idx = len(token_ids)

    # Find the tokens that correspond to the extracted answer
    text_before_answer = text[:start_pos]
    tokens_before_answer = tokenizer.tokenize(text_before_answer)
    start_token_idx = len(tokens_before_answer)

    text_with_answer = text[:end_pos]
    tokens_with_answer = tokenizer.tokenize(text_with_answer)
    end_token_idx = len(tokens_with_answer)

    # [Debug] Decode the tokens to map them to the answer
    # Get the actual token IDs from the model output
    answer_token_ids = token_ids[start_token_idx:end_token_idx]

    # Decode the answer tokens to verify they match the extracted_answer
    decoded_answer = [tokenizer.decode(token_id) for token_id in answer_token_ids]
    # Save the decoded token and confidence to a file
    # Create a list of dictionaries with token and confidence info
    
    try:
        for i, (token, conf) in enumerate(zip(decoded_answer, confs[start_token_idx:end_token_idx])):
            answer_tokens_with_conf.append({
                "index": i,
                "token": token,
                "confidence": conf
            })
    except Exception as e:
        print(f"Error creating token confidence data: {str(e)}")

    # Return the confidence scores for the answer tokens
    return confs[start_token_idx:end_token_idx], answer_tokens_with_conf

def process_output(output, ground_truth, window_size, append_prefix='', tokenizer=None):
    """Process a single vLLM output"""
    text = output.text
    token_ids = output.token_ids
    logprobs = output.logprobs
    
    # Calculate confidence
    confs = compute_confidence(logprobs) if logprobs else []
    sliding_window = compute_least_grouped(confs, group_size=window_size) if confs else [0]
    
    extracted_answer = extract_answer(append_prefix + text)
    
    is_correct = False
    if extracted_answer and ground_truth:
        try:
            is_correct = equal_func(extracted_answer, ground_truth)
        except:
            is_correct = str(extracted_answer) == str(ground_truth)
    
    answer_conf, answer_token_conf = locate_answer_conf(token_ids, confs, extracted_answer, text, tokenizer)

    try:
        avg_conf = np.mean(answer_conf)
        max_conf = np.max(answer_conf)
        min_conf = np.min(answer_conf)
    except Exception:
        avg_conf = 0
        max_conf = 0
        min_conf = 0

    return {
        "stop_reason": output.finish_reason,
        "text": text,
        "token_ids": token_ids,
        "num_tokens": len(token_ids) if token_ids else 0,
        "confs": confs,
        "avg_conf": avg_conf,
        "max_conf": max_conf,
        "min_conf": min_conf,
        "group_confs": sliding_window,
        "min_conf": min(sliding_window) if sliding_window else 0,
        "extracted_answer": extracted_answer,
        "answer_token_conf": answer_token_conf,
        "is_correct": is_correct,
    }

def recompute_traces(traces, tokenizer):
    new_traces = []
    for trace in traces:
        new_trace = copy.deepcopy(trace)
        text = new_trace['text']
        token_ids = new_trace['token_ids']
        confs = new_trace['confs']
        extracted_answer = new_trace['extracted_answer']
        answer_conf, answer_token_conf = locate_answer_conf(token_ids, confs, extracted_answer, text, tokenizer)
        try:
            new_trace['avg_conf'] = np.mean(answer_conf)
            new_trace['max_conf'] = np.max(answer_conf)
            new_trace['min_conf'] = np.min(answer_conf)
        except Exception:
            new_trace['avg_conf'] = 0
            new_trace['max_conf'] = 0
            new_trace['min_conf'] = 0
        new_trace["answer_token_conf"] = answer_token_conf
        new_traces.append(new_trace)
    return new_traces



def process_batch_results(batch_outputs, ground_truth, window_size, prefix='', tokenizer=None):
    """Process batch results from vLLM for a single question"""
    for i in range(len(batch_outputs)):
        question_outputs = batch_outputs[i].outputs
        
        # Process all traces for this question
        traces = []
        min_confs = []
        total_tokens = 0
        
        for output in question_outputs:
            trace_data = process_output(output, ground_truth, window_size, prefix, tokenizer)
            traces.append(trace_data)
            min_confs.append(trace_data["min_conf"])
            total_tokens += trace_data["num_tokens"]
        
        yield {
            'traces': traces,
            'min_confs': min_confs,
            'ground_truth': ground_truth,
            'total_tokens': total_tokens,
            'num_traces': len(traces)
        }


def process_batch_results_original(batch_outputs, ground_truth, window_size, tokenizer=None):
    """Process batch results from vLLM for a single question"""
    question_outputs = batch_outputs[0].outputs
    
    # Process all traces for this question
    traces = []
    min_confs = []
    total_tokens = 0
    
    for output in question_outputs:
        trace_data = process_output(output, ground_truth, window_size, tokenizer=tokenizer)
        traces.append(trace_data)
        min_confs.append(trace_data["min_conf"])
        total_tokens += trace_data["num_tokens"]
    
    return {
        'traces': traces,
        'min_confs': min_confs,
        'ground_truth': ground_truth,
        'total_tokens': total_tokens,
        'num_traces': len(traces)
    }

def weighted_majority_vote(answers, weights):
    """Perform weighted majority voting"""
    if not answers:
        return None
    
    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
    
    if not answer_weights:
        return None
    
    return max(answer_weights.keys(), key=lambda x: answer_weights[x]), answer_weights


def compute_voting_answer_for_completed_traces(completed_traces_results, indicator='avg_conf'):
    """Analyze processed results from DeepConf experiments"""
    # Aggregate results across different traces
    answers = []
    weights = []
    for trace in completed_traces_results:
        answers.append(trace['extracted_answer'])
        weights.append(trace[indicator])
    
    return answers, weights

def compute_voting_answer_for_truncated_traces(truncated_traces_results, indicator='avg_conf'):
    """Analyze processed results from DeepConf experiments"""
    # Aggregate results across different traces
    answers = []
    weights = []
    for result in truncated_traces_results:
        answers.append(result['traces'][0]['extracted_answer'])
        weights.append(result['traces'][0][indicator])
    
    return answers, weights

def compute_voting_answer(completed_traces_results, truncated_traces_results):
    structured_voting_answer = {} 
    for indicator in ["min_conf", "max_conf", "avg_conf"]:
        completed_traces_answers, completed_traces_weights = compute_voting_answer_for_completed_traces(completed_traces_results, indicator)
        truncated_traces_answers, truncated_traces_weights = compute_voting_answer_for_truncated_traces(truncated_traces_results, indicator)
        combined_traces_answers = completed_traces_answers + truncated_traces_answers
        combined_traces_weights = completed_traces_weights + truncated_traces_weights

        completed_traces_voting_answer = weighted_majority_vote(completed_traces_answers, completed_traces_weights)
        truncated_traces_voting_answer = weighted_majority_vote(truncated_traces_answers, truncated_traces_weights)
        combined_traces_voting_answer = weighted_majority_vote(combined_traces_answers, combined_traces_weights)

        structured_voting_answer[indicator] = {
            "completed_traces": completed_traces_voting_answer,
            "truncated_traces": truncated_traces_voting_answer,
            "combined_traces": combined_traces_voting_answer
        }

    return structured_voting_answer


def extract_structured_conf(completed_traces_results, truncated_traces_results):

    structured_conf = {}
    for indicator in ["min_conf", "max_conf", "avg_conf"]:
        completed_traces_answers, completed_traces_weights = compute_voting_answer_for_completed_traces(completed_traces_results, indicator)
        truncated_traces_answers, truncated_traces_weights = compute_voting_answer_for_truncated_traces(truncated_traces_results, indicator)
        combined_traces_answers = completed_traces_answers + truncated_traces_answers
        combined_traces_weights = completed_traces_weights + truncated_traces_weights

        structured_conf[indicator] = {
            "completed_traces": np.mean(completed_traces_weights),
            "truncated_traces": np.mean(truncated_traces_weights),
            "combined_traces": np.mean(combined_traces_weights)
        }

    return structured_conf

def compute_instance_accuracy(completed_traces_results, truncated_traces_results, ground_truth):
    completed_correctness = [trace['is_correct'] for trace in completed_traces_results]
    truncated_correctness = [trace['traces'][0]['is_correct'] for trace in truncated_traces_results]
    return {
        "completed_traces": np.mean(completed_correctness),
        "truncated_traces": np.mean(truncated_correctness),
        "combined_traces": np.mean(completed_correctness + truncated_correctness)
    }