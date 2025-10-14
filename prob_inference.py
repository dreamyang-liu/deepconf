import os
from probe_utils import prepare_batch_messages,  probe_answers

def warmup_from_file(path, warmup_budget, percentile):
    pass


def process(output, qid, prob_token, window_size):
    if os.path.exists(f"./prob_analysis/analysis_results_{qid}_probtoken_{prob_token}_windowsize_{window_size}.json"):
        print(f"Alreay processed, skipping for {qid} {prob_token} {window_size} ...")
        return
    if prob_token > max(trace['num_tokens'] for trace in output['all_traces']):
        print(f"Reached max token, skipping for {qid} {prob_token} {window_size} ...")
        return 

    question = output['question']
    traces = output['all_traces']
    batch_messages = prepare_batch_messages(question, traces, prob_token)
    batch_confs = [trace['confs'] for trace in traces]

    # Calculate confidence metrics for each trace
    conf_stats = []

    for i, confs in enumerate(batch_confs):
        if len(confs) >= window_size:
            conf_stats.append({
                "min_window_conf": conf_metric_min_window_conf(confs, window_size),
                "max_window_conf": conf_metric_max_window_conf(confs, window_size),
                "mean_window_conf": conf_metric_mean_window_conf(confs, window_size),
                "last_window_conf": conf_metric_last_window_conf(confs, window_size)
            })
        else:
            # For traces with fewer tokens than window_size
            conf_stats.append({
                "min_window_conf": np.mean(confs) if confs else 0,
                "max_window_conf": np.mean(confs) if confs else 0,
                "mean_window_conf": np.mean(confs) if confs else 0,
                "last_window_conf": np.mean(confs) if confs else 0
            })
    answers = probe_answers(batch_messages)

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