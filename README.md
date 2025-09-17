# Offline Results Reproduce

See README-offline.md

# Online Results Reproduce

See README-online.md

# Dataset prepare

Taking aime25 as an example. All needed dataset can be found under MathArena (https://huggingface.co/MathArena)

```
import json
from datasets import load_dataset

# Load dataset
dataset = load_dataset("MathArena/aime_2025", split="train")

# Convert to JSONL
with open("aime_2025.jsonl", "w", encoding="utf-8") as f:
    for example in dataset:
        entry = {
            "question": example["problem"],
            "answer": str(example["answer"])
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Converted {len(dataset)} examples to aime_2025.jsonl")
```

# License
DeepConf is MIT licensed, as found in the LICENSE file.


# Reuse Dropped Traces

Result on question 0 in brumo_2025

```
INFO 09-17 07:54:38 [__init__.py:241] Automatically detected platform cuda.
Tokenizer initialized in 0.56 seconds
Fields in the pickle file:
dict_keys(['stop_reason', 'text', 'token_ids', 'num_tokens', 'group_confs', 'min_conf', 'extracted_answer', 'is_correct'])
Loading cached outputs from final_outputs_cache.pkl
/opt/dlami/nvme/projects/deepconf/deepconf/inspect_result.py:133: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  plt.figure(figsize=(12, 6))
Revivied Trace Correctness: 160/169
```

Conf change can be found under fig directory

Step to reproduce 

1. Follow the README-offline to set up environment and run the command to collect traces

Remember to modify the `logprobs.py` to disable conf stop

```
def check_conf_stop(self) -> bool:
        """Return True if the confidence window triggers early stopping."""
        if self.conf_threshold is None:
            return False
        if self.conf_group_list is None or len(self.conf_group_list) == 0:
            return False
        return False
```


```
python deepconf-baseline.py --qid 0 --rid 0
```

2. Run inspect_result.py

Change the Tensor Parall size to GPU number you have and update file path as necessary.

```
python inspect_result.py 
```