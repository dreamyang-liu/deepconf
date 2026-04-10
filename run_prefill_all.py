"""Auto-run prefill for all questions. Retries with smaller chunks on OOM.
Supports --parallel N to run N workers, each on a TP=2 GPU pair."""
import argparse, glob, os, pickle, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SD = os.path.dirname(os.path.abspath(__file__))
DM = {"aime24": "aime_2024.jsonl", "brumo25": "brumo_2025.jsonl", "hmmt": "hmmt_feb_2025.jsonl"}
CSS = [512, 64, 8, 1]


def asm(ip, cd, op):
    with open(ip, "rb") as f:
        data = pickle.load(f)
    tt = 0
    for cf in sorted(glob.glob(os.path.join(cd, "trace_*.pkl"))):
        i = int(os.path.basename(cf).replace("trace_", "").replace(".pkl", ""))
        with open(cf, "rb") as f:
            c = pickle.load(f)
        data["all_traces"][i]["confs"] = c["confs"]
        data["all_traces"][i]["num_tokens"] = c["num_tokens"]
        tt += c["num_tokens"]
    data["token_stats"]["total_tokens"] = tt
    data["token_stats"]["avg_tokens_per_trace"] = tt / max(len(data["all_traces"]), 1)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "wb") as f:
        pickle.dump(data, f)
    wc = sum(1 for t in data["all_traces"] if len(t.get("confs", [])) > 0)
    print(f"  Assembled: {wc}/{len(data['all_traces'])} -> {op}")


CONDA_PYTHON = "/opt/dlami/nvme/miniconda3/envs/deepconf/bin/python"


def rp(ds, df, qid, cs, gpus=None):
    env = os.environ.copy()
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    subprocess.run([
        CONDA_PYTHON,
        os.path.join(SD, "prefill-confs-vllm.py"),
        "--input-dir", os.path.join(SD, f"outputs-bedrock/{ds}"),
        "--dataset-file", os.path.join(SD, df),
        "--model-path", "Qwen/Qwen3-32B", "--tp", "2",
        "--chunk-size", str(cs), "--max-model-len", "40960",
        "--qids", str(qid),
        "--output-dir", os.path.join(SD, f"outputs-bedrock-confs/{ds}"),
    ], cwd=SD, env=env)


GPU_PAIRS = ["0,1", "2,3", "4,5", "6,7"]


def run_qid(ds, df, qid, gpus):
    """Run prefill for a single qid on a specific GPU pair. Returns (qid, status)."""
    idir = os.path.join(SD, f"outputs-bedrock/{ds}")
    odir = os.path.join(SD, f"outputs-bedrock-confs/{ds}")
    pkls = glob.glob(os.path.join(idir, f"deepconf_simple_qid{qid}_rid*.pkl"))
    if not pkls:
        return qid, "skip"
    ip = pkls[0]
    bn = os.path.basename(ip)
    op = os.path.join(odir, bn)
    if os.path.exists(op):
        return qid, "done"
    qt = time.time()
    for cs in CSS:
        print(f"  [GPU {gpus}] QID {qid} chunk={cs}")
        rp(ds, df, qid, cs, gpus=gpus)
        if os.path.exists(op):
            print(f"  [GPU {gpus}] QID {qid} OK ({time.time()-qt:.0f}s)")
            return qid, "ok"
        cd = os.path.join(odir, f"ckpt_qid{qid}")
        nc = len(glob.glob(os.path.join(cd, "trace_*.pkl")))
        print(f"  [GPU {gpus}] QID {qid} ckpts: {nc}")
    # Assemble from checkpoints
    cd = os.path.join(odir, f"ckpt_qid{qid}")
    nc = len(glob.glob(os.path.join(cd, "trace_*.pkl")))
    if nc > 0:
        print(f"  [GPU {gpus}] QID {qid} assembling {nc} ckpts...")
        asm(ip, cd, op)
        return qid, "assembled"
    return qid, "FAILED"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=list(DM.keys()))
    p.add_argument("--start-qid", type=int, default=0)
    p.add_argument("--parallel", type=int, default=1,
                   help="Number of parallel workers (max 4, each uses TP=2)")
    a = p.parse_args()
    ds, df = a.dataset, DM[a.dataset]
    idir = os.path.join(SD, f"outputs-bedrock/{ds}")
    odir = os.path.join(SD, f"outputs-bedrock-confs/{ds}")
    with open(os.path.join(SD, df)) as f:
        nq = sum(1 for _ in f)

    nw = min(a.parallel, len(GPU_PAIRS))
    print(f"=== Prefill: {ds} ({nq} qs), {nw} workers ===")
    t0 = time.time()

    if nw <= 1:
        # Sequential mode (original behavior)
        for qid in range(a.start_qid, nq):
            print(f"\n=== QID {qid}/{nq-1} ===")
            _, status = run_qid(ds, df, qid, GPU_PAIRS[0])
            print(f"  -> {status}")
    else:
        # Parallel mode
        qids = list(range(a.start_qid, nq))
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futures = {}
            gpu_avail = list(range(nw))
            qi = 0  # index into qids

            # Submit initial batch
            while gpu_avail and qi < len(qids):
                wi = gpu_avail.pop(0)
                f = ex.submit(run_qid, ds, df, qids[qi], GPU_PAIRS[wi])
                futures[f] = (qids[qi], wi)
                qi += 1

            # Process completions and submit next
            while futures:
                for f in as_completed(futures):
                    qid, wi = futures.pop(f)
                    _, status = f.result()
                    print(f"=== QID {qid} -> {status} ===")
                    if qi < len(qids):
                        nf = ex.submit(run_qid, ds, df, qids[qi], GPU_PAIRS[wi])
                        futures[nf] = (qids[qi], wi)
                        qi += 1
                    break  # re-check as_completed after submitting

    nd = len(glob.glob(os.path.join(odir, "deepconf_simple_*.pkl")))
    print(f"\n=== DONE: {ds} === {nd} files, {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    main()
