"""Auto-run prefill for all questions. Retries with smaller chunks on OOM.
Supports --parallel N to run N workers, each on a TP=2 GPU pair."""
import argparse, glob, os, pickle, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SD = os.path.dirname(os.path.abspath(__file__))
DM = {"aime24": "aime_2024.jsonl", "brumo25": "brumo_2025.jsonl", "hmmt": "hmmt_feb_2025.jsonl"}
# Default OOM-retry chunk size sequence. Override with --chunk-sizes.
CSS_DEFAULT = [512, 64, 8, 1]


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


# Use the interpreter that launched this script, so subprocess workers
# inherit the same Python env (handy when running inside docker).
CONDA_PYTHON = sys.executable


def rp(ds, df, qid, cs, cfg, gpus=None):
    env = os.environ.copy()
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    subprocess.run([
        CONDA_PYTHON,
        os.path.join(SD, "prefill-confs-vllm.py"),
        "--input-dir", os.path.join(SD, cfg["input_dir_root"], ds),
        "--dataset-file", os.path.join(SD, df),
        "--model-path", cfg["model_path"], "--tp", str(cfg["tp"]),
        "--chunk-size", str(cs), "--max-model-len", str(cfg["max_model_len"]),
        "--qids", str(qid),
        "--output-dir", os.path.join(SD, cfg["output_dir_root"], ds),
    ], cwd=SD, env=env)


def compute_gpu_assignments(tp, total_gpus=8):
    """Build CUDA_VISIBLE_DEVICES strings so (total_gpus // tp) workers each own tp GPUs.
    TP=1 → ['0','1',...,'7'] (8 workers)
    TP=2 → ['0,1','2,3','4,5','6,7'] (4 workers)
    TP=4 → ['0,1,2,3','4,5,6,7'] (2 workers)
    """
    nw = total_gpus // tp
    return [",".join(str(i * tp + j) for j in range(tp)) for i in range(nw)]


def run_qid(ds, df, qid, gpus, cfg):
    """Run prefill for a single qid on a specific GPU assignment. Returns (qid, status)."""
    idir = os.path.join(SD, cfg["input_dir_root"], ds)
    odir = os.path.join(SD, cfg["output_dir_root"], ds)
    pkls = glob.glob(os.path.join(idir, f"deepconf_simple_qid{qid}_rid*.pkl"))
    if not pkls:
        return qid, "skip"
    ip = pkls[0]
    bn = os.path.basename(ip)
    op = os.path.join(odir, bn)
    if os.path.exists(op):
        return qid, "done"
    qt = time.time()
    for cs in cfg["chunk_sizes"]:
        print(f"  [GPU {gpus}] QID {qid} chunk={cs}")
        rp(ds, df, qid, cs, cfg, gpus=gpus)
        if os.path.exists(op):
            print(f"  [GPU {gpus}] QID {qid} OK ({time.time()-qt:.0f}s)")
            return qid, "ok"
        cd = os.path.join(odir, f"ckpt_qid{qid}")
        nc = len(glob.glob(os.path.join(cd, "trace_*.pkl")))
        print(f"  [GPU {gpus}] QID {qid} ckpts: {nc}")
    # Assemble only if every trace has a ckpt (otherwise next run would
    # see an existing output pkl and skip the qid with partial confs).
    cd = os.path.join(odir, f"ckpt_qid{qid}")
    nc = len(glob.glob(os.path.join(cd, "trace_*.pkl")))
    with open(ip, "rb") as _f:
        n_expected = len(pickle.load(_f)["all_traces"])
    if nc >= n_expected:
        print(f"  [GPU {gpus}] QID {qid} assembling {nc}/{n_expected} ckpts...")
        asm(ip, cd, op)
        return qid, "assembled"
    print(f"  [GPU {gpus}] QID {qid} FAILED with {nc}/{n_expected} ckpts (not assembling; will retry)")
    return qid, "FAILED"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=list(DM.keys()))
    p.add_argument("--start-qid", type=int, default=0)
    p.add_argument("--parallel", type=int, default=None,
                   help="Number of parallel workers (defaults to 8//tp; capped there).")
    p.add_argument("--model-path", type=str, default="Qwen/Qwen3-32B")
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--max-model-len", type=int, default=40960)
    p.add_argument("--input-dir-root", type=str, default="outputs-bedrock",
                   help="Root dir; looks for <root>/<dataset>/deepconf_simple_qid*.pkl")
    p.add_argument("--output-dir-root", type=str, default="outputs-bedrock-confs")
    p.add_argument("--chunk-sizes", type=int, nargs="+", default=CSS_DEFAULT,
                   help="Chunk-size sequence for OOM retry (largest first).")
    a = p.parse_args()
    ds, df = a.dataset, DM[a.dataset]
    cfg = {
        "model_path": a.model_path,
        "tp": a.tp,
        "max_model_len": a.max_model_len,
        "input_dir_root": a.input_dir_root,
        "output_dir_root": a.output_dir_root,
        "chunk_sizes": a.chunk_sizes,
    }
    idir = os.path.join(SD, a.input_dir_root, ds)
    odir = os.path.join(SD, a.output_dir_root, ds)
    with open(os.path.join(SD, df)) as f:
        nq = sum(1 for _ in f)

    gpu_assignments = compute_gpu_assignments(a.tp)
    max_workers = len(gpu_assignments)
    nw = min(a.parallel, max_workers) if a.parallel else max_workers
    print(f"=== Prefill: {ds} ({nq} qs), {nw} workers, model={a.model_path}, tp={a.tp}, "
          f"chunks={a.chunk_sizes} ===")
    print(f"    GPU assignments: {gpu_assignments[:nw]}")
    print(f"    input:  {idir}")
    print(f"    output: {odir}")
    t0 = time.time()

    if nw <= 1:
        # Sequential mode (original behavior)
        for qid in range(a.start_qid, nq):
            print(f"\n=== QID {qid}/{nq-1} ===")
            _, status = run_qid(ds, df, qid, gpu_assignments[0], cfg)
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
                f = ex.submit(run_qid, ds, df, qids[qi], gpu_assignments[wi], cfg)
                futures[f] = (qids[qi], wi)
                qi += 1

            # Process completions and submit next
            while futures:
                for f in as_completed(futures):
                    qid, wi = futures.pop(f)
                    _, status = f.result()
                    print(f"=== QID {qid} -> {status} ===")
                    if qi < len(qids):
                        nf = ex.submit(run_qid, ds, df, qids[qi], gpu_assignments[wi], cfg)
                        futures[nf] = (qids[qi], wi)
                        qi += 1
                    break  # re-check as_completed after submitting

    nd = len(glob.glob(os.path.join(odir, "deepconf_simple_*.pkl")))
    print(f"\n=== DONE: {ds} === {nd} files, {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    main()
