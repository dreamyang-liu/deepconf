"""Auto-run prefill for all questions. Retries with smaller chunks on OOM."""
import argparse, glob, os, pickle, subprocess, time

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


def rp(ds, df, qid, cs):
    subprocess.run([
        "conda", "run", "-n", "deepconf", "python",
        os.path.join(SD, "prefill-confs-vllm.py"),
        "--input-dir", os.path.join(SD, f"outputs-bedrock/{ds}"),
        "--dataset-file", os.path.join(SD, df),
        "--model-path", "Qwen/Qwen3-32B", "--tp", "2",
        "--chunk-size", str(cs), "--max-model-len", "40960",
        "--qids", str(qid),
        "--output-dir", os.path.join(SD, f"outputs-bedrock-confs/{ds}"),
    ], cwd=SD)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=list(DM.keys()))
    p.add_argument("--start-qid", type=int, default=0)
    a = p.parse_args()
    ds, df = a.dataset, DM[a.dataset]
    idir = os.path.join(SD, f"outputs-bedrock/{ds}")
    odir = os.path.join(SD, f"outputs-bedrock-confs/{ds}")
    with open(os.path.join(SD, df)) as f:
        nq = sum(1 for _ in f)
    print(f"=== Prefill: {ds} ({nq} qs) ===")
    t0 = time.time()
    for qid in range(a.start_qid, nq):
        print(f"\n=== QID {qid}/{nq-1} ===")
        pkls = glob.glob(os.path.join(idir, f"deepconf_simple_qid{qid}_rid*.pkl"))
        if not pkls:
            print("  skip")
            continue
        ip = pkls[0]
        bn = os.path.basename(ip)
        op = os.path.join(odir, bn)
        if os.path.exists(op):
            print("  done")
            continue
        qt = time.time()
        ok = False
        for cs in CSS:
            print(f"  chunk={cs}")
            rp(ds, df, qid, cs)
            if os.path.exists(op):
                print(f"  OK ({time.time()-qt:.0f}s)")
                ok = True
                break
            cd = os.path.join(odir, f"ckpt_qid{qid}")
            nc = len(glob.glob(os.path.join(cd, "trace_*.pkl")))
            print(f"  ckpts: {nc}")
        if not ok:
            cd = os.path.join(odir, f"ckpt_qid{qid}")
            nc = len(glob.glob(os.path.join(cd, "trace_*.pkl")))
            if nc > 0:
                print(f"  Assembling {nc} ckpts...")
                asm(ip, cd, op)
            else:
                print("  FAILED")
    nd = len(glob.glob(os.path.join(odir, "deepconf_simple_*.pkl")))
    print(f"\n=== DONE: {ds} === {nd} files, {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    main()
