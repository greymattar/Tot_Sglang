
# scripts/run_tierA_smoke.py
import os, json, time, argparse
import yaml
import torch

from tot_harness.token_accounting import TokenCounter
from tot_harness.controller import run_tot_one_problem
from tot_harness.scorer_prm import PRMScorer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "sglang", "vllm"], default="hf")
    ap.add_argument("--sgl_url", default="http://127.0.0.1:30000")
    ap.add_argument("--vllm_url", default="http://127.0.0.1:18000")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    EXP_ROOT = os.environ["EXP_ROOT"]
    cfg = yaml.safe_load(open(os.path.join(EXP_ROOT, "configs/contract/dpts_match.yaml")))

    slice_path = os.path.join(EXP_ROOT, cfg["dataset"]["slice_path"])
    out_dir = os.path.join(EXP_ROOT, "runlogs")
    os.makedirs(out_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = args.tag or args.backend
    log_path = os.path.join(out_dir, f"tierA_{tag}_prm_smoke_{ts}.jsonl")

    # Backend + token counter
    if args.backend == "hf":
        #  existing HF backend 
        from tot_harness.backend_hf import HFBackend
        backend = HFBackend(cfg["generator"]["model_id"],
                            dtype=cfg["generator"]["dtype"],
                            device="cuda:0")
    elif args.backend == "sglang":
        # SGLang backend (HTTP client)
        from src.backends.sglang_backend import SGLangBackend
        backend = SGLangBackend(args.sgl_url)
    else:
        from tot_harness.backend_vllm import VLLMBackend
        backend = VLLMBackend(base_url=args.vllm_url, model=cfg["generator"]["model_id"])

    tc = TokenCounter(cfg["generator"]["model_id"])

    # PRM scorer: LOAD ONCE
    prm_cfg = cfg["prm"]
    scorer = PRMScorer(
        prm_cfg["model_id"],
        device="cuda",
        prm_dtype=prm_cfg["prm_dtype"],
        step_tag=prm_cfg["step_tag"],
        good_token=prm_cfg["good_token"],
        bad_token=prm_cfg["bad_token"],
        aggregation=prm_cfg["aggregation"],
    )

    summaries = []
    with open(log_path, "w") as log_f:
        with open(slice_path) as f:
            for i, line in enumerate(f):
                if i >= args.n:
                    break
                ex = json.loads(line)

                torch.cuda.reset_peak_memory_stats()
                s = run_tot_one_problem(
                    backend, tc, scorer,
                    ex["id"], ex["question"], ex["answer"],
                    cfg, log_f
                )
                s["peak_gpu_mem_bytes"] = int(torch.cuda.max_memory_allocated())
                summaries.append(s)

    print("Wrote:", log_path)
    print("Summaries:", summaries)

if __name__ == "__main__":
    main()




