import os, json, time
import yaml
import torch

from tot_harness.backend_hf import HFBackend
from tot_harness.token_accounting import TokenCounter
from tot_harness.controller import run_tot_one_problem
from tot_harness.scorer_prm import PRMScorer

EXP_ROOT = os.environ["EXP_ROOT"]
cfg = yaml.safe_load(open(os.path.join(EXP_ROOT, "configs/contract/dpts_match.yaml")))

slice_path = os.path.join(EXP_ROOT, cfg["dataset"]["slice_path"])
out_dir = os.path.join(EXP_ROOT, "runlogs")
os.makedirs(out_dir, exist_ok=True)

ts = time.strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(out_dir, f"tierA_hf_prm_smoke_{ts}.jsonl")

# Backend + token counter
backend = HFBackend(cfg["generator"]["model_id"], dtype=cfg["generator"]["dtype"], device="cuda:0")
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
            if i >= 3:
                break
            ex = json.loads(line)

            torch.cuda.reset_peak_memory_stats()
            s = run_tot_one_problem(
                backend, tc, scorer,
                ex["id"], ex["question"],ex["answer"],
                cfg, log_f
            )
            s["peak_gpu_mem_bytes"] = int(torch.cuda.max_memory_allocated())
            summaries.append(s)

print("Wrote:", log_path)
print("Summaries:", summaries)
