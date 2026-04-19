
import argparse

import yaml

import json

import os

import torch

import time

from transformers import AutoTokenizer
from src.prm.genprm_verifier import GenPRMVerifier

#from tot_harness.backend_vllm import VLLMBackend

from tot_harness.backend_vllm import VLLMBackend as VLLMBackend
from tot_harness.batched_controller import run_batched_tot
from tot_harness.voting import ALL_METHODS
from tot_harness.scorer_prm import PRMScorer 


# A simple wrapper if you don't have a specific TokenCounter class

class SimpleTokenCounter:

    def __init__(self, model_name):

        try:

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        except:

            print("Warning: Could not load specific tokenizer. using 'gpt2' as fallback.")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")


    def count_generated_tokens_delta(self, prompt, completion):

        return len(self.tokenizer.encode(completion, add_special_tokens=False))

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, add_special_tokens=False)) 


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--vllm_url", default="http://127.0.0.1:18000")

    parser.add_argument("--batch_problems", type=int, default=10, help="Concurrent problems")

    parser.add_argument("--batch_width", type=int, default=4, help="Nodes per problem per step")

    parser.add_argument("--metrics_url", default=None, help="Prometheus /metrics endpoint (e.g. http://127.0.0.1:18000/metrics). ""If not set, will use --vllm_url + '/metrics'.")
    parser.add_argument("--budget_new_tokens", type=int, default=12000,
                    help="Override dpts_config.max_new_tokens for budget sweeps")

    parser.add_argument("--disable_metrics", action="store_true",help="Disable vLLM /metrics polling (avoid 503 during warmup)")
    parser.add_argument("--embed_url", default=None)
    parser.add_argument("--embed_model", default="BAAI/bge-small-en-v1.5")

    args = parser.parse_args()


    # Load Config

    EXP_ROOT = os.environ.get("EXP_ROOT", ".")

    config_path = os.path.join(EXP_ROOT, "configs/contract/dpts_match.yaml")

    

    # Load raw yaml

    raw_cfg = yaml.safe_load(open(config_path))

    




    cfg = {

        "dpts_config": {

            "tree_width": args.batch_width,

            "tree_depth": raw_cfg.get("max_depth", 16),

            "max_rollout": 20, # Safety limit

            "max_step_time": 480,

            "max_new_tokens": int(args.budget_new_tokens) if args.budget_new_tokens is not None else 12000,

            "voting_method": "all",

            "num_branch": raw_cfg.get("sampling", {}).get("n", 4),
            "depth_bonus_mode": "until_first_candidate", "alpha_depth": 0.8, 
            "priority_mode": "prm_nll_hybrid",
            "nll_lambda": 0.0,
            "nll_norm_mode": "welford_global",
            "nll_token_filter_mode": "math_tokens",
            "logprobs_required": False,
            "post_candidate_branch_width": 4,
            "max_expansions_per_node": 20,
            "post_candidate_sampling_enabled": False,
            "post_candidate_sampling_top_m": 10,
            "post_candidate_sampling_tau": 0.3,
            "locality_penalty_enabled": False,
            "locality_penalty_lambda": 0.3,
            "locality_penalty_anchor_depth": 5,
            "locality_penalty_recent_k": 1,
            "depth_guardrail_enabled": False,
            "depth_guardrail_delta": 3,
            "depth_guardrail_use_first_candidate": False,
            "adaptive_locality_enabled": True,
            "adaptive_locality_top_m": 10,
            "adaptive_locality_lambda_init": 0.2,
            "adaptive_locality_lambda_min": 0.0,
            "adaptive_locality_lambda_max": 1.0,
            "adaptive_locality_eta_up": 0.1,
            "adaptive_locality_eta_down": 0.05,
            "adaptive_locality_overlap_target": 0.8,

            "adaptive_depth_enabled": False,
            "adaptive_depth_mu_init": 0.2,
            "adaptive_depth_mu_min": 0.0,
            "adaptive_depth_mu_max": 1.0,
            "adaptive_depth_eta_up": 0.1,
            "adaptive_depth_eta_down": 0.05,

            "adaptive_branching_enabled": True,
            "adaptive_branch_probe_m0": 2,
            "adaptive_branch_mmax": 4,
            "adaptive_branch_tau": 1.5,
            "adaptive_branch_top_r": 2,
            "adaptive_branch_embed_truncate_chars": 800,


        },

        "llm_config": {

            "temperature": 1,

            "top_p": 0.9,

            "max_new_tokens": 100,

            **raw_cfg.get("generator", {}) # Override with yaml if present

        },

        "prm": raw_cfg["prm"],

        "dataset": raw_cfg["dataset"]

    }

    print(f"[BUDGET] dpts_config.max_new_tokens={cfg['dpts_config']['max_new_tokens']}")

    print(f"[top_m] dpts_config.max_new_tokens={cfg['dpts_config']['max_new_tokens']}")



    # Setup Backend (vLLM)

    backend = VLLMBackend(base_url=args.vllm_url, model=raw_cfg["generator"]["model_id"])


    # Setup Scorer (PRM)

    prm_cfg = cfg["prm"]

    prm_device = prm_cfg.get("device", "cuda:0")    
    scorer = PRMScorer(
            prm_model_id=prm_cfg["model_id"],
            device=prm_device,
            prm_dtype=prm_cfg["prm_dtype"],
            step_tag=prm_cfg["step_tag"],
            good_token=prm_cfg["good_token"],
            bad_token=prm_cfg["bad_token"],
            aggregation="last", #tried with min also , try with mean next
            mini_step=False,
            #genprm_max_new_tokens=int(prm_cfg.get("genprm_max_new_tokens", 96)),
            #genprm_temperature=float(prm_cfg.get("genprm_temperature", 0.0)),
            #genprm_samples=int(prm_cfg.get("genprm_samples", 1)),
        )
    print(f"[PRM] backend={prm_cfg.get('backend')} model={prm_cfg['model_id']} Good Token={prm_cfg['good_token']}")

    verifier = None
    vcfg = cfg["dpts_config"].get("verifier", {}) or {}
    if vcfg.get("enabled", False):
        v_device = vcfg.get("device", "cuda:1")
        verifier = GenPRMVerifier(
                model_id=vcfg["model_id"],
                device=v_device,
                dtype=vcfg.get("dtype", "bfloat16"),
                
            )
        print(f"[VERIFIER] enabled model={vcfg['model_id']} samples_per_candidate={vcfg.get('samples_per_candidate', 3)}")
    else:
        print("[VERIFIER] disabled")

    # --- NEW: TOKEN COUNTER ---

    token_counter = SimpleTokenCounter(raw_cfg["generator"]["model_id"])


    # --- NEW: TRACE LOG FILE ---

    # This file is for the DETAILED step-by-step trace (events like expand, dead_child, etc.)

    out_dir = os.path.join(EXP_ROOT, "runlogs")

    os.makedirs(out_dir, exist_ok=True)

    

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    trace_file_path = os.path.join(out_dir, f"batched_trace_{timestamp}.jsonl")

    log_f = open(trace_file_path, "w")

    print(f"Detailed traces will be streamed to: {trace_file_path}")

    metrics_url = args.metrics_url or (args.vllm_url.rstrip("/") + "/metrics")
    agg_path = os.path.join(out_dir, f"result_aggregated_{timestamp}.jsonl")
    agg_f = open(agg_path, "w")
    agg_stats = { m: {"total_samples": 0, "correct_samples": 0, "no_match_samples": 0} for m in ALL_METHODS }
    batch_metrics_path = os.path.join(out_dir, f"batch_metrics_{timestamp}.jsonl")
    batch_metrics_f = open(batch_metrics_path, "w")
    print(f"Batch metrics traces will be streamed to: {batch_metrics_path}")
    print(f"Using metrics URL: {metrics_url}")

    
    #Load Dataset
    TARGET_IDS = [
    "2","5","8","12","13","15","16","22","23","24","26","37","44","51","52",
    "65","79","74","84","85","96","97","101","102","103","104","105","109",
    "110","111","124","127","130","140","139","144","146","150","151","152",
    "155","157","163","165","169","171","174","176","180","185"
    ]

    TARGET_SET = set(TARGET_IDS)
    slice_path = os.path.join(EXP_ROOT, cfg["dataset"]["slice_path"])

    problems = []

    with open(slice_path) as f:

        for line in f:
            problems.append(json.loads(line))
            #obj = json.loads(line)
            #if obj.get("id") in TARGET_SET:
                #problems.append(obj)


    # RUN BATCHED SEARCH

    print(f"Starting Batch Search: {args.batch_problems} concurrent problems...")

    

    # Run the controller

    # Note: run_batched_tot now returns a list of summary dictionaries (not just state objects)

    summaries = run_batched_tot(

        backend=backend,

        scorer=scorer,

        token_counter=token_counter,

        problems=problems,

        cfg=cfg,

        log_f=log_f, # Passing the file handle
        agg_f=agg_f,
        agg_stats=agg_stats,

        batch_problems=args.batch_problems,
        batch_metrics_f=batch_metrics_f,
        metrics_url=metrics_url,
        verifier=verifier,
        embedder=embedder,

    )

    

    # Close the trace log

    log_f.close()
    batch_metrics_f.close()
    

    # --- SAVE FINAL SUMMARIES ---

    # This matches your request to keep the final summary saving

    out_file = os.path.join(out_dir, f"batched_run_summary_{timestamp}.jsonl")

    

    print(f"Search complete. Saving summaries to {out_file}...")

    with open(out_file, "w") as f:

        for summary in summaries:

            # We are writing the summary dict returned by the new controller

            # It already contains id, question, final_answer, is_correct, etc.

            f.write(json.dumps(summary) + "\n")

    
    report = {}
    for m in ALL_METHODS:
        total = agg_stats[m]["total_samples"]
        corr = agg_stats[m]["correct_samples"]
        no_match = agg_stats[m]["no_match_samples"]
        acc = (corr / total) if total else 0.0
        report[m] = {
                "accuracy": acc,
                "total_samples": total,
                "correct_samples": corr,
                "no_match_samples": no_match,
            }
    print(json.dumps(report, indent=4))
    agg_f.close()
    # Optional: Print simple stats to console

    correct = sum(1 for s in summaries if s.get("is_correct"))

    total = len(summaries)

    print(f"\nFinal Results: {correct}/{total} ({(correct/total)*100:.1f}%) Correct")


if __name__ == "__main__":

    main()

