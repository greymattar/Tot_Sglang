
import argparse

import yaml

import json

import os

import torch

import time

from transformers import AutoTokenizer
from tot_harness.embedder_backend import VLLMEmbedder

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

    parser.add_argument("--batch_problems", type=int, default=1, help="Concurrent problems")

    parser.add_argument("--batch_width", type=int, default=4, help="Nodes per problem per step")

    parser.add_argument("--metrics_url", default=None, help="Prometheus /metrics endpoint (e.g. http://127.0.0.1:18000/metrics). ""If not set, will use --vllm_url + '/metrics'.")
    parser.add_argument("--disable_metrics", action="store_true",help="Disable vLLM /metrics polling (avoid 503 during warmup)")
    parser.add_argument("--nll_lambda", type=float, default=0.0,
                        help="Weight for NLL in hybrid priority_mode=prm_nll_hybrid")
    parser.add_argument("--embed_url", default=None)
    parser.add_argument("--embed_model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--uaa_enabled", action="store_true")
    parser.add_argument("--uaa_lambda", type=float, default=1.5)
    parser.add_argument("--uaa_q", type=int, default=2)

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

            "max_new_tokens": 8000,

            "voting_method": "all",

            "num_branch": raw_cfg.get("sampling", {}).get("n", 4),
            "depth_bonus_mode": "until_first_candidate", "alpha_depth": 0.0, 
            "priority_mode": "prm_nll_hybrid",
            "nll_lambda": args.nll_lambda,
            "nll_norm_mode": "welford_global",
            "nll_token_filter_mode": "math_tokens",
            "logprobs_required": True

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
    print(f"[CONFIG] nll_lambda={cfg['dpts_config']['nll_lambda']}")
    print(f"[CONFIG] max_token={cfg['dpts_config']['max_new_tokens']}")
    cfg["dpts_config"]["uncertainty"] = {
    "enabled": bool(args.uaa_enabled),
    "metric": "effective_rank",
    "lambda": float(args.uaa_lambda),
    "q": int(args.uaa_q),
    "canonicalize": True,
    
    }
    #cfg["dpts_config"]["uncertainty"] = raw_cfg.get("dpts_config", {}).get("uncertainty", {})
    embedder = None
    if cfg["dpts_config"]["uncertainty"]["enabled"]:
        embed_url = args.embed_url or args.vllm_url
        embedder = VLLMEmbedder(base_url=embed_url, model=args.embed_model)

    cfg["dpts_config"]["depth_bonus_adaptive"] = {
        "enabled": False,
         "alpha_min": 0.2,
        "alpha_max": 1.0,
        "sigmoid_s": 0.15,   # smoothness
        "ema_beta": 0.9,     # EMA smoothing; 0.9 = slow, stable
        "tau": 3.0,          # anneal time after first candidate
    }


    # Setup Backend (vLLM)

    backend = VLLMBackend(base_url=args.vllm_url, model=raw_cfg["generator"]["model_id"])


    # Setup Scorer (PRM)

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

    metrics_url = None if args.disable_metrics else (args.metrics_url or (args.vllm_url.rstrip("/") + "/metrics"))
    agg_path = os.path.join(out_dir, f"result_aggregated_{timestamp}.jsonl")
    agg_f = open(agg_path, "w")
    agg_stats = { m: {"total_samples": 0, "correct_samples": 0, "no_match_samples": 0} for m in ALL_METHODS }
    
    batch_metrics_f = None

    if metrics_url is not None:
        batch_metrics_path = os.path.join(out_dir, f"batch_metrics_{timestamp}.jsonl")
        batch_metrics_f = open(batch_metrics_path, "w")
        print(f"Batch metrics traces will be streamed to: {batch_metrics_path}")
        print(f"Using metrics URL: {metrics_url}")
    else:
        print("Metrics disabled (not polling /metrics).")
        
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
        embedder=embedder,

    )

    

    # Close the trace log

    log_f.close()
    if batch_metrics_f is not None:
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

