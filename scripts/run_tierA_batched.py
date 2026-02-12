
import argparse

import yaml

import json

import os

import torch

import time

from transformers import AutoTokenizer


#from tot_harness.backend_vllm import VLLMBackend

from tot_harness.backend_vllm import VLLMBackend as VLLMBackend
from tot_harness.batched_controller import run_batched_tot

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

            "max_step_time": 360,

            "max_new_tokens": 7168,

            "voting_method": "all",

            "num_branch": raw_cfg.get("sampling", {}).get("n", 4)

        },

        "llm_config": {

            "temperature": 1,

            "top_p": 0.9,

            "max_new_tokens": 32,

            **raw_cfg.get("generator", {}) # Override with yaml if present

        },

        "prm": raw_cfg["prm"],

        "dataset": raw_cfg["dataset"]

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

    metrics_url = args.metrics_url or (args.vllm_url.rstrip("/") + "/metrics")

    batch_metrics_path = os.path.join(out_dir, f"batch_metrics_{timestamp}.jsonl")
    batch_metrics_f = open(batch_metrics_path, "w")
    print(f"Batch metrics traces will be streamed to: {batch_metrics_path}")
    print(f"Using metrics URL: {metrics_url}")


    # Load Dataset

    slice_path = os.path.join(EXP_ROOT, cfg["dataset"]["slice_path"])

    problems = []

    with open(slice_path) as f:

        for line in f:

            problems.append(json.loads(line))


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

        batch_problems=args.batch_problems,
        batch_metrics_f=batch_metrics_f,
        metrics_url=metrics_url,

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


    # Optional: Print simple stats to console

    correct = sum(1 for s in summaries if s.get("is_correct"))

    total = len(summaries)

    print(f"\nFinal Results: {correct}/{total} ({(correct/total)*100:.1f}%) Correct")


if __name__ == "__main__":

    main()

