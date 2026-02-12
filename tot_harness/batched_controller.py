import heapq

import time

import math

import json

import logging

import time

from dataclasses import dataclass, field

from typing import List, Dict, Any, Optional


from statistics import mean
from tot_harness.vllm_metrics import VLLMMetrics, get_hist_delta, get_counter_delta, get_gauge, safe_mean

from tot_harness.voting import aggregate, extract_answer


@dataclass

class Node:

    node_id: str

    parent_id: Optional[str]

    depth: int

    prompt: str

    completion: str

    prm_score: float

    step_scores: List[float]

    generated_tokens: int

    created_at_iter: int = 0


def _path_ids(nodes: Dict[str, Node], leaf_id: str) -> List[str]:

    """Helper to trace back the path from leaf to root for token counting."""

    out = []

    cur = leaf_id

    while cur is not None and cur in nodes:

        out.append(cur)

        cur = nodes[cur].parent_id

    return list(reversed(out))

def normalize_math_str(s):
    # Remove whitespace, \\left, \\right, and box
        s = s.replace(" ", "").replace("\\left", "").replace("\\right", "").replace("\\boxed", "")
        return s

class ProblemState:

    """

    Encapsulates the context of a single problem being solved.

    Acts as the 'local variables' from my previous single-problem loop.

    """

    def __init__(self, p_data: Dict, cfg: Dict, log_f):

        self.p_data = p_data

        self.id = p_data['id']

        self.question = p_data['question']

        self.gold_answer = p_data['answer'] # Adjust key if your json uses 'gold'

        self.log_f = log_f

        

        # Config Extraction

        dcfg = cfg["dpts_config"]

        self.tree_width = int(dcfg["tree_width"])

        self.tree_depth = int(dcfg["tree_depth"])

        self.max_rollout_iters = int(dcfg["max_rollout"])

        self.max_step_time_s = int(dcfg["max_step_time"])

        self.max_tokens_global = int(dcfg["max_new_tokens"])

        self.voting_method = dcfg.get("voting_method", "all")
        self.max_candidates = int(dcfg.get("max_candidates", 5))
        

        # State Initialization

        self.t_start = time.time()

        self.nodes: Dict[str, Node] = {}

        self.frontier = [] # Heapq of (-score, push_seq, node_id)

        self.push_seq = 0

        self.cand_ids: List[str] = []

        

        # Counters

        self.total_generated_tokens = 0

        self.total_llm_time = 0.0

        self.total_prm_time = 0.0

        self.next_id = 0

        self.expanded_children = 0

        self.iter = 0

        self.finished = False

        self.stop_reason = "running"


        # Root Node Setup

        self.root_prompt = (

            "Please reason step by step, and ensure that the final answer includes the correct unit (e.g., ^\circ for degrees if it’s an angle). Put your final answer as '#### <number>'.\n\n"

            f"{self.question}\n\nSolution:\n"

        )
        self.prompt_len = len(self.root_prompt)

        root = Node("root", None, 0, self.root_prompt, "", float("-inf"), [], 0,created_at_iter=0)

        self.nodes[root.node_id] = root

        heapq.heappush(self.frontier, (float("-inf"), self.push_seq, "root"))

        self.push_seq += 1


    def check_budgets(self) -> bool:

        """Returns True if problem should stop."""

        if self.finished: return True

        

        if (time.time() - self.t_start) > self.max_step_time_s:

            self.stop_reason = "budget_time"

            return True

        if self.total_generated_tokens >= self.max_tokens_global:

            self.stop_reason = "budget_tokens"

            return True

        if self.iter >= self.max_rollout_iters:

            self.stop_reason = "budget_rollout"

            return True

        if not self.frontier:

            self.stop_reason = "frontier_empty"

            return True

        #if len(self.cand_ids) >= self.max_candidates:
            #self.stop_reason = "budget_candidates"
            #return True

        return False


    def get_parents_to_expand(self) -> List[Node]:

        """Pops the next batch of parents from the frontier."""

        parents_to_expand = []

        parent_ids_log = []

        

        # Expand up to tree_width

        for _ in range(self.tree_width):

            if self.frontier:

                _, _, nid = heapq.heappop(self.frontier)

                parents_to_expand.append(self.nodes[nid])

                parent_ids_log.append(nid)

        

        if not parents_to_expand:

            return []


        # Log rollout_iter event (matching your format)

        log_entry = {

            "problem_id": self.id,

            "event": "rollout_iter",

            "iter": self.iter,

            "parents": parent_ids_log,

            "frontier_len_before": len(self.frontier) + len(parents_to_expand),

            "total_generated_tokens_so_far": self.total_generated_tokens,

        }

        self.log_f.write(json.dumps(log_entry) + "\n")

        

        return parents_to_expand
    
    

    def finalize(self):

        """Performs voting, cleanup, and summary logging."""

        if self.finished: return

        self.finished = True

        

        # If no specific candidates found, fallback to best PRM node (non-root)

        if not self.cand_ids:

            all_ids = [k for k in self.nodes.keys() if k != "root"]

            if all_ids:

                best = max(all_ids, key=lambda i: self.nodes[i].prm_score)

                self.cand_ids = [best]


        # Voting Logic

        cand_texts = [self.nodes[c].completion for c in self.cand_ids]

        cand_vlists = [self.nodes[c].step_scores for c in self.cand_ids]


        final_text = aggregate(self.voting_method, cand_texts, cand_vlists)

        final_answer = extract_answer(final_text)

        gold_cleaned = extract_answer(str(self.gold_answer))
        if gold_cleaned == "":
            gold_cleaned = str(self.gold_answer).strip()

        

        is_correct = (final_answer != "" and gold_cleaned != "" and (normalize_math_str(final_answer) == normalize_math_str(gold_cleaned)))


        # Token Accounting

        chosen_id = None

        for cid in self.cand_ids:

            if self.nodes[cid].completion == final_text:

                chosen_id = cid

                break

        if chosen_id is None and self.cand_ids:

            chosen_id = self.cand_ids[0]


        tokens_kept = 0

        if chosen_id:

            kept_ids = _path_ids(self.nodes, chosen_id)

            tokens_kept = sum(self.nodes[i].generated_tokens for i in kept_ids if i != "root")

            

        tokens_pruned = max(0, self.total_generated_tokens - tokens_kept)


        # Summary Log

        summary = {

            "problem_id": self.id,

            "stop_reason": self.stop_reason,

            "rollout_iters_target": self.max_rollout_iters,

            "nodes_generated": self.expanded_children,

            "num_candidates": len(self.cand_ids),

            "final_method": self.voting_method,

            "final_answer": final_answer,

            "gold_answer": gold_cleaned,

            "is_correct": bool(is_correct),

            "total_generated_tokens": int(self.total_generated_tokens),

            "tokens_kept_in_final_path": int(tokens_kept),

            "tokens_pruned": int(tokens_pruned),

            "time_total_s": float(time.time() - self.t_start),

            "time_llm_forward_s": float(self.total_llm_time),

            "time_prm_s": float(self.total_prm_time),

        }

        self.log_f.write(json.dumps({"problem_id": self.id, "event": "summary", **summary}) + "\n")

        self.summary_data = summary



def run_batched_tot(

    backend, 

    scorer, 

    token_counter, 

    problems: List[Dict], 

    cfg: Dict,

    log_f,         # Added file handle for centralized logging

    batch_problems: int = 10,
    batch_metrics_f=None,
    metrics_url: Optional[str] = None,

):

    """

    Main controller for Batched Tree-of-Thought.

    """
    metrics = VLLMMetrics(metrics_url) if (metrics_url and batch_metrics_f) else None
    batch_id = 0
    

    # 1. Initialize States

    active_states: List[ProblemState] = []

    finished_states: List[ProblemState] = []

    

    # Initial fill of the batch

    queue_problems = problems.copy()

    

    def refill_batch():

        while len(active_states) < batch_problems and queue_problems:

            p = queue_problems.pop(0)

            active_states.append(ProblemState(p, cfg, log_f))

    

    refill_batch()

    

    dcfg = cfg["dpts_config"]

    lcfg = cfg["llm_config"]

    branch_factor = int(dcfg.get("num_branch", 1))

    

    # Sampling params (shared)

    sampling = dict(

        temperature=float(lcfg["temperature"]),

        top_p=float(lcfg["top_p"]),

        max_new_tokens=int(lcfg["max_new_tokens"]),
        max_tokens=int(lcfg["max_new_tokens"]),

        n=branch_factor 

    )


    global_step = 0


    try:

        while active_states:

            global_step += 1

            print(f"--- Batch Step {global_step} | Active: {len(active_states)} | Finished: {len(finished_states)} ---")

            

            # --- PHASE 1: SELECTION & COLLECTION ---

            batch_prompts = []

            request_map = [] # Maps batch_index -> (state_index, parent_node_obj)

            

            # We iterate a copy so we can remove finished states safely

            for state in list(active_states):

                # 1. Check Budgets

                if state.check_budgets():

                    state.finalize()

                    active_states.remove(state)

                    finished_states.append(state)

                    continue

                

                # 2. Pop Parents

                parents = state.get_parents_to_expand()

                

                # 3. Add to Batch

                for parent_node in parents:

                    if parent_node.depth >= state.tree_depth:

                        continue

                    

                    # Store mapping for the response

                    full_text = parent_node.prompt + parent_node.completion

                    batch_prompts.append(full_text)

                    request_map.append((state, parent_node))

            

            # Refill if spots opened up

            refill_batch()


            if not batch_prompts:

                # If active_states is not empty but no one had prompts (e.g. all empty frontiers), 

                # we just loop again (budgets will catch them).

                if not active_states: break

                continue


            # --- PHASE 2: GENERATION ---

            t0 = time.time()
            

            try:

                # Backend generates 'n' (branch_factor) per prompt
                batch_id += 1

                # Build per-request batch metadata BEFORE calling vLLM
                req_meta = []
                stales = []
                prompt_toks = []
                depths = []


                for req_idx, (state, parent_node) in enumerate(request_map):
                    st = state.iter - parent_node.created_at_iter
                    pt = token_counter.count_prompt_tokens(batch_prompts[req_idx]) if hasattr(token_counter, "count_prompt_tokens") else None
                    req_meta.append({
                        "req_idx": req_idx,
                        "problem_id": state.id,
                        "parent_id": parent_node.node_id,
                        "depth": parent_node.depth,
                        "staleness": st,
                        "prompt_tokens": pt,
                    })
                    stales.append(st)
                    depths.append(parent_node.depth)
                    if pt is not None:
                        prompt_toks.append(pt)

                pre_t, pre_m = (None, None)
                if metrics:
                    pre_t, pre_m = metrics.snapshot()


                gen_results = backend.generate_batch(batch_prompts, sampling=sampling)

            except Exception as e:

                print(f"CRITICAL BACKEND ERROR: {e}")

                break


            post_t, post_m = (None, None)
            if metrics:
                post_t, post_m = metrics.snapshot()

            if batch_metrics_f and metrics and pre_m is not None and post_m is not None:
                ttft = get_hist_delta(pre_m, post_m, "vllm:time_to_first_token_seconds")
                prefill_t = get_hist_delta(pre_m, post_m, "vllm:request_prefill_time_seconds")
                e2e = get_hist_delta(pre_m, post_m, "vllm:e2e_request_latency_seconds")
                kv_comp = get_hist_delta(pre_m, post_m, "vllm:request_prefill_kv_computed_tokens")

                # Counters
                d_hits = get_counter_delta(pre_m, post_m, "vllm:prefix_cache_hits_total")
                d_queries = get_counter_delta(pre_m, post_m, "vllm:prefix_cache_queries_total")

                kv_usage = get_gauge(post_m, "vllm:kv_cache_usage_perc", agg="max")
                n_run = get_gauge(post_m, "vllm:num_requests_running", agg="sum")
                n_wait = get_gauge(post_m, "vllm:num_requests_waiting", agg="sum")


                rec = {
                        "event": "batch_metrics",
                        "batch_id": batch_id,
                        "timestamp_pre": pre_t,
                        "timestamp_post": post_t,
                        "batch_wall_s": float((post_t - pre_t) if (post_t and pre_t) else (t1 - t0)),
                        "num_requests": int(len(batch_prompts)),

                        # What's in the batch:
                        "staleness_min": int(min(stales)) if stales else None,
                        "staleness_mean": float(mean(stales)) if stales else None,
                        "staleness_max": int(max(stales)) if stales else None,
                        "depth_mean": float(mean(depths)) if depths else None,
                        "prompt_tokens_mean": float(mean(prompt_toks)) if prompt_toks else None,

                        "requests": req_meta,  # optional but super useful offline

                        # Server-side deltas (windowed):
                        "delta_prefix_cache_hits": float(d_hits),
                        "delta_prefix_cache_queries": float(d_queries),
                        "prefix_hit_rate_window": (float(d_hits) / float(d_queries)) if d_queries > 0 else None,

                        "delta_ttft_count": float(ttft["count"]),
                        "ttft_mean_s_window": safe_mean(ttft["sum"], ttft["count"]),

                        "delta_prefill_count": float(prefill_t["count"]),
                        "prefill_mean_s_window": safe_mean(prefill_t["sum"], prefill_t["count"]),

                        "delta_e2e_count": float(e2e["count"]),
                        "e2e_mean_s_window": safe_mean(e2e["sum"], e2e["count"]),

                        "delta_prefill_kv_comp_count": float(kv_comp["count"]),
                        "prefill_kv_comp_tokens_mean_window": safe_mean(kv_comp["sum"], kv_comp["count"]),

                        # Cache pressure / queue context:
                        "kv_cache_usage_perc_post": kv_usage,
                        "num_requests_running_post": n_run,
                        "num_requests_waiting_post": n_wait,
                    }
                batch_metrics_f.write(json.dumps(rec) + "\n")


            t1 = time.time()

            

            # Attribute time roughly

            avg_time = (t1 - t0) / len(batch_prompts) if batch_prompts else 0


            # --- PHASE 3: EXPANSION & SCORING ---

            for req_idx, res in enumerate(gen_results):

                state, parent_node = request_map[req_idx]

                

                # Update LLM Time

                state.total_llm_time += res.time_llm_forward_s if hasattr(res, 'time_llm_forward_s') else avg_time

                

                for child_text in res.texts:

                    # Budget Check inside the loop

                    if state.total_generated_tokens >= state.max_tokens_global:

                        state.stop_reason = "budget_tokens"

                        continue


                    # Count Tokens

                    # Note: parent_node.prompt + parent_node.completion is the 'prompt' for this step

                    context = parent_node.prompt + parent_node.completion

                    gen_toks = token_counter.count_generated_tokens_delta(context, child_text)

                    

                    # Dead Node Check

                    if gen_toks == 0 or not child_text.strip():

                        state.log_f.write(json.dumps({

                            "problem_id": state.id,

                            "event": "dead_child",

                            "iter": state.iter,

                            "parent_id": parent_node.node_id,

                            "why": "empty_completion"

                        }) + "\n")

                        continue


                    state.total_generated_tokens += gen_toks

                    state.expanded_children += 1

                    full_text = parent_node.prompt + parent_node.completion + child_text
                   # 2. Extract ONLY the solution steps (remove system prompt)
                    if len(full_text) > state.prompt_len:
                        solution_so_far = full_text[state.prompt_len:]
                    else:
                        solution_so_far = child_text # Fallback
                  

                    # PRM Scoring

                    s0 = time.time()

                    try:
                        prm_score = float(scorer.score(state.question, solution_so_far))
                    except Exception as e:
                        print(f"Scoring Error: {e}")
                        prm_score = -1.0

                

                    

                    if not math.isfinite(prm_score): prm_score = -1.0

                    state.total_prm_time += (time.time() - s0)


                    # Create Node

                    nid = f"n{state.next_id}"

                    state.next_id += 1

                    

                    new_node = Node(

                        node_id=nid,

                        parent_id=parent_node.node_id,

                        depth=parent_node.depth + 1,

                        prompt=parent_node.prompt + parent_node.completion,

                        completion=child_text,

                        prm_score=prm_score,

                        step_scores=[], # PRM usually gives one float, list if granular

                        generated_tokens=gen_toks,
                        created_at_iter=state.iter

                    )

                    

                    state.nodes[nid] = new_node

                    

                    # Push to Frontier

                    heapq.heappush(state.frontier, (-prm_score, state.push_seq, nid))

                    state.push_seq += 1


                    

                    # Candidate Check (Logic: extract_answer)

                    if extract_answer(child_text):

                        state.cand_ids.append(nid)


                    # Log Expand
                    staleness = state.iter - parent_node.created_at_iter


                    state.log_f.write(json.dumps({

                        "problem_id": state.id,

                        "event": "expand",
                        "timestamp": time.time(),

                        "iter": state.iter,

                        "parent_id": parent_node.node_id,

                        "node_id": nid,

                        "staleness": staleness,

                        "depth": new_node.depth,

                        "prm_score": prm_score,

                        "generated_tokens": gen_toks

                    }) + "\n")


            # Increment Iterations for all touched states

            # effectively incremented the 'iter' counter for every state that contributed prompts

    

            unique_states_touched = set(s for s, _ in request_map)

            for s in unique_states_touched:

                s.iter += 1


    except KeyboardInterrupt:

        print("\n[User Interrupt] Stopping search and finalizing...")


    # Finalize all remaining

    for s in active_states:

        s.finalize()

        finished_states.append(s)


    # Return summary dicts

    return [s.summary_data for s in finished_states if hasattr(s, 'summary_data')]
