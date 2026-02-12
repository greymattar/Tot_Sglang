
import heapq

import time

import math

import json

import logging

import time

from dataclasses import dataclass, field

from typing import List, Dict, Any, Optional



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

        self.gold_answer = p_data['answer'] 

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
        if len(self.cand_ids) >= self.max_candidates:
            self.stop_reason = "budget_candidates"
            return True

            

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
    
    def log_expansion_with_metrics(self, parent_node, new_node, batch_metrics, req_idx, gen_toks, prm_score, child_text):
        """Enhanced logging for evidence collection"""
    
        staleness = self.iter - parent_node.created_at_iter
    
        # Calculate prompt token count (approximate)
        full_prompt = parent_node.prompt + parent_node.completion
        prompt_tokens_approx = len(full_prompt) // 4  # ~4 chars per token
    
        log_entry = {
            "problem_id": self.id,
            "event": "expand",
            "timestamp": time.time(),
            "iter": self.iter,
        
        # Tree structure evidence
            "node_id": new_node.node_id,
            "parent_id": parent_node.node_id,
            "depth": new_node.depth,
            "staleness": staleness,
            "parent_created_at_iter": parent_node.created_at_iter,
        
        # Request characteristics
            "prompt_chars": len(full_prompt),
            "prompt_tokens_approx": prompt_tokens_approx,
            "generated_tokens": gen_toks,
            "prm_score": prm_score,
        
        # Performance evidence
            "e2e_latency_s": getattr(batch_metrics, 'e2e_latency_s', 0),
        
        # vLLM state evidence (CRITICAL FOR LRU)
            "kv_cache_usage_pre": batch_metrics.get('kv_cache_usage', 0),
            "preemptions_pre": batch_metrics.get('preemptions', 0),
            "requests_running_pre": batch_metrics.get('requests_running', 0),
            "preemptions_delta": batch_metrics.get('preemptions_delta', 0),
        
        # Batch context
            "batch_size": batch_metrics.get('batch_size', 0),
            "batch_position": req_idx,
            "avg_prompt_length_batch": batch_metrics.get('avg_prompt_length', 0),
        
        # For debugging
            "parent_prompt_snippet": parent_node.prompt[-100:] if len(parent_node.prompt) > 100 else parent_node.prompt
        }
    
        self.log_f.write(json.dumps(log_entry) + "\n")    

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

    token_counter, # Added to match your logic

    problems: List[Dict], 

    cfg: Dict,

    log_f,         # Added file handle for centralized logging

    batch_problems: int = 10

):

    """

    Main controller for Batched Tree-of-Thought.

    """

    

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

        #max_new_tokens=int(lcfg["max_new_tokens"]),
        #max_tokens=int(lcfg["max_new_tokens"]),

        #n=branch_factor 

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

                batch_output = backend.generate_batch(batch_prompts, sampling=sampling)
                gen_results = batch_output['results']
                batch_metadata = batch_output['metrics']

            except Exception as e:

                print(f"CRITICAL BACKEND ERROR: {e}")

                break

            t1 = time.time()

            

            # Attribute time roughly

            batch_duration = batch_metadata.get('deltas', {}).get('batch_duration_s', t1 - t0)
            avg_time = batch_duration / len(batch_prompts) if batch_prompts else 0


            # --- PHASE 3: EXPANSION & SCORING ---

            for req_idx, res in enumerate(gen_results):

                state, parent_node = request_map[req_idx]
                #batch_metadata = batch_result['metrics']

                prompt_latency = res.get('e2e_latency_s', avg_time)
                state.total_llm_time += prompt_latency

                

                for child_text in res['texts']:

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


                    state.log_expansion_with_metrics(
                        parent_node=parent_node,
                        new_node=new_node,
                        batch_metrics={
                            'kv_cache_usage': batch_metadata['pre'].get('kv_cache_usage', 0),
                            'preemptions': batch_metadata['pre'].get('preemptions', 0),
                            'requests_running': batch_metadata['pre'].get('requests_running', 0),
                            'preemptions_delta': batch_metadata['deltas'].get('preemptions_delta', 0),
                            'batch_size': batch_metadata['batch_info']['size'],
                            'avg_prompt_length': batch_metadata['deltas'].get('avg_prompt_length', 0),
                            'e2e_latency_s': prompt_latency
                        },
                        req_idx=req_idx,
                        gen_toks=gen_toks,
                        prm_score=prm_score,
                        child_text=child_text
                    )

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

