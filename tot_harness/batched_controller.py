import heapq

import time

import math

import json

import logging

import time,os
import re
import regex as reg
from dataclasses import dataclass, field

from typing import List, Dict, Any, Optional
from tot_harness.grading.robust_grader import extract_final_answer, math_equal, clean_latex

from tot_harness.nll_priority import NLLPriorityHelper
from statistics import mean
from tot_harness.vllm_metrics import VLLMMetrics, get_hist_delta, get_counter_delta, get_gauge, safe_mean
from tot_harness.voting import aggregate_one, aggregate_all, ALL_METHODS
from tot_harness.voting import aggregate, extract_answer

HASH_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)
BOX_RE  = reg.compile(r"\\boxed\{((?:[^{}]|(?R))*)\}")
def extract_final_answer_strict(text: str) -> str:
    """Return ONLY the final answer string; return '' if no explicit final answer marker exists."""
    if not text:
        return ""
    m = HASH_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".")
    boxed = BOX_RE.findall(text)
    boxed = [b.strip() for b in boxed if b.strip()]
    if boxed:
        return boxed[-1]
    return ""


import regex
from math import isclose
from sympy import simplify, N
from sympy.parsing.latex import parse_latex

PAIR_RE = regex.compile(r"^\((.+),(.+)\)$")  # after normalization, no spaces
def parse_numeric_value(val: str):
    val = regex.sub(",", "", str(val))
    # strip trailing punctuation
    val = regex.sub(r"[\.，,;:]+$", "", val)
    try:
        return float(val)
    except:
        pass
    if val.endswith("%"):
        v = val[:-1]
        try:
            return float(v) / 100.0
        except:
            return None
    return None

def numeric_equal(a: float, b: float, tol=1e-4) -> bool:
    return isclose(a, b, rel_tol=tol)

def numeric_match_with_percentage(pred_s: str, ref_s: str, allow_percentage=True) -> bool:
    p = parse_numeric_value(pred_s)
    r = parse_numeric_value(ref_s)
    if p is None or r is None:
        return False

    # direct match
    if numeric_equal(p, r):
        return True

    if allow_percentage:
        # accept common scale confusions: 10% vs 10, 0.1 vs 10%, etc.
        # i.e., compare pred to r, r/100, r*100
        return numeric_equal(p, r / 100.0) or numeric_equal(p, r * 100.0)

    return False
def _try_sympy(expr: str):
    try:
        return parse_latex(expr.replace("\\\\", "\\"))
    except Exception:
        return None

def _sym_equal(a: str, b: str) -> bool:
    A = _try_sympy(a)
    B = _try_sympy(b)
    if A is None or B is None:
        return False
    try:
        if A == B:
            return True
    except Exception:
        pass
    try:
        return simplify(A - B) == 0
    except Exception:
        pass
    try:
        return isclose(float(N(A)), float(N(B)), rel_tol=1e-4)
    except Exception:
        return False

def strip_wrappers(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()

    # remove surrounding math mode (possibly repeated)
    s = s.strip()
    while s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()

    # remove common latex wrappers
    s = re.sub(r"\\boxed\{(.+)\}", r"\1", s)
    s = re.sub(r"\\text\{(.+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{(.+)\}", r"\1", s)

    # normalize escaped dollar
    s = s.replace("\\$", "$")

    return s

def normalize_degrees(s: str) -> str:
    if not s:
        return s

    # Unicode degree -> LaTeX ^\circ
    s = s.replace("°", r"^\circ")

    # collapse variants of "\circ" into "^\circ"
    s = re.sub(r"\\circ", r"^\\circ", s)                
    s = re.sub(r"\^\{\s*\\circ\s*\}", r"^\\circ", s)    
    s = re.sub(r"\^\s*\\circ", r"^\\circ", s)           

    # remove spaces around ^
    s = re.sub(r"\s*\^\s*", "^", s)
    return s

ASSIGN_RE = re.compile(r"^[a-zA-Z]\w*=(.+)$")

def rhs_if_assignment(s: str) -> str:
    m = ASSIGN_RE.match(s.replace(" ", ""))
    return m.group(1) if m else ""

def strip_currency(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    # if it looks like currency (starts with $ and then number)
    if re.match(r"^\$\s*[-+]?\d", s):
        s = s[1:].strip()
    return s
DEG_RE = re.compile(r"^(.+?)(?:\^\\circ|°)$")

def strip_degree_if_present(s: str) -> str:
    m = DEG_RE.match(s)
    return m.group(1) if m else s

def answers_match(pred: str, ref: str) -> bool:
    p = normalize_math_str(pred)
    r = normalize_math_str(ref)

    if not p or not r:
        return False
    if p.lower() == r.lower():
        return True

    # try degree-insensitive match (ONLY if one has degree and other doesn't)
    p_no_deg = strip_degree_if_present(p)
    r_no_deg = strip_degree_if_present(r)
    if (p != p_no_deg) or (r != r_no_deg):
        if p_no_deg.lower() == r_no_deg.lower():
            return True
        if numeric_match_with_percentage(p_no_deg, r_no_deg, allow_percentage=True):
            return True

    # numeric-only fast path
    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None
    pf = to_float(p); rf = to_float(r)
    if pf is not None and rf is not None:
        return isclose(pf, rf, rel_tol=1e-4)

    # tuple path: (a,b)
    mp = PAIR_RE.match(p)
    mr = PAIR_RE.match(r)
    if mp and mr:
        p1, p2 = mp.group(1), mp.group(2)
        r1, r2 = mr.group(1), mr.group(2)
        return answers_match(p1, r1) and answers_match(p2, r2)
    if numeric_match_with_percentage(p, r, allow_percentage=True):
        return True

    # symbolic fallback
    return _sym_equal(p, r)

@dataclass

class Node:

    node_id: str

    parent_id: Optional[str]

    depth: int

    prompt: str

    completion: str

    prm_score: float

    generated_tokens: int

    created_at_iter: int = 0
    effective_score: Optional[float] = None
    step_scores: List[float]= field(default_factory=list)


def _path_ids(nodes: Dict[str, Node], leaf_id: str) -> List[str]:

    """Helper to trace back the path from leaf to root for token counting."""

    out = []

    cur = leaf_id

    while cur is not None and cur in nodes:

        out.append(cur)

        cur = nodes[cur].parent_id

    return list(reversed(out))



def normalize_math_str(s: str) -> str:
    if s is None:
        return ""

    s = strip_wrappers(s)
    s = strip_currency(s)
    s = normalize_degrees(s)
    s = str(s).strip()
    rhs = rhs_if_assignment(s)
    if rhs:
        s = rhs

    # strip math mode
    s = s.strip("$")

    # remove latex sizing wrappers
    s = s.replace("\\left", "").replace("\\right", "").replace("\\,", "")
    s = s.replace("\\!", "").replace("\\;", "").replace("\\:", "")

    # normalize pi glyphs
    s = s.replace("π", "\\pi")

    # normalize \dfrac -> \frac
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\%", "%")

    s0 = s.strip()
    m = re.match(r"^([-+]?\d+(?:\.\d+)?)(?:\s*[a-zA-Z][a-zA-Z\s\/\-\^]*)$", s0)
    if m:
        s = m.group(1)

    # remove whitespace
    s = re.sub(r"\s+", "", s)
    #remove trailing junk markers like "<"
    s = re.sub(r"[<>]+$", "", s)
    # remove trailing LaTeX/English punctuation
    s = re.sub(r"[\.\s,;:]+$", "", s)

    # strip wrapping punctuation/brackets if they are just wrappers
    s = s.strip(" .;,:\n\t")
    # normalize braces around simple tokens: {x} -> x (careful: this is mild)
    s = re.sub(r"\{([a-zA-Z0-9\\]+)\}", r"\1", s)
    # remove trailing punctuation
    s = re.sub(r"[\.，,;:]+$", "", s)



    return s

class ProblemState:

    """

    Encapsulates the context of a single problem being solved.

    Acts as the 'local variables' from my previous single-problem loop.

    """

    def __init__(self, p_data: Dict, cfg: Dict, log_f, agg_f, agg_stats):

        self.p_data = p_data

        self.id = p_data['id']

        self.question = p_data['question']

        self.gold_answer = p_data['answer'] # Adjust key if your json uses 'gold'
        self.log_f = log_f

        self.agg_f = agg_f
        self.agg_stats = agg_stats

        

        # Config Extraction

        dcfg = cfg["dpts_config"]

        self.tree_width = int(dcfg["tree_width"])

        self.tree_depth = int(dcfg["tree_depth"])

        self.max_rollout_iters = int(dcfg["max_rollout"])

        self.max_step_time_s = int(dcfg["max_step_time"])

        self.max_tokens_global = int(dcfg["max_new_tokens"])

        self.voting_method = dcfg.get("voting_method", "all")
        self.max_candidates = int(dcfg.get("max_candidates", 5))

        # -------------------------------
        # Depth-bonus / trigger knobs (ablation-friendly)
        # -------------------------------
        # A) alpha_depth (float)
        self.alpha_depth = float(dcfg.get("alpha_depth", 0.0))
        # B) depth_bonus_mode: "until_first_candidate" | "always_on" | "always_off"
        self.depth_bonus_mode = str(dcfg.get("depth_bonus_mode", "always_off"))
        # C) candidate_trigger_mode: implement only "non_empty_extract_answer" now
        self.candidate_trigger_mode = str(dcfg.get("candidate_trigger_mode", "non_empty_extract_answer"))
        # D) depth_bonus_cap (int or None): bonus = alpha * min(depth, cap)
        cap = dcfg.get("depth_bonus_cap", None)
        self.depth_bonus_cap = int(cap) if cap is not None else None
        # E) tie_breaker: if priorities equal, prefer deeper nodes until first candidate (deterministic)
        self.tie_breaker = str(dcfg.get("tie_breaker", "prefer_deeper_until_candidate"))

        # Trigger / instrumentation state
        self.candidate_found_yet: bool = False
        self.candidate_first_found_iter: Optional[int] = None
        self.candidate_first_found_time: Optional[float] = None

        # Per-iter instrumentation accumulators
        self._last_popped_parent_depths: List[int] = []
        self._expanded_children_this_iter: int = 0
        

        # State Initialization

        self.t_start = time.time()

        self.nodes: Dict[str, Node] = {}

        # Heapq of (priority_key, secondary_key, push_seq, node_id)
        self.frontier = []

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

            "Please reason step by step, and ensure that the final answer includes the correct unit (e.g., ^\circ for degrees if it says find the angle in degrees).Output format MUST be exactly one line at the end. Put your final answer on its own line between tags like this:\n <<FINAL>> <answer> <</FINAL>>  .Do not output anything after that final line. Examples: <<FINAL>> 90 <</FINAL>>\n for tuples example : <FINAL>> (3, \pi/2) <</FINAL>>\n\n"

            f"{self.question}\n\nSolution:\n"

        )
        self.prompt_len = len(self.root_prompt)

        root = Node(node_id="root", parent_id=None, depth=0, prompt=self.root_prompt,completion= "", prm_score=float("-inf"), step_scores=[], generated_tokens= 0,created_at_iter=0,)

        self.nodes[root.node_id] = root

        heapq.heappush(self.frontier, (float("-inf"), 0, self.push_seq, "root"))

        self.push_seq += 1


    def _candidate_trigger(self, child_completion: str) -> bool:
        if self.candidate_trigger_mode == "non_empty_extract_answer":
            return bool(extract_answer(child_completion))
        return False

    def _depth_bonus_active(self) -> bool:
        if self.depth_bonus_mode == "always_off":
            return False
        if self.depth_bonus_mode == "always_on":
            return True
        if self.depth_bonus_mode == "until_first_candidate":
            return (not self.candidate_found_yet)
        # Unknown mode -> safest fallback (baseline)
        return False

    def _depth_bonus_value(self, depth: int) -> float:
        if not self._depth_bonus_active():
            return 0.0
        if self.alpha_depth == 0.0:
            return 0.0
        d = depth
        if self.depth_bonus_cap is not None:
            d = min(d, self.depth_bonus_cap)
        return self.alpha_depth * float(d)

    def frontier_item_for_node(self, node: Node):
    
        base_score = node.effective_score if (node.effective_score is not None) else node.prm_score
        effective_score = float(base_score) + self._depth_bonus_value(node.depth)
        priority_key = -effective_score  # heapq is min-heap; negative makes this max-by-score

        # Secondary / deterministic tie-breaker
        secondary_key = 0
        if (not self.candidate_found_yet) and (self.tie_breaker == "prefer_deeper_until_candidate"):
            secondary_key = -int(node.depth)

        item = (priority_key, secondary_key, self.push_seq, node.node_id)
        self.push_seq += 1
        return item

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

                _, _, _, nid = heapq.heappop(self.frontier)

                parents_to_expand.append(self.nodes[nid])

                parent_ids_log.append(nid)


        # Per-iter instrumentation setup
        self._last_popped_parent_depths = [p.depth for p in parents_to_expand]
        self._expanded_children_this_iter = 0
        

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

            "candidate_found_yet": bool(self.candidate_found_yet),
            "candidate_first_found_iter": self.candidate_first_found_iter,
            "avg_depth_of_popped_parents": (sum(self._last_popped_parent_depths) / len(self._last_popped_parent_depths)) if self._last_popped_parent_depths else None,
            "max_depth_of_popped_parents": max(self._last_popped_parent_depths) if self._last_popped_parent_depths else None,
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
        cand_texts  = [self.nodes[c].completion for c in self.cand_ids]
        cand_vlists = [self.nodes[c].step_scores for c in self.cand_ids]
        cand_answers = [extract_answer(t) for t in cand_texts]
        
        triples = [(cid, txt, ans) for cid, txt, ans in zip(self.cand_ids, cand_texts, cand_answers) if ans]

        final_answer = ""
        final_text = ""
        chosen_id = None
        if triples:
            from collections import Counter
            voted_answer = Counter(ans for _, _, ans in triples).most_common(1)[0][0]
            final_answer = voted_answer

            for cid, txt, ans in triples:
                if ans == voted_answer:
                    chosen_id = cid
                    final_text = txt
                    break
        else:
            final_text = aggregate(self.voting_method, cand_texts, cand_vlists)
            final_answer = extract_answer(final_text)

            for cid in self.cand_ids:
                if self.nodes[cid].completion == final_text:
                    chosen_id = cid
                    break

        if chosen_id is None and self.cand_ids:
            chosen_id = self.cand_ids[0]
            if not final_text:
                final_text = self.nodes[chosen_id].completion
            if not final_answer:
                final_answer = extract_answer(final_text)

        gold_raw = str(self.gold_answer).strip()
        gold_cleaned = extract_final_answer_strict(gold_raw) or gold_raw
        is_correct = answers_match(final_answer, gold_cleaned)


        if self.voting_method == "all" and getattr(self, "agg_f", None) is not None and getattr(self, "agg_stats", None) is not None:
            # IMPORTANT: run aggregation on the SAME candidate set you used for voting.
            # Here we use cand_texts and cand_vlists (already built above).
            from tot_harness.voting import aggregate_all, ALL_METHODS
            all_choices = aggregate_all(cand_texts, cand_vlists)
            methods_out = {}
            for m in ALL_METHODS:
                pred_ans = all_choices[m]["chosen_answer"]
                self.agg_stats[m]["total_samples"] += 1
                if not pred_ans:
                    self.agg_stats[m]["no_match_samples"] += 1
                    corr = False
                else :
                    corr = answers_match(pred_ans, gold_cleaned)
                    if corr:
                        self.agg_stats[m]["correct_samples"] += 1

                methods_out[m] = {
                        "final_answer": pred_ans,
                        "is_correct": bool(corr),
                    }
            agg_record = {
                    "problem_id": self.id,
                    "gold_answer_clean": gold_cleaned,
                    "num_candidates": len(cand_texts),
                    "methods": methods_out,
                    "majority_vote_answer": final_answer,
                    "candidate_extracted_answers": cand_answers,
                }
            self.agg_f.write(json.dumps(agg_record) + "\n")
            self.agg_f.flush()
        # Token Accounting



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
            "sample_candidate_answers": cand_answers[:10],
            "final_method": self.voting_method,

            "final_answer":final_answer,
        
            "gold_answer_clean": gold_cleaned,

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

    log_f,    # Added file handle for centralized logging
    agg_f,
    agg_stats,


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

            active_states.append(ProblemState(p, cfg, log_f,agg_f,agg_stats,))

    

    refill_batch()

    

    dcfg = cfg["dpts_config"]

    lcfg = cfg["llm_config"]
    nll_helper = NLLPriorityHelper(dcfg)

    branch_factor = int(dcfg.get("num_branch", 1))

    max_new_tokens=int(lcfg["max_new_tokens"])

    
    return_logprobs = nll_helper.want_logprobs()
    # Sampling params (shared)

    sampling = dict(

        temperature=float(lcfg["temperature"]),

        top_p=float(lcfg["top_p"]),

        max_new_tokens=int(lcfg["max_new_tokens"]),
        max_tokens=int(lcfg["max_new_tokens"]),
        return_logprobs=return_logprobs,
        logprobs_k=1,

        n=2 

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
                any_lp = any((getattr(gr, "token_logprobs", None) is not None) for gr in gen_results) if gen_results else False
                nll_helper.ensure_or_fallback(any_lp)

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

                

                for j, child_text in enumerate(res.texts):
                    lp = None
                    toks = None
                    if hasattr(res, "token_logprobs") and res.token_logprobs is not None and j < len(res.token_logprobs):
                        lp = res.token_logprobs[j]
                    if hasattr(res, "tokens") and res.tokens is not None and j < len(res.tokens):
                        toks = res.tokens[j]

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
                    eff_score, nll_dbg = nll_helper.compute_effective_score(
                            problem_id=state.id,
                            prm_score=prm_score,
                            token_logprobs=lp,
                            tokens=toks,
                        )


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
                        effective_score=eff_score,

                        step_scores=[], # PRM usually gives one float, list if granular

                        generated_tokens=gen_toks,
                        created_at_iter=state.iter

                    )

                    

                    state.nodes[nid] = new_node

                    

                    # Push to Frontier

                    heapq.heappush(state.frontier, state.frontier_item_for_node(new_node))

                    

                    # Candidate trigger + state transition instrumentation
                    is_candidate = state._candidate_trigger(child_text) 
                    #is_candidate = state._candidate_trigger(solution_so_far, gen_toks, max_new_tokens)

                    if is_candidate:

                        state.cand_ids.append(nid)

                        if not state.candidate_found_yet:

                            state.candidate_found_yet = True

                            state.candidate_first_found_iter = state.iter

                            state.candidate_first_found_time = time.time()

                            state.log_f.write(json.dumps({

                                "problem_id": state.id,

                                "event": "candidate_first_found",
                                "timestamp": state.candidate_first_found_time,

                                "iter": state.candidate_first_found_iter,

                                "node_id": nid,

                                "depth": new_node.depth,

                                "prm_score": prm_score,
                                **nll_dbg,

                                "num_candidates_total": len(state.cand_ids),

                                "depth_bonus_mode": state.depth_bonus_mode,

                                "alpha_depth": state.alpha_depth,

                                "depth_bonus_cap": state.depth_bonus_cap,

                                "candidate_trigger_mode": state.candidate_trigger_mode,

                            }) + "\n")

                    # per-iter expansion counter for instrumentation
                    state._expanded_children_this_iter += 1
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
                        **nll_dbg,
                        "child_completion": child_text,
                        "child_extract_answer": extract_answer(child_text),

                        "generated_tokens": gen_toks

                    }) + "\n")


            # Increment Iterations for all touched states

            # effectively incremented the 'iter' counter for every state that contributed prompts

    

            unique_states_touched = set(s for s, _ in request_map)

            # Per-iteration summary instrumentation (one record per touched state)
            for s in unique_states_touched:

                depths = list(getattr(s, "_last_popped_parent_depths", []))

                s.log_f.write(json.dumps({

                    "problem_id": s.id,

                    "event": "iter_stats",
                    "timestamp": time.time(),

                    "iter": s.iter,

                    "candidate_found_yet": bool(s.candidate_found_yet),

                    "candidate_first_found_iter": s.candidate_first_found_iter,

                    "avg_depth_of_popped_parents": (sum(depths)/len(depths)) if depths else None,

                    "max_depth_of_popped_parents": max(depths) if depths else None,

                    "number_of_nodes_expanded_this_iter": int(getattr(s, "_expanded_children_this_iter", 0)),

                    "number_of_candidates_total": int(len(s.cand_ids)),

                    "frontier_len_post": int(len(s.frontier)),

                    "depth_bonus_mode": s.depth_bonus_mode,

                    "alpha_depth": s.alpha_depth,

                    "depth_bonus_cap": s.depth_bonus_cap,

                }) + "\n")

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

