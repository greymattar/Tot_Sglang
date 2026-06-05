import heapq

import time

import math

import json
import numpy as np

import logging
import statistics
import time,os
import re
import regex as reg
from dataclasses import dataclass, field

from typing import List, Dict, Any, Optional
from tot_harness.grading.robust_grader import extract_final_answer, math_equal, clean_latex

from tot_harness.nll_priority import NLLPriorityHelper
from tot_harness.backend_vllm import GenResult
from statistics import mean
from tot_harness.vllm_metrics import VLLMMetrics, get_hist_delta, get_counter_delta, get_gauge, safe_mean
from tot_harness.voting import aggregate_one, aggregate_all, ALL_METHODS
from tot_harness.voting import aggregate, extract_answer
import regex
from math import isclose
from sympy import simplify, N
from sympy.parsing.latex import parse_latex

HASH_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)
BOX_RE = regex.compile(r"\\boxed\{((?:[^{}]|(?R))*)\}")
PAIR_RE = regex.compile(r"^\((.+),(.+)\)$")
ASSIGN_RE = re.compile(r"^[a-zA-Z]\w*=(.+)$")
DEG_RE = re.compile(r"^(.+?)(?:\^\\circ|°)$")

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


def strip_wrappers(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()

    # remove surrounding math mode repeatedly
    while s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()

    # remove common latex wrappers
    s = re.sub(r"\\boxed\{(.+)\}", r"\1", s)
    s = re.sub(r"\\text\{(.+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{(.+)\}", r"\1", s)

    # normalize escaped dollar
    s = s.replace("\\$", "$")
    return s


def strip_currency(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if re.match(r"^\$\s*[-+]?\d", s):
        s = s[1:].strip()
    return s


def normalize_degrees(s: str) -> str:
    if not s:
        return s

    # unicode degree -> latex style
    s = s.replace("°", r"^\circ")

    # normalize variants of \circ to ^\circ
    s = re.sub(r"(?<!\^)\\circ", r"^\\circ", s)
    s = re.sub(r"\^\{\s*\\circ\s*\}", r"^\\circ", s)
    s = re.sub(r"\^\s*\\circ", r"^\\circ", s)

    # tighten spaces around ^
    s = re.sub(r"\s*\^\s*", "^", s)
    return s


def rhs_if_assignment(s: str) -> str:
    m = ASSIGN_RE.match(s.replace(" ", ""))
    return m.group(1) if m else ""


def strip_degree_if_present(s: str) -> str:
    m = DEG_RE.match(s)
    return m.group(1) if m else s


def _normalize_top_level_frac(s: str) -> str:
    """
    If there is a top-level /, strip one outer layer of parentheses
    around numerator/denominator only.
    Example: (a)/(b) -> a/b
    """
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "/" and depth == 0:
            left, right = s[:i].strip(), s[i + 1:].strip()

            def strip_one_layer(x):
                if x.startswith("(") and x.endswith(")"):
                    inner = x[1:-1]
                    d = 0
                    for c in inner:
                        if c == "(":
                            d += 1
                        elif c == ")":
                            d -= 1
                            if d < 0:
                                return x
                    if d == 0:
                        return inner.strip()
                return x

            left = strip_one_layer(left)
            right = strip_one_layer(right)
            return f"{left}/{right}"
    return s


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


def parse_numeric_value(val: str):
    val = regex.sub(",", "", str(val))
    val = regex.sub(r"[\.，,;:]+$", "", val)

    try:
        return float(val)
    except Exception:
        pass

    if val.endswith("%"):
        v = val[:-1]
        try:
            return float(v) / 100.0
        except Exception:
            return None

    return None


def numeric_equal(a: float, b: float, tol=1e-4) -> bool:
    return isclose(a, b, rel_tol=tol)


def numeric_match_with_percentage(pred_s: str, ref_s: str, allow_percentage=True) -> bool:
    p = parse_numeric_value(pred_s)
    r = parse_numeric_value(ref_s)
    if p is None or r is None:
        return False

    if numeric_equal(p, r):
        return True

    if allow_percentage:
        return numeric_equal(p, r / 100.0) or numeric_equal(p, r * 100.0)

    return False

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
    s = s.replace("$", "")

    # remove latex spacing / sizing helpers
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("\\:", "")

    # normalize common glyphs / escapes
    s = s.replace("\\%", "%")
    s = s.replace("π", "pi")
    s = s.replace("\\pi", "pi")
    s = s.replace("\\dfrac", "\\frac")

    # unescape JSON-ish doubled backslashes
    s = s.replace("\\\\", "\\")

    # convert \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", r"(\1)/(\2)", s)

    # convert simple unbraced \frac a b -> (a)/(b)
    s = re.sub(r"\\frac\s+([^\s\{]+)\s+([^\s\{]+)", r"(\1)/(\2)", s)

    # convert leftover bare frac patterns after slash stripping
    s = re.sub(r"frac\{?\s*([^\s\{\}/()]+)\s*\}?\{?\s*([^\s\{\}/()]+)\s*\}?", r"\1/\2", s)

    # convert \sqrt{...} -> sqrt(...)
    s = re.sub(r"\\sqrt\s*\{\s*([^{}]+?)\s*\}", r"sqrt(\1)", s)

    # matrix/pmatrix -> tuple-like form
    s = s.replace("\\begin{pmatrix}", "(").replace("\\end{pmatrix}", ")")
    s = s.replace("\\begin{matrix}", "(").replace("\\end{matrix}", ")")
    s = s.replace("\\\\", ",")

    # normalize simple text wrappers again if they survived
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\bdegrees?\b", "", s, flags=re.IGNORECASE)

    # normalize single choice forms like (C) -> C
    m_choice = re.match(r"^\(?\s*([A-Za-z])\s*\)?$", s.strip())
    if m_choice:
        s = m_choice.group(1)

    # strip one outer pair of parens if whole thing is wrapped
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        d = 0
        ok = True
        for ch in inner:
            if ch == "(":
                d += 1
            elif ch == ")":
                d -= 1
                if d < 0:
                    ok = False
                    break
        if ok and d == 0:
            s = inner

    # remove parens around top-level fractions: (a)/(b) -> a/b
    s = _normalize_top_level_frac(s)
    s = re.sub(r"\(([^()]+)\)/\(([^()]+)\)", r"\1/\2", s)
    s = re.sub(r"\(([^()]+)\)/", r"\1/", s)
    s = re.sub(r"/\(([^()]+)\)", r"/\1", s)

    # permit some lenient tuple-ish cases: "1 -2" or "1-2" -> "1,-2"
    s = re.sub(r"(?<=\d)-(?=\d)", ",-", s)
    s = re.sub(r"(?<=\d)\s+(?=-?\d)", ",", s)

    # implied multiplication
    s = re.sub(r"(?<=\d)(?=sqrt\()", "*", s)
    s = re.sub(r"(?<=\d)(?=\()", "*", s)

    # normalize plain ^circ text
    s = s.replace("^circ", r"^\circ")

    # remove stray backslashes except \circ
    s = re.sub(r"\\(?!circ)", "", s)

    # if answer is something like "42cm", keep 42
    s0 = s.strip()
    m = re.match(r"^([-+]?\d+(?:\.\d+)?)(?:\s*[a-zA-Z][a-zA-Z\s/\-\^]*)$", s0)
    if m:
        s = m.group(1)

    # whitespace + punctuation cleanup
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[,\s]*,[,\s]*", ",", s)
    s = re.sub(r"[<>]+$", "", s)
    s = re.sub(r"[\.\s,;:]+$", "", s)
    s = s.strip(" ;,:\n\t")

    # mild brace cleanup
    s = re.sub(r"\{([a-zA-Z0-9\\]+)\}", r"\1", s)
    s = re.sub(r"[\.，,;:]+$", "", s)

    return s


def answers_match(pred: str, ref: str) -> bool:
    p = normalize_math_str(pred)
    r = normalize_math_str(ref)

    if not p or not r:
        return False

    # exact after normalization
    if p.lower() == r.lower():
        return True

    # degree-insensitive path
    p_no_deg = strip_degree_if_present(p)
    r_no_deg = strip_degree_if_present(r)
    if (p != p_no_deg) or (r != r_no_deg):
        if p_no_deg.lower() == r_no_deg.lower():
            return True
        if numeric_match_with_percentage(p_no_deg, r_no_deg, allow_percentage=True):
            return True

    # numeric fast path
    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    pf = to_float(p)
    rf = to_float(r)
    if pf is not None and rf is not None:
        return isclose(pf, rf, rel_tol=1e-4)

    # tuple path: (a,b)
    mp = PAIR_RE.match(p)
    mr = PAIR_RE.match(r)
    if mp and mr:
        p1, p2 = mp.group(1), mp.group(2)
        r1, r2 = mr.group(1), mr.group(2)
        return answers_match(p1, r1) and answers_match(p2, r2)

    # lenient comma-split fallback for tuple-ish outputs after normalization
    if "," in p and "," in r:
        ps = [x for x in p.split(",") if x != ""]
        rs = [x for x in r.split(",") if x != ""]
        if len(ps) == len(rs) and len(ps) > 1:
            if all(answers_match(a, b) for a, b in zip(ps, rs)):
                return True

    if numeric_match_with_percentage(p, r, allow_percentage=True):
        return True

    # symbolic fallback
    if _sym_equal(p, r):
        return True

    # final ultra-lenient fallback:
    # compare after removing commas/parentheses if both become same
    p2 = re.sub(r"[(),]", "", p).lower()
    r2 = re.sub(r"[(),]", "", r).lower()
    if p2 and r2 and p2 == r2:
        return True

    return False


def _truncate_for_embedding(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

def _spectral_mode_count(vectors: List[List[float]]) -> float:
    """
    Effective semantic mode-count M = exp(H(pi)),
    where pi is normalized eigen-spectrum of cosine Gram matrix.
    """
    if vectors is None or len(vectors) == 0:
        return 1.0
    if len(vectors) == 1:
        return 1.0

    X = np.asarray(vectors, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return 1.0

    # l2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    Xn = X / norms

    # cosine Gram
    G = Xn @ Xn.T

    # eigvals
    evals = np.linalg.eigvalsh(G)
    evals = np.clip(evals, 0.0, None)

    s = float(evals.sum())
    if s <= 1e-12:
        return 1.0

    pi = evals / s
    pi = pi[pi > 1e-12]
    if len(pi) == 0:
        return 1.0

    H = -np.sum(pi * np.log(pi))
    M = float(np.exp(H))
    return max(1.0, min(float(len(vectors)), M))

def _compute_probe_mode_count(embedder, texts: List[str], max_chars: int) -> float:
    if embedder is None or texts is None or len(texts) == 0:
        return 1.0

    clean_texts = [_truncate_for_embedding(t, max_chars) for t in texts]
    try:
        emb_res = embedder.embed(clean_texts)
        return _spectral_mode_count(emb_res.vectors)
    except Exception:
        return 1.0

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
    times_expanded: int = 0
    is_terminal_candidate: bool = False

def _path_ids(nodes: Dict[str, Node], leaf_id: str) -> List[str]:

    """Helper to trace back the path from leaf to root for token counting."""

    out = []

    cur = leaf_id

    while cur is not None and cur in nodes:

        out.append(cur)

        cur = nodes[cur].parent_id

    return list(reversed(out))






class ProblemState:

    """

    Encapsulates the context of a single problem being solved.

    Acts as the 'local variables' from my previous single-problem loop.

    """

    def __init__(self, p_data: Dict, cfg: Dict, log_f, agg_f, agg_stats, verifier=None):

        self.p_data = p_data

        self.id = p_data['id']

        self.question = p_data['question']

        self.gold_answer = p_data['answer'] # Adjust key if your json uses 'gold'
        self.log_f = log_f

        self.agg_f = agg_f
        self.agg_stats = agg_stats
        self.verifier = verifier
        self.cfg = cfg

        

        # Config Extraction

        dcfg = cfg["dpts_config"]

        self.tree_width = int(dcfg["tree_width"])

        self.tree_depth = int(dcfg["tree_depth"])

        self.max_rollout_iters = int(dcfg["max_rollout"])

        self.max_step_time_s = int(dcfg["max_step_time"])

        self.max_tokens_global = int(dcfg["max_new_tokens"])

        self.voting_method = dcfg.get("voting_method", "all")
        self.max_candidates = int(dcfg.get("max_candidates", 15))

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
        self.post_candidate_branch_width = int(dcfg.get("post_candidate_branch_width", 4))
        self.max_expansions_per_node = int(dcfg.get("max_expansions_per_node", 10))
        self.post_candidate_sampling_enabled = bool(dcfg.get("post_candidate_sampling_enabled", False))
        self.post_candidate_sampling_top_m = int(dcfg.get("post_candidate_sampling_top_m", 5))
        self.post_candidate_sampling_tau = float(dcfg.get("post_candidate_sampling_tau", 0.3))
        self.locality_penalty_enabled = bool(dcfg.get("locality_penalty_enabled", False))
        self.locality_penalty_lambda = float(dcfg.get("locality_penalty_lambda", 0.3))
        self.locality_penalty_anchor_depth = int(dcfg.get("locality_penalty_anchor_depth", 3))
        self.locality_penalty_recent_k = int(dcfg.get("locality_penalty_recent_k", 1))
        self.recent_candidate_ids = []

        self.depth_guardrail_enabled = bool(dcfg.get("depth_guardrail_enabled", False))
        self.depth_guardrail_delta = int(dcfg.get("depth_guardrail_delta", 2))
        self.depth_guardrail_use_first_candidate = bool(dcfg.get("depth_guardrail_use_first_candidate", True))
        self.pending_escape_from_candidate = False
        self.escape_candidate_id = None

        # Per-iter instrumentation accumulators
        self._last_popped_parent_depths: List[int] = []
        self._expanded_children_this_iter: int = 0
        self.candidate_first_found_depth = None

        self.adaptive_locality_enabled = bool(dcfg.get("adaptive_locality_enabled", False))
        self.adaptive_locality_lambda_init = float(dcfg.get("adaptive_locality_lambda_init", 0.2))
        self.adaptive_locality_lambda_min = float(dcfg.get("adaptive_locality_lambda_min", 0.0))
        self.adaptive_locality_lambda_max = float(dcfg.get("adaptive_locality_lambda_max", 1.0))
        self.adaptive_locality_eta_up = float(dcfg.get("adaptive_locality_eta_up", 0.1))
        self.adaptive_locality_eta_down = float(dcfg.get("adaptive_locality_eta_down", 0.05))
        self.adaptive_locality_overlap_target = float(dcfg.get("adaptive_locality_overlap_target", 0.8))

        self.adaptive_locality_lambda_t = self.adaptive_locality_lambda_init
        self.latest_candidate_id = None
        self.latest_selected_overlap = None     
        self.adaptive_locality_top_m = int(dcfg.get("adaptive_locality_top_m", 10))

        self.adaptive_depth_enabled = bool(dcfg.get("adaptive_depth_enabled", False))
        self.adaptive_depth_mu_init = float(dcfg.get("adaptive_depth_mu_init", 0.2))
        self.adaptive_depth_mu_min = float(dcfg.get("adaptive_depth_mu_min", 0.0))
        self.adaptive_depth_mu_max = float(dcfg.get("adaptive_depth_mu_max", 1.0))
        self.adaptive_depth_eta_up = float(dcfg.get("adaptive_depth_eta_up", 0.1))
        self.adaptive_depth_eta_down = float(dcfg.get("adaptive_depth_eta_down", 0.05))

        self.adaptive_depth_mu_t = self.adaptive_depth_mu_init
        self.candidate_first_found_depth = None
        self.latest_selected_depth_term = None
        self.adaptive_depth_use_deadband = bool(dcfg.get("adaptive_depth_use_deadband", True))

        # Adaptive branching knobs
        # -------------------------------
        self.adaptive_branching_enabled = bool(dcfg.get("adaptive_branching_enabled", False))
        self.adaptive_branch_probe_m0 = int(dcfg.get("adaptive_branch_probe_m0", 2))
        self.adaptive_branch_mmax = int(dcfg.get("adaptive_branch_mmax", 4))
        self.adaptive_branch_tau = float(dcfg.get("adaptive_branch_tau", 1.5))
        self.adaptive_branch_top_r = int(dcfg.get("adaptive_branch_top_r", 2))
        self.adaptive_branch_embed_truncate_chars = int(dcfg.get("adaptive_branch_embed_truncate_chars", 800))


        self.early_stop_dpts_style_enabled = bool(dcfg.get("early_stop_dpts_style_enabled", False))
        self.early_stop_t_star = int(dcfg.get("early_stop_t_star", 5))
        self.early_stop_lambda_es = float(dcfg.get("early_stop_lambda_es", 0.8))

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

                "Please reason step by step, and ensure that the final answer includes the correct unit .Output format MUST be exactly one line at the end. Put your final answer on its own line between tags like this:\n <<FINAL>> <answer> <</FINAL>>  .For example : <<FINAL>> 90 <</FINAL>> \n Do not output anything after that final line.\n\n"

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

        eff_depth = depth
        if self.depth_bonus_cap is not None:
            eff_depth = min(depth, self.depth_bonus_cap)

        # Before first candidate, compute alpha adaptively from frontier
        if not self.candidate_found_yet:
            alpha = self._adaptive_alpha_depth()
        else:
            alpha = self.alpha_depth  # usually irrelevant if mode is until_first_candidate

        return alpha * eff_depth

    def _adaptive_alpha_depth(self) -> float:
        """
        Compute adaptive alpha_depth from current frontier PRM/effective scores
        before first candidate is found:
            alpha_t = clip(max_score - median_score, 0.1, 1.0)
        Uses whole frontier.
        """
        if not self.frontier:
            return 0.1

        scores = []
        seen = set()

        for item in self.frontier:
            try:
                nid = item[3]
            except Exception:
                continue
            if nid in seen:
                continue
            seen.add(nid)

            if nid not in self.nodes:
                continue
            node = self.nodes[nid]

            if getattr(node, "is_terminal_candidate", False):
                continue
            if node.depth >= self.tree_depth:
                continue
            if getattr(node, "times_expanded", 0) >= self.max_expansions_per_node:
                continue

            s = node.effective_score if (node.effective_score is not None) else node.prm_score
            if s is None or not math.isfinite(s):
                continue
            scores.append(float(s))

        if not scores:
            return 0.1

        s_max = max(scores)
        s_med = float(np.median(scores))
        alpha_t = s_max - s_med
        alpha_t = max(0.1, min(1.0, alpha_t))
        return alpha_t

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

    def _sample_parent_from_top_m(self, width: int) -> List["Node"]:
        """
        After first candidate: sample up to `width` parents from the top-M frontier items
        using softmax over current heap priority score.
        Assumes frontier items are tuples like (priority_key, tie1, tie2, node_id),
        where lower priority_key means better node because heap stores -score.
        """
        if not self.frontier:
            return []

        top_m = max(1, self.post_candidate_sampling_top_m)
        tau = max(1e-6, self.post_candidate_sampling_tau)

        # pull a small prefix from heap
        popped = []
        for _ in range(min(top_m, len(self.frontier))):
            item = heapq.heappop(self.frontier)
            popped.append(item)

        # filter eligible nodes
        eligible = []
        requeue_items = []
        for item in popped:
            _, _, _, nid = item
            node = self.nodes[nid]

            if getattr(node, "is_terminal_candidate", False):
                continue
            if node.depth >= self.tree_depth:
                continue
            if getattr(node, "times_expanded", 0) >= self.max_expansions_per_node:
                continue
            if not self._passes_depth_guardrail(node):
                continue

            # recover score from heap key: heap stores priority_key = -effective_score_like
            priority_key = item[0]
            try:
                score = -float(priority_key)
            except Exception:
                continue

            if not np.isfinite(score):
                continue
            score = score - self._locality_penalty(node)
            eligible.append((item, node, score))
            requeue_items.append(item)

        # if nothing eligible, push everything back and return []
        if not eligible:
            return []

        selected_nodes = []
        selected_ids = set()

        # sample without replacement up to `width`
        for _ in range(min(width, len(eligible))):
            scores = np.array([x[2] for x in eligible], dtype=float)

            finite_mask = np.isfinite(scores)
            if not finite_mask.all():
                eligible = [x for x, keep in zip(eligible, finite_mask) if keep]
                if not eligible:
                    break
                scores = np.array([x[2] for x in eligible], dtype=float)

            # softmax with stabilization
            z = (scores - scores.max()) / tau
            probs = np.exp(z)
            probs_sum = probs.sum()

            if not np.isfinite(probs_sum) or probs_sum <= 0:
                idx = 0
            else:
                probs = probs / probs_sum
                idx = np.random.choice(len(eligible), p=probs)



            chosen_item, chosen_node, _ = eligible.pop(idx)

            selected_nodes.append(chosen_node)
            selected_ids.add(chosen_node.node_id)

        # push back everything not selected
        for item in popped:
            _, _, _, nid = item
            if nid not in selected_ids:
                heapq.heappush(self.frontier, item)


        return selected_nodes

    def check_budgets(self) -> bool:

        """Returns True if problem should stop."""

        if self.finished: return True

        

        if (time.time() - self.t_start) > self.max_step_time_s:

            self.stop_reason = "budget_time"

            return True

        if self.total_generated_tokens >= self.max_tokens_global:

            self.stop_reason = "budget_tokens"

            return True

        #if self.iter >= self.max_rollout_iters:

            #self.stop_reason = "budget_rollout"

            #return True

        if not self.frontier:

            self.stop_reason = "frontier_empty"

            return True

        if self.stop_reason == "early_stop":
            return True

        #if len(self.cand_ids) >= self.max_candidates:
            #self.stop_reason = "budget_candidates"
            #return True

        return False

    def _ancestor_at_depth(self, node: Node, target_depth: int):
        cur = node
        while cur is not None and cur.depth > target_depth:
            pid = cur.parent_id
            if pid is None or pid not in self.nodes:
                return None
            cur = self.nodes[pid]
        if cur is None:
            return None
        return cur if cur.depth == target_depth else None

    def _same_subtree_at_depth(self, node_a: Node, node_b: Node, anchor_depth: int) -> bool:
        anc_a = self._ancestor_at_depth(node_a, anchor_depth)
        anc_b = self._ancestor_at_depth(node_b, anchor_depth)
        if anc_a is None or anc_b is None:
            return False
        return anc_a.node_id == anc_b.node_id

    def _locality_penalty(self, node: Node) -> float:
        if not self.locality_penalty_enabled:
            return 0.0
        if not self.candidate_found_yet:
            return 0.0
        if not self.recent_candidate_ids:
            return 0.0
        if self.escape_candidate_id is None:
            return 0.0
        if not self.pending_escape_from_candidate:
            return 0.0
        if self.escape_candidate_id not in self.nodes:
            return 0.0

        penalty = 0.0
        for cid in self.recent_candidate_ids:
            if cid not in self.nodes:
                continue
            cand_node = self.nodes[self.escape_candidate_id]
            if self._same_subtree_at_depth(node, cand_node, self.locality_penalty_anchor_depth):
                penalty = max(penalty, self.locality_penalty_lambda)

        return penalty


    def _passes_depth_guardrail(self, node: Node) -> bool:
        if not self.depth_guardrail_enabled:
            return True
        if not self.candidate_found_yet:
            return True
        if self.candidate_first_found_depth is None:
            return True

        d_ref = self.candidate_first_found_depth if self.depth_guardrail_use_first_candidate else 0
        d_min = max(0, d_ref - self.depth_guardrail_delta)
        return node.depth >= d_min


    def get_parents_to_expand(self) -> List[Node]:
        parents_to_expand = []
        parent_ids_log = []

        #width = 4 if not self.candidate_found_yet else self.post_candidate_branch_width
        width = 1

        # pre-candidate: keep old behavior
        if not self.candidate_found_yet:
            for _ in range(width):
                if self.frontier:
                    _, _, _, nid = heapq.heappop(self.frontier)
                    node = self.nodes[nid]

                    if getattr(node, "is_terminal_candidate", False):
                        continue
                    if node.depth >= self.tree_depth:
                        continue
                    if getattr(node, "times_expanded", 0) >= self.max_expansions_per_node:
                        continue

                    parents_to_expand.append(node)
                    parent_ids_log.append(nid)

        else:
            # post-candidate: inspect top-M, recompute score with adaptive locality, pick best
            top_m = max(1, self.adaptive_locality_top_m)
            popped = []

            for _ in range(min(top_m, len(self.frontier))):
                popped.append(heapq.heappop(self.frontier))

            eligible = []
            for item in popped:
                _, _, _, nid = item
                node = self.nodes[nid]

                if getattr(node, "is_terminal_candidate", False):
                    continue
                if node.depth >= self.tree_depth:
                    continue
                if getattr(node, "times_expanded", 0) >= self.max_expansions_per_node:
                    continue

                score, locality_penalty, depth_term = self._selection_score_with_adaptive_controls(node)
                if not math.isfinite(score):
                    continue
                eligible.append((score, locality_penalty, depth_term, node, item))

            if eligible:
                # choose top `width` adjusted-score nodes
                eligible.sort(key=lambda x: x[0], reverse=True)
                chosen = eligible[:width]
                unchosen = eligible[width:]

                chosen_ids = set()

                for chosen_score, chosen_locality_penalty, chosen_depth_term, chosen_node, chosen_item in chosen:
                    parents_to_expand.append(chosen_node)
                    parent_ids_log.append(chosen_node.node_id)
                    chosen_ids.add(chosen_node.node_id)

                # log overlap/depth term using the first chosen node only
                if self.latest_candidate_id is not None and self.latest_candidate_id in self.nodes and chosen:
                    cand_node = self.nodes[self.latest_candidate_id]
                    first_chosen_node = chosen[0][3]
                    self.latest_selected_overlap = self._shared_path_fraction(first_chosen_node, cand_node)
                    self.latest_selected_depth_term = chosen[0][2]
                else:
                    self.latest_selected_overlap = None
                    self.latest_selected_depth_term = None

                # push back all non-chosen popped items
                for score, locality_penalty, depth_term, node, item in unchosen:
                    heapq.heappush(self.frontier, item)

            else:
                self.latest_selected_overlap = None
                self.latest_selected_depth_term = None

        self._last_popped_parent_depths = [p.depth for p in parents_to_expand]
        self._expanded_children_this_iter = 0

        # adaptive lambda update AFTER a post-candidate parent has been chosen
        if self.candidate_found_yet and self.adaptive_depth_enabled and parents_to_expand:
            selected_node = parents_to_expand[0]   # use first selected parent as controller signal

            if self.adaptive_depth_use_deadband and self.candidate_first_found_depth is not None:
                d_ref, pivot = self._adaptive_depth_pivot()
                d_sel = float(selected_node.depth)

                if d_sel < pivot:
                    # too shallow -> push harder
                    self.adaptive_depth_mu_t = min(
                        self.adaptive_depth_mu_max,
                        self.adaptive_depth_mu_t + self.adaptive_depth_eta_up
                    )
                elif d_sel <= d_ref:
                    # acceptable band -> relax slowly
                    self.adaptive_depth_mu_t = max(
                        self.adaptive_depth_mu_min,
                        self.adaptive_depth_mu_t - 0.5 * self.adaptive_depth_eta_down
                    )
                else:
                    # deeper than horizon -> relax faster
                    self.adaptive_depth_mu_t = max(
                        self.adaptive_depth_mu_min,
                        self.adaptive_depth_mu_t - self.adaptive_depth_eta_down
                    )

            else:
                dt = self.latest_selected_depth_term
                if dt is not None and math.isfinite(dt):
                    if dt < 0:
                        self.adaptive_depth_mu_t = min(
                            self.adaptive_depth_mu_max,
                            self.adaptive_depth_mu_t + self.adaptive_depth_eta_up
                        )
                    else:
                        self.adaptive_depth_mu_t = max(
                            self.adaptive_depth_mu_min,
                            self.adaptive_depth_mu_t - self.adaptive_depth_eta_down
                        )

        if self.candidate_found_yet and self.adaptive_locality_enabled and parents_to_expand:
            ov = self.latest_selected_overlap
            if ov is not None and math.isfinite(ov):
                if ov > self.adaptive_locality_overlap_target:
                    self.adaptive_locality_lambda_t = min(
                            self.adaptive_locality_lambda_max,
                            self.adaptive_locality_lambda_t + self.adaptive_locality_eta_up
                        )
                else:
                    self.adaptive_locality_lambda_t = max(
                            self.adaptive_locality_lambda_min,
                            self.adaptive_locality_lambda_t - self.adaptive_locality_eta_down
                        )
    

        if not parents_to_expand:
            return []

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
            "adaptive_locality_enabled": self.adaptive_locality_enabled,
            "adaptive_locality_lambda_t": self.adaptive_locality_lambda_t,
            "adaptive_locality_overlap_target": self.adaptive_locality_overlap_target,
            "latest_selected_overlap": self.latest_selected_overlap,
            "adaptive_depth_enabled": self.adaptive_depth_enabled,
            "adaptive_depth_mu_t": self.adaptive_depth_mu_t,
            "candidate_first_found_depth": self.candidate_first_found_depth,
            "latest_selected_depth_term": self.latest_selected_depth_term,
        }
        self.log_f.flush()
        self.log_f.write(json.dumps(log_entry) + "\n")

        return parents_to_expand




    def _path_to_root_ids(self, node: Node):
        path = []
        cur = node
        seen = set()

        while cur is not None and cur.node_id not in seen:
            seen.add(cur.node_id)
            path.append(cur.node_id)

            pid = cur.parent_id
            if pid is None or pid not in self.nodes:
                break
            cur = self.nodes[pid]

        return list(reversed(path))

    def _lca_depth(self, node_a: Node, node_b: Node) -> float:
        path_a = self._path_to_root_ids(node_a)
        path_b = self._path_to_root_ids(node_b)

        if not path_a or not path_b:
            return float("nan")

        lca_id = None
        for a, b in zip(path_a, path_b):
            if a == b:
                lca_id = a
            else:
                break

        if lca_id is None or lca_id not in self.nodes:
            return float("nan")

        d = self.nodes[lca_id].depth
        return float(d) if d is not None else float("nan")

    
    def _shared_path_fraction(self, node: Node, cand_node: Node) -> float:
        lca_d = self._lca_depth(node, cand_node)
        if not math.isfinite(lca_d):
            return 0.0

        denom = min(node.depth, cand_node.depth)
        if denom is None or denom <= 0:
            return 0.0

        frac = float(lca_d) / float(denom)
        if not math.isfinite(frac):
            return 0.0

        return max(0.0, min(1.0, frac))

    def _adaptive_locality_penalty(self, node: Node) -> float:
        if not self.adaptive_locality_enabled:
            return 0.0
        if not self.candidate_found_yet:
            return 0.0
        if self.latest_candidate_id is None:
            return 0.0
        if self.latest_candidate_id not in self.nodes:
            return 0.0

        cand_node = self.nodes[self.latest_candidate_id]
        overlap = self._shared_path_fraction(node, cand_node)
        return self.adaptive_locality_lambda_t * overlap

    def _selection_score_with_adaptive_controls(self, node: Node):
        base_score = node.effective_score if (node.effective_score is not None) else node.prm_score
        if base_score is None or not math.isfinite(base_score):
            base_score = -1e9

        # keep old pre-candidate depth bonus before first candidate
        score = float(base_score) + self._depth_bonus_value(node.depth)

        locality_penalty = 0.0
        depth_term = 0.0

        if self.adaptive_locality_enabled and self.candidate_found_yet:
            locality_penalty = self._adaptive_locality_penalty(node)
            score -= locality_penalty

        if self.adaptive_depth_enabled and self.candidate_found_yet:
            depth_term = self._adaptive_depth_term(node)
            score += self.adaptive_depth_mu_t * depth_term

        return score, locality_penalty, depth_term

    def _adaptive_depth_term(self, node: Node) -> float:
        if not self.adaptive_depth_enabled:
            return 0.0
        if not self.candidate_found_yet:
            return 0.0
        if self.candidate_first_found_depth is None:
            return 0.0

        d_ref = float(self.candidate_first_found_depth)
        s_t = max(1.0, 0.25 * d_ref)

        val = math.tanh((float(node.depth) - d_ref) / s_t)
        if not math.isfinite(val):
            return 0.0
        return val

    def _adaptive_depth_pivot(self):
        if self.candidate_first_found_depth is None:
            return None, None

        d_ref = float(self.candidate_first_found_depth)
        delta = max(1, int(math.floor(math.log(d_ref + 1.0))))
        pivot = d_ref - delta
        return d_ref, pivot
    def should_stop_after_new_candidate(self, candidate_node) -> bool:
        if not self.early_stop_dpts_style_enabled:
            return False

        if len(self.cand_ids) < self.early_stop_t_star:
            return False

        ref_scores = [
            n.prm_score for n in self.nodes.values()
            if getattr(n, "times_expanded", 0) > 0 and n.prm_score is not None and math.isfinite(n.prm_score)
        ]

        if not ref_scores or candidate_node.prm_score is None or not math.isfinite(candidate_node.prm_score):
            return False

        t = len(self.cand_ids)

        if t <= self.early_stop_t_star:
            theta_es = self.early_stop_lambda_es * statistics.mean(ref_scores)
        else:
            theta_es = max(ref_scores)

        if candidate_node.prm_score < theta_es:
            self.stop_reason = "early_stop"
            return True

        return False

    

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
        #verifier logic 




        triples = [(cid, txt, ans) for cid, txt, ans in zip(self.cand_ids, cand_texts, cand_answers) if ans]
        vcfg = self.cfg["dpts_config"].get("verifier", {}) or {}
        use_verifier = bool(vcfg.get("enabled", False)) and (self.verifier is not None) and bool(self.cand_ids)
        verifier_debug = None
        final_answer = ""
        final_text = ""
        chosen_id = None

        if use_verifier:
            topk_groups = int(vcfg.get("topk_groups", 6))
            reps_per_group = int(vcfg.get("reps_per_group", 2))
            samples = int(vcfg.get("samples_per_candidate", 5))
            alpha_vote = float(vcfg.get("alpha_vote", 0.5))  # 0 => verifier only, 1 => majority only
            # Group candidates by extracted answer (skip empties)
            groups = {}  # ans -> list[cid]
            for cid in self.cand_ids:
                sol = self.nodes[cid].completion
                ans = extract_answer(sol)
                if not ans:
                    continue
                groups.setdefault(ans, []).append(cid)
            if not groups:
                use_verifier = False
            else:
                # top-K groups by frequency (majority prior)
                ranked_answers = sorted(groups.keys(), key=lambda a: len(groups[a]), reverse=True)[:topk_groups]
                max_freq = max(len(groups[a]) for a in ranked_answers)

                per_group = {}
                best = (-1e9, None, None)  # (combined_score, best_ans, best_cid)

                for ans in ranked_answers:
                    cids = groups[ans]
                    reps = sorted(cids, key=lambda cid: self.nodes[cid].prm_score, reverse=True)[:reps_per_group]
                    rep_meta = {}
                    rep_scores = []
                    for rcid in reps:
                        sol = self.nodes[rcid].completion
                        res = self.verifier.verify(self.question, ans, solution=sol[-2000:], samples=samples)
                        rep_meta[rcid] = {
                                "score": float(res["score"]),
                                "yes": res.get("yes", None),
                                "no": res.get("no", None),
                                "parsed": res.get("parsed", None),
                                "samples": res.get("samples", None),
                                "fallback_used": res.get("fallback_used", None),
                            }
                        rep_scores.append(float(res["score"]))

                    ver_score = sum(rep_scores) / max(1, len(rep_scores))
                    vote_score = len(cids) / max(1, max_freq)  # normalized majority in [0,1]
                    combined = alpha_vote * vote_score + (1.0 - alpha_vote) * ver_score
                    # choose best representative for this answer group
                    best_rep = max(reps, key=lambda cid: (rep_meta[cid]["score"], self.nodes[cid].prm_score))
                    per_group[ans] = {
                            "freq": len(cids),
                            "vote_score": vote_score,
                            "ver_score": ver_score,
                            "combined": combined,
                            "reps": reps,
                            "per_rep": rep_meta,
                            "chosen_rep": best_rep,
                        }
                    if combined > best[0]:
                        best = (combined, ans, best_rep)
                _, best_ans, best_cid = best
                chosen_id = best_cid
                final_text = self.nodes[best_cid].completion
                final_answer = best_ans  # group key is already extracted answer
                verifier_debug = {
                        "used": True,
                        "topk_groups": topk_groups,
                        "reps_per_group": reps_per_group,
                        "samples_per_candidate": samples,
                        "alpha_vote": alpha_vote,
                        "best_cid": chosen_id,
                        "best_answer": final_answer,
                        "per_group": per_group,
                    }
                if not final_answer:
                    use_verifier = False

        # --- fallback to your original logic if verifier disabled/failed ---
        if not use_verifier:
            if triples:
                from collections import Counter
                norm_map = []  # (cid, txt, raw_ans, norm_ans)
                for cid, txt, ans in triples:
                    norm_ans = normalize_math_str(ans)
                    norm_map.append((cid, txt, ans, norm_ans))

                cnt = Counter(norm_ans for _, _, _, norm_ans in norm_map)
                voted_norm = cnt.most_common(1)[0][0]

                final_answer = voted_norm
                for cid, txt, ans, norm_ans in norm_map:
                    if norm_ans == voted_norm:
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

        self.verifier_debug = verifier_debug
        self.final_method = "genprm_verifier" if use_verifier else self.voting_method


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
        self.final_method = "genprm_verifier" if use_verifier else self.voting_method
        self.verifier_debug = verifier_debug


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
            
            "final_method": getattr(self, "final_method", self.voting_method),

            "final_answer":final_answer,
        
            "gold_answer_clean": gold_cleaned,

            "is_correct": bool(is_correct),

            "total_generated_tokens": int(self.total_generated_tokens),

            "tokens_kept_in_final_path": int(tokens_kept),

            "tokens_pruned": int(tokens_pruned),

            "time_total_s": float(time.time() - self.t_start),

            "time_llm_forward_s": float(self.total_llm_time),

            "time_prm_s": float(self.total_prm_time),
            "final_method": self.final_method,
            "verifier": getattr(self, "verifier_debug", None),

        }
        self.log_f.flush()

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
    verifier=None,
    embedder=None,


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

            active_states.append(ProblemState(p, cfg, log_f,agg_f,agg_stats,verifier=verifier))

    

    refill_batch()

    

    dcfg = cfg["dpts_config"]

    lcfg = cfg["llm_config"]
    nll_helper = NLLPriorityHelper(dcfg)

    branch_factor = int(dcfg.get("num_branch", 4))

    max_new_tokens=int(lcfg["max_new_tokens"])
    pre_candidate_branch_width = int(dcfg.get("num_branch", 4))

    post_candidate_branch_width = int(dcfg.get("post_candidate_branch_width", 4))

    
    return_logprobs = nll_helper.want_logprobs()
    # Sampling params (shared)

    sampling = dict(

        temperature=float(lcfg["temperature"]),

        top_p=float(lcfg["top_p"]),

        max_new_tokens=int(lcfg["max_new_tokens"]),
        max_tokens=int(lcfg["max_new_tokens"]),
        return_logprobs=return_logprobs,
        logprobs_k=1,

        n=pre_candidate_branch_width,

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

                phase_branch_width = (
                        post_candidate_branch_width
                        if any(state.candidate_found_yet for state, _ in request_map)
                        else pre_candidate_branch_width
                    )
                if state.adaptive_branching_enabled:
                    n_req = state.adaptive_branch_probe_m0
                else:
                    n_req = phase_branch_width
                sampling["n"] = n_req
            


                gen_results = backend.generate_batch(batch_prompts, sampling=sampling)
                any_lp = any((getattr(gr, "token_logprobs", None) is not None) for gr in gen_results) if gen_results else False
                nll_helper.ensure_or_fallback(any_lp)
                # --- ADAPTIVE BRANCHING: batched extra generation for selected parents ---
                probe_mode_counts = [1.0] * len(gen_results)
                extra_needed_per_req = [0] * len(gen_results)
                extra_indices = []

                if gen_results:
                    for req_idx, res in enumerate(gen_results):
                        state_i, parent_node_i = request_map[req_idx]

                        if not state_i.adaptive_branching_enabled:
                            continue

                        probe_texts_i = list(res.texts)
                        try:
                            probe_mode_count_i = _compute_probe_mode_count(
                                embedder=embedder,
                                texts=probe_texts_i,
                                max_chars=state_i.adaptive_branch_embed_truncate_chars,
                            )
                        except Exception as e:
                            #print(f"[DEBUG] probe_mode_count failed for parent={parent_node_i.node_id}: {e}")
                            probe_mode_count_i = 1.0

                        probe_mode_counts[req_idx] = probe_mode_count_i

                        if probe_mode_count_i >= state_i.adaptive_branch_tau:
                            extra_needed_i = max(0, state_i.adaptive_branch_mmax - len(probe_texts_i))
                            if extra_needed_i > 0:
                                extra_needed_per_req[req_idx] = extra_needed_i
                                extra_indices.append(req_idx)
                                #print(f"[DEBUG] adaptive extra candidates={len(extra_indices)} extra_indices={extra_indices}")

                # batched extra generation only if all selected parents need the same extra count
                # this is the usual case when probe size is fixed and mmax is fixed
                if extra_indices:
                    extra_counts = [extra_needed_per_req[i] for i in extra_indices]
                    same_extra_n = len(set(extra_counts)) == 1

                    if same_extra_n:
                        extra_n = extra_counts[0]
                        extra_prompts = [batch_prompts[i] for i in extra_indices]

                        #print(f"[DEBUG] batched extra generation: num_extra_parents={len(extra_indices)} extra_n={extra_n}")

                        extra_sampling = dict(sampling)
                        extra_sampling["n"] = extra_n
                        gen_results_extra = backend.generate_batch(extra_prompts, sampling=extra_sampling)

                        gen_results = list(gen_results)
                        for k, req_idx in enumerate(extra_indices):
                            probe_res = gen_results[req_idx]
                            extra_res = gen_results_extra[k]

                            merged_texts = list(probe_res.texts) + list(extra_res.texts)
                            merged_lp = None
                            merged_toks = None

                            if getattr(probe_res, "token_logprobs", None) is not None or getattr(extra_res, "token_logprobs", None) is not None:
                                merged_lp = list(getattr(probe_res, "token_logprobs", None) or []) + \
                                            list(getattr(extra_res, "token_logprobs", None) or [])

                            if getattr(probe_res, "tokens", None) is not None or getattr(extra_res, "tokens", None) is not None:
                                merged_toks = list(getattr(probe_res, "tokens", None) or []) + \
                                            list(getattr(extra_res, "tokens", None) or [])

                            merged_time = float(getattr(probe_res, "time_llm_forward_s", 0.0)) + \
                                        float(getattr(extra_res, "time_llm_forward_s", 0.0))

                            gen_results[req_idx] = GenResult(
                                texts=merged_texts,
                                time_llm_forward_s=merged_time,
                                token_logprobs=merged_lp,
                                tokens=merged_toks,
                            )
                            #print(f"[DEBUG] merged child counts={[len(r.texts) for r in gen_results]}")
                    else:
                        print(f"[DEBUG] adaptive branching skipped batched extra because extra_n differs across parents: {extra_counts}")

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
                parent_node.times_expanded += 1

                

                # Update LLM Time

                state.total_llm_time += res.time_llm_forward_s if hasattr(res, 'time_llm_forward_s') else avg_time

                probe_texts = list(res.texts)
                
                probe_mode_count = probe_mode_counts[req_idx] if req_idx < len(probe_mode_counts) else 1.0
                all_texts = list(probe_texts)
                


                final_mode_count = probe_mode_count
                if state.adaptive_branching_enabled:
                    final_mode_count = _compute_probe_mode_count(
                        embedder=embedder,
                        texts=all_texts,
                        max_chars=state.adaptive_branch_embed_truncate_chars,
                    )

                parent_child_records = []

                

                for j, child_text in enumerate(all_texts):
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

                     # Candidate trigger + state transition instrumentation
                    is_candidate = state._candidate_trigger(child_text)

                    rec = {
                        "node": new_node,
                        "is_candidate": is_candidate,
                        "prm_score": prm_score,
                        "effective_score": eff_score,
                        "nll_dbg": nll_dbg,
                        "child_text": child_text,
                    }
                    parent_child_records.append(rec)


                    

                    # Push to Frontier

                    #heapq.heappush(state.frontier, state.frontier_item_for_node(new_node))
                    
                    #is_candidate = state._candidate_trigger(solution_so_far, gen_toks, max_new_tokens)

                    if is_candidate:
                        new_node.is_terminal_candidate = True

                        state.cand_ids.append(nid)

                        state.latest_candidate_id = nid
                        state.recent_candidate_ids.append(nid)

                        if state.should_stop_after_new_candidate(new_node):
                            state.log_f.write(json.dumps({
                                "problem_id": state.id,
                                "event": "early_stop_triggered",
                                "iter": state.iter,
                                "node_id": nid,
                                "candidate_prm_score": new_node.prm_score,
                                "num_candidates_total": len(state.cand_ids),
                                "stop_reason": state.stop_reason,
                            }) + "\n")


                        state.pending_escape_from_candidate = True
                        state.escape_candidate_id = nid
                        if len(state.recent_candidate_ids) > state.locality_penalty_recent_k:
                            state.recent_candidate_ids = state.recent_candidate_ids[-state.locality_penalty_recent_k:]

                        if not state.candidate_found_yet:

                            state.candidate_found_yet = True

                            state.candidate_first_found_iter = state.iter
                            state.candidate_first_found_depth = new_node.depth

                            state.candidate_first_found_time = time.time()
                            fresh_frontier = []
                            state.push_seq = 0
                            seen = set()

                            for _, _, _, old_nid in state.frontier:
                                if old_nid in seen:
                                    continue
                                seen.add(old_nid)
                                node = state.nodes[old_nid]
                                if getattr(node, "is_terminal_candidate", False):
                                    continue
                                if node.depth >= state.tree_depth:
                                    continue
                                if getattr(node, "times_expanded", 0) >= state.max_expansions_per_node:
                                    continue
                                heapq.heappush(fresh_frontier, state.frontier_item_for_node(node))

                            state.frontier = fresh_frontier
            

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

                records_to_keep = parent_child_records
                if state.adaptive_branching_enabled and final_mode_count < state.adaptive_branch_tau:
                    nonterm = [r for r in parent_child_records if not r["is_candidate"]]
                    term = [r for r in parent_child_records if r["is_candidate"]]

                    nonterm.sort(
                        key=lambda r: (
                            r["effective_score"] if r["effective_score"] is not None else r["prm_score"]
                        ),
                        reverse=True
                    )

                    nonterm = nonterm[:state.adaptive_branch_top_r]
                    records_to_keep = term + nonterm

                for r in records_to_keep:
                    if not r["is_candidate"]:
                        heapq.heappush(state.frontier, state.frontier_item_for_node(r["node"]))


                if (not getattr(parent_node, "is_terminal_candidate", False)
                        and parent_node.depth < state.tree_depth
                        and parent_node.times_expanded < state.max_expansions_per_node):
                        heapq.heappush(state.frontier, state.frontier_item_for_node(parent_node))

                    

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

