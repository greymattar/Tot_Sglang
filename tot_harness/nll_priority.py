# tot_harness/nll_priority.py
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_MATH_CHARS = set("0123456789+-*/^=<>%()[]{}.,\\|:_")

def _is_mathy_token(tok: str) -> bool:
    t = (tok or "").strip()
    if not t:
        return False
    return any((ch.isdigit() or ch in _MATH_CHARS) for ch in t)

@dataclass
class WelfordState:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))

def _select_logprobs(
    tokens: Optional[List[str]],
    token_logprobs: Optional[List[Optional[float]]],
    mode: str,
) -> Tuple[Optional[List[float]], int]:
    if not token_logprobs or not isinstance(token_logprobs, list):
        return None, 0

    usable = [(i, lp) for i, lp in enumerate(token_logprobs) if lp is not None]
    if not usable:
        return None, 0

    if mode == "all_tokens" or tokens is None or not isinstance(tokens, list):
        sel = [lp for _, lp in usable]
        return sel, len(sel)

    # math_tokens
    math_sel: List[float] = []
    for i, lp in usable:
        tok = tokens[i] if i < len(tokens) else ""
        if _is_mathy_token(tok):
            math_sel.append(lp)

    if len(math_sel) < 3:
        sel = [lp for _, lp in usable]
        return sel, len(sel)

    return math_sel, len(math_sel)

class NLLPriorityHelper:
    """
    Backend-agnostic helper: given PRM score and per-token logprobs for the new completion,
    compute AvgNLL_step, online-normalize (Welford), and return effective score.

    effective_score = prm_score - lambda * z(AvgNLL_step)
    """
    def __init__(self, cfg_dpts: Dict[str, Any]):
        self.priority_mode = str(cfg_dpts.get("priority_mode", "prm_only"))
        self.nll_lambda = float(cfg_dpts.get("nll_lambda", 0.0))
        self.nll_norm_mode = str(cfg_dpts.get("nll_norm_mode", "welford_global"))
        self.nll_token_filter_mode = str(cfg_dpts.get("nll_token_filter_mode", "all_tokens"))
        self.logprobs_required = bool(cfg_dpts.get("logprobs_required", False))
        self.eps = float(cfg_dpts.get("nll_eps", 1e-6))

        self.welford_global = WelfordState()
        self.welford_per_problem: Dict[str, WelfordState] = {}
        self._fallback_logged = False

    def want_logprobs(self) -> bool:
        return (self.priority_mode == "prm_nll_hybrid" and self.nll_lambda != 0.0)

    def ensure_or_fallback(self, any_logprobs_available: bool) -> None:
        """
        Call once per batch (or early in run) to enforce logprobs_required / fallback behavior.
        """
        if not self.want_logprobs():
            return

        if any_logprobs_available:
            return

        if self.logprobs_required:
            raise RuntimeError("logprobs_required=True but backend did not return token_logprobs.")

        # fallback to prm_only for stability
        if not self._fallback_logged:
            print("[WARN] logprobs missing; falling back to priority_mode=prm_only")
            self._fallback_logged = True
        self.priority_mode = "prm_only"

    def compute_effective_score(
        self,
        *,
        problem_id: str,
        prm_score: float,
        token_logprobs: Optional[List[Optional[float]]],
        tokens: Optional[List[str]],
    ) -> Tuple[float, Dict[str, Any]]:
        dbg: Dict[str, Any] = {
            "prm_score": float(prm_score),
            "logprobs_available": bool(token_logprobs),
        }

        # Baseline behavior
        if self.priority_mode != "prm_nll_hybrid" or self.nll_lambda == 0.0:
            dbg["effective_score"] = float(prm_score)
            return float(prm_score), dbg

        # Missing logprobs -> baseline (unless required; enforced elsewhere)
        if token_logprobs is None:
            dbg.update({
                "effective_score": float(prm_score),
                "fallback_reason": "missing_logprobs",
            })
            return float(prm_score), dbg

        # Choose token set for scoring
        filt_mode = self.nll_token_filter_mode
        sel_lps, n_sel = _select_logprobs(tokens, token_logprobs, filt_mode)

        if sel_lps is None or n_sel == 0:
            dbg.update({
                "effective_score": float(prm_score),
                "fallback_reason": "no_usable_logprobs",
                "num_tokens_filtered": int(n_sel),
            })
            return float(prm_score), dbg

        # AvgNLL_step = -(1/L) * sum logp(token_i)
        avg_nll_filtered = -sum(sel_lps) / float(n_sel)
        dbg["avg_nll_filtered"] = float(avg_nll_filtered)
        dbg["num_tokens_filtered"] = int(n_sel)

        # Also report all-token AvgNLL for instrumentation
        all_lps, all_cnt = _select_logprobs(tokens, token_logprobs, "all_tokens")
        if all_lps is not None and all_cnt > 0:
            dbg["avg_nll_all_tokens"] = float(-sum(all_lps) / float(all_cnt))
            dbg["num_tokens_step"] = int(all_cnt)
        else:
            dbg["avg_nll_all_tokens"] = None
            dbg["num_tokens_step"] = 0

        # Pick normalization state
        if self.nll_norm_mode == "welford_per_problem":
            st = self.welford_per_problem.setdefault(problem_id, WelfordState())
        else:
            st = self.welford_global

        mu = st.mean
        sigma = st.std()
        nll_norm = 0.0 if st.count < 2 else (avg_nll_filtered - mu) / (sigma + self.eps)

        effective = float(prm_score) - float(self.nll_lambda) * float(nll_norm)

        dbg.update({
            "nll_norm": float(nll_norm),
            "nll_lambda": float(self.nll_lambda),
            "effective_score": float(effective),
            "welford_count": int(st.count),
            "welford_mean": float(mu),
            "welford_m2": float(st.m2),
            "welford_std": float(sigma),
        })

        # Update stats AFTER computing z-score
        st.update(avg_nll_filtered)

        return effective, dbg

