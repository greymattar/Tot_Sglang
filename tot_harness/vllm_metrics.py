# tot_harness/vllm_metrics.py
import time
import requests
from typing import Dict, Any, Optional, Tuple
from typing import Dict, Any, Optional
def _parse_prometheus_text(text: str) -> Dict[str, float]:
    """
    Minimal Prometheus text parser:
    - Ignores comments (# HELP / # TYPE)
    - Ignores labeled series by default (we only keep the *exact* metric name as it appears)
      BUT for histograms we want buckets too, which are labeled. We'll store them under a synthetic key.
    Returns: mapping key -> value (float)
    """
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        # Example:
        # vllm:kv_cache_usage_perc 0.8125
        # vllm:time_to_first_token_seconds_bucket{le="0.5"} 123
        try:
            metric_and_labels, val_str = line.rsplit(" ", 1)
            val = float(val_str)
        except Exception:
            continue

        if "{" in metric_and_labels and metric_and_labels.endswith("}"):
            # labeled series: turn into a stable key
            name = metric_and_labels[: metric_and_labels.index("{")]
            labels = metric_and_labels[metric_and_labels.index("{") + 1 : -1]
            key = f"{name}{{{labels}}}"
            out[key] = val
        else:
            out[metric_and_labels] = val
    return out

class VLLMMetrics:
    """
    Fetches /metrics and lets you snapshot + compute deltas for selected metrics.
    """
    def __init__(self, metrics_url: str, timeout_s: float = 2.0):
        self.metrics_url = metrics_url
        self.timeout_s = timeout_s

    def snapshot(self) -> Tuple[float, Dict[str, float]]:
        t = time.time()
        r = requests.get(self.metrics_url, timeout=self.timeout_s)
        r.raise_for_status()
        data = _parse_prometheus_text(r.text)
        return t, data

def _sum_series(m: Dict[str, float], base: str) -> float:
    """Sum unlabeled `base` and all labeled `base{...}` series."""
    s = 0.0
    pref = base + "{"
    for k, v in m.items():
        if k == base or k.startswith(pref):
            s += v
    return s

def _max_series(m: Dict[str, float], base: str) -> Optional[float]:
    """Max over unlabeled `base` and all labeled `base{...}` series (useful for gauges)."""
    vals = []
    pref = base + "{"
    for k, v in m.items():
        if k == base or k.startswith(pref):
            vals.append(v)
    return max(vals) if vals else None

def get_counter_delta(pre: Dict[str, float], post: Dict[str, float], name: str) -> float:
    """Delta for a counter that may be labeled."""
    return _sum_series(post, name) - _sum_series(pre, name)

def get_gauge(post: Dict[str, float], name: str, agg: str = "max") -> Optional[float]:
    """
    Read a gauge that may be labeled. Aggregation:
      - "max" (default): good for usage perc across devices/engines
      - "sum": if you truly want totals
    """
    if agg == "sum":
        return _sum_series(post, name)
    return _max_series(post, name)

def _parse_labels(label_str: str) -> Dict[str, str]:
    # label_str like: engine="0",model_name="..."
    out = {}
    parts = [p.strip() for p in label_str.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out

def get_hist_delta(pre: Dict[str, float], post: Dict[str, float], base: str) -> Dict[str, Any]:
    """
    Histogram base name -> delta of count/sum and bucket deltas aggregated across labels.
    Buckets are aggregated by `le` across all label combinations.
    """
    count_name = base + "_count"
    sum_name = base + "_sum"
    d_count = _sum_series(post, count_name) - _sum_series(pre, count_name)
    d_sum = _sum_series(post, sum_name) - _sum_series(pre, sum_name)

    # buckets: keys look like base+"_bucket{le="0.5",engine="0",...}"
    bucket_prefix = base + "_bucket{"
    bucket_deltas_by_le: Dict[str, float] = {}

    for k, v_post in post.items():
        if not k.startswith(bucket_prefix):
            continue
        v_pre = pre.get(k, 0.0)
        delta = v_post - v_pre

        # extract label dict and pull 'le'
        label_str = k[k.index("{")+1 : -1]
        labels = _parse_labels(label_str)
        le = labels.get("le", None)
        if le is None:
            continue
        bucket_deltas_by_le[le] = bucket_deltas_by_le.get(le, 0.0) + delta

    return {"count": d_count, "sum": d_sum, "buckets_by_le": bucket_deltas_by_le}

def safe_mean(sum_delta: float, count_delta: float) -> Optional[float]:
    if count_delta <= 0:
        return None
    return float(sum_delta) / float(count_delta)

