import re
import uuid
from collections import Counter
from typing import List

def extract_answer_boxed(text: str) -> str:
    # Your SSDP snippet: last \boxed{...}
    matches = re.findall(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', text)
    if matches:
        ans = matches[-1].strip()
        return ans if ans != "" else str(uuid.uuid1())
    return ""

def extract_answer_gsm8k(text: str) -> str:
    # GSM8K: "#### <number>"
    m = re.search(r"####\s*([-\d\.,]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    return ""

def extract_answer(text: str) -> str:
    # Prefer boxed, else GSM8K, else empty
    a = extract_answer_boxed(text)
    if a:
        return a
    a = extract_answer_gsm8k(text)
    if a:
        return a
    return ""

def agg_prm_min_max(x_list: List[str], ans_list: List[str], v_list: List[List[float]]) -> str:
    # v_list per candidate = list of step scores; take min per candidate, then max across candidates
    vals = [min(v) if v else -1.0 for v in v_list]
    return x_list[vals.index(max(vals))]

def agg_prm_last_max(x_list: List[str], ans_list: List[str], v_list: List[List[float]]) -> str:
    vals = [v[-1] if v else -1.0 for v in v_list]
    return x_list[vals.index(max(vals))]

def agg_majority_vote(x_list: List[str], ans_list: List[str], v_list: List[List[float]]) -> str:
    valid = [a for a in ans_list if a]
    if not valid:
        # fallback to last_max like SSDP snippet
        return agg_prm_last_max(x_list, ans_list, v_list)

    counts = Counter(valid)
    most_common = max(counts, key=counts.get)

    for ans, text in zip(ans_list, x_list):
        if ans == most_common:
            return text
    return x_list[0]

def aggregate(voting_method: str, x_list: List[str], v_list: List[List[float]]) -> str:
    ans_list = [extract_answer(x) for x in x_list]
    if voting_method in ("all", "majority_vote"):
        return agg_majority_vote(x_list, ans_list, v_list)
    if voting_method in ("prm_last_max", "last_max"):
        return agg_prm_last_max(x_list, ans_list, v_list)
    if voting_method in ("prm_min_max", "min_max"):
        return agg_prm_min_max(x_list, ans_list, v_list)
    # default: majority vote
    return agg_majority_vote(x_list, ans_list, v_list)
