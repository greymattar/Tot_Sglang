import re
import uuid
from collections import Counter
from typing import List, Dict
from typing import List, Dict, Any, Optional
import regex  # if you already use it for BOX_RE; otherwise use re

import regex as reg

BAD_SUBSTRINGS = [
    "[[Category:", "Attribution-", "creativecommons", "license",
    "Note by", "Tagged:", "http://", "https://", "Solution:", "## Step"

]
#adding below , can keep or discard later
BAD_SUBSTRINGS += [
    "</", "<s", "<final", "<|", "|>",      # tag-ish / special tokens
    "to:", "cc:", "subject:",              # email spill
    "dear", "best regards", "thank you",   # letter-ish spill
    "please let me", "i hope",             # common trailing chatter
    "\\end{document}", "begin{document}",  # latex spill
]



FINAL_CUE_RE = re.compile(
    r"(<<FINAL>>|final\s+answer\s+is|therefore\s*,?\s*the\s+final\s+answer\s+is|answer\s*[:=]|\\boxed\{)",
    re.IGNORECASE | re.MULTILINE
)

CUT_RE = re.compile(
    r"(####|</|<\||<s|</s|To\s*:|CC\s*:|Subject\s*:|##\s*Step|\\end\{document\}|\\begin\{document\})",
    re.IGNORECASE
)


def trim_trailing_junk(a: str) -> str:
    if not a:
        return ""
    m = CUT_RE.search(a)
    if m:
        a = a[:m.start()]
    return a.strip().rstrip(" .;,:")

HASH_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)
BOX_RE = reg.compile(r"\\boxed\{((?:[^{}]|(?R))*)\}")


FINAL_TAG_RE = re.compile(r"(?is)<<FINAL>>\s*(.+?)\s*<</FINAL>>")
def _has_balanced_braces(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0

def is_plausible_final_answer(ans: str) -> bool:
    if not ans:
        return False
    a = ans.strip()
    if len(a) > 80:
        return False
    if "\n" in a or "\r" in a:
        return False
    low = a.lower()
    for b in BAD_SUBSTRINGS:
        if b.lower() in low:
            return False
    if a.endswith("\\"):
        return False
    if "\\frac{" in a and not _has_balanced_braces(a):
        return False
    if "\\boxed{" in a and "}" not in a:
        return False
    if a in {"<final answer>", "<final answer>."}:
        return False
    return True

def looks_terminalish(text: str) -> bool:
    if not text:
        return False
    t = text.replace("\\n", "\n")
    return bool(FINAL_CUE_RE.search(t))



def extract_answer_final_tag(text: str) -> str:
    # Only look near the end to avoid random earlier matches
    tail = "\n".join(text.replace("\\n", "\n").splitlines()[-12:])
    matches = FINAL_TAG_RE.findall(tail)
    if not matches:
        return ""
    ans = matches[-1].strip()
    ans = trim_trailing_junk(ans)
    return ans


def extract_answer_boxed(text: str) -> str:
    """
    Extracts content inside boxed{...}.
    Useful for MATH datasets if the model uses LaTeX boxing.
    """
    # Finds the last occurence of \boxed{...}
    # This regex is simple; a full nested brace parser is complex, 
    # but this covers 99% of model outputs.
    #matches = re.findall(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', text)
    tail = "\n".join(text.replace("\\n","\n").splitlines()[-8:])
    matches = BOX_RE.findall(tail)
    if matches:
        ans = matches[-1].strip()
        ans = trim_trailing_junk(ans)
        return ans
    return ""

def extract_answer_gsm8k(text: str) -> str:
    """
    Extracts content after '####'.
    UPDATED: Now captures EVERYTHING after ####, not just numbers.
    This supports LaTeX answers like '#### p - q'.
    """
    # re.DOTALL allows the dot (.) to match newlines if the answer is multiline
    #m = re.search(r"####\s*(.+)$", text, re.DOTALL)
    tail = "\n".join(text.replace("\\n","\n").splitlines()[-8:])
    m = HASH_RE.search(tail)
    if m:
        ans = m.group(1).strip()
        ans = trim_trailing_junk(ans)
        # NOTE: We removed .replace(",", "") because in MATH, 
        # commas are needed for coordinates like (3, 4).
        return ans
    return ""

def extract_answer(text: str) -> str:
    """
    Master extractor (STRICT).
    Returns "" unless we see a plausible final answer.
    This single function feeds:
      - candidate trigger
      - child_extract_answer logging
      - candidate list
      - final answer
    """
    if not text:
        return ""

    # If it doesn't look like it's trying to finish, don't treat it as an answer.
    # This kills most Llama-8B garbage candidates early.
    if not looks_terminalish(text):
        return ""
    a = extract_answer_final_tag(text)
    if a:
        a = a.strip().rstrip(" .;,")
        if is_plausible_final_answer(a):
            return a

    

    # Priority 2: '####' answer
    a = extract_answer_gsm8k(text)
    if a:
        a = a.strip().rstrip(" .;,")
        if is_plausible_final_answer(a):
            return a

    # Priority 3 boxed near end
    a = extract_answer_boxed(text)
    if a:
        a = a.strip().rstrip(" .;,")
        if is_plausible_final_answer(a):
            return a

    return ""


ALL_METHODS = ["min_max", "last_max", "majority_vote", "min_vote", "last_vote"]


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






def agg_prm_min_vote(x_list: List[str], ans_list: List[str], v_list: List[List[float]]) -> str:
    # choose candidate with MAX( min(step_scores) )
    # (same as min_max) -> but "vote" version usually means: do it on answers then pick best-scoring among that answer group
    # We'll implement true "min_vote": group by extracted answer, score each candidate by min(step_scores), then pick best group.
    scored = []
    for x, a, v in zip(x_list, ans_list, v_list):
        if not a:
            continue
        s = min(v) if v else -1.0
        scored.append((a, s, x))
    if not scored:
        return agg_prm_min_max(x_list, ans_list, v_list)

    # pick the answer group that has the best single candidate score
    best_a, best_s, best_x = max(scored, key=lambda t: t[1])
    return best_x

def agg_prm_last_vote(x_list: List[str], ans_list: List[str], v_list: List[List[float]]) -> str:
    scored = []
    for x, a, v in zip(x_list, ans_list, v_list):
        if not a:
            continue
        s = v[-1] if v else -1.0
        scored.append((a, s, x))
    if not scored:
        return agg_prm_last_max(x_list, ans_list, v_list)

    best_a, best_s, best_x = max(scored, key=lambda t: t[1])
    return best_x


def aggregate_one(method: str, x_list: List[str], v_list: List[List[float]]) -> str:
    ans_list = [extract_answer(x) for x in x_list]

    if method in ("majority_vote",):
        return agg_majority_vote(x_list, ans_list, v_list)
    if method in ("last_max", "prm_last_max"):
        return agg_prm_last_max(x_list, ans_list, v_list)
    if method in ("min_max", "prm_min_max"):
        return agg_prm_min_max(x_list, ans_list, v_list)
    if method in ("min_vote",):
        return agg_prm_min_vote(x_list, ans_list, v_list)
    if method in ("last_vote",):
        return agg_prm_last_vote(x_list, ans_list, v_list)

    # default fallback
    return agg_majority_vote(x_list, ans_list, v_list)


def aggregate_all(x_list: List[str], v_list: List[List[float]]) -> Dict[str, Dict[str, str]]:
    """
    Returns per-method:
      - chosen_text
      - chosen_answer (extracted)
    """
    out: Dict[str, Dict[str, str]] = {}
    for m in ALL_METHODS:
        chosen_text = aggregate_one(m, x_list, v_list)
        out[m] = {
            "chosen_text": chosen_text,
            "chosen_answer": extract_answer(chosen_text)  # keeps everything consistent with your extractor
        }
    return out
