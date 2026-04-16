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
    r"(<<FINAL>>|<FINAL>|</FINAL>|"
    r"final\s+answer\s*(?:is)?\s*[:=]?\s*|"
    r"therefore\s*,?\s*the\s+final\s+answer\s*(?:is)?\s*[:=]?\s*|"
    r"answer\s*[:=]\s*|"
    r"\\boxed\{)",
    re.IGNORECASE | re.MULTILINE
)

CUT_RE = re.compile(
    r"(####|</|<\||<s|</s|To\s*:|CC\s*:|Subject\s*:|##\s*Step|\\end\{document\}|\\begin\{document\})",
    re.IGNORECASE
)

ASSIGN_RE = re.compile(r"^[a-zA-Z]\w*=(.+)$")



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

def _canon_key(ans: str) -> str:
    """
    Canonical key for voting/dedup.
    - If answer looks like assignment (x=..., y=...), vote on RHS only.
    - Otherwise vote on normalized math string.
    """
    if not ans:
        return ""

    a = ans.strip()


    rhs = rhs_if_assignment(a)  # returns "" if not assignment
    if rhs:
        return normalize_math_str(rhs)
    num = parse_numeric_value(a)  
    if num is not None:
        return normalize_math_str(num)

    return normalize_math_str(a)


def trim_trailing_junk(a: str) -> str:
    if not a:
        return ""
    m = CUT_RE.search(a)
    if m:
        a = a[:m.start()]
    return a.strip().rstrip(" .;,:")

HASH_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)
BOX_RE = reg.compile(r"\\boxed\{((?:[^{}]|(?R))*)\}")


FINAL_TAG_RE = re.compile(
    r"(?is)(?:<<FINAL>>|<FINAL>)\s*(.+?)\s*(?:<</FINAL>>|</FINAL>)"
)
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




def _extract_last_boxed_balanced(text: str) -> str:
    t = text.replace("\\n", "\n")
    tail = "\n".join(t.splitlines()[-12:])   # a bit larger tail is safer

    key = r"\boxed{"
    start = tail.rfind(key)
    if start == -1:
        return ""

    i = start + len(key)
    depth = 1
    buf = []

    while i < len(tail):
        ch = tail[i]
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                ans = "".join(buf).strip()
                return trim_trailing_junk(ans)
            buf.append(ch)
        else:
            buf.append(ch)
        i += 1

    return ""

def extract_answer_boxed(text: str) -> str:
    return _extract_last_boxed_balanced(text)

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
    # Build (idx, raw_ans, canon_ans) for valid answers
    triples = []
    for i, a in enumerate(ans_list):
        if not a:
            continue
        canon = _canon_key(a)
        if not canon:
            continue
        triples.append((i, a, canon))

    if not triples:
        # fallback to last_max like SSDP snippet
        return agg_prm_last_max(x_list, ans_list, v_list)

    counts = Counter(canon for _, _, canon in triples)
    winner_canon = counts.most_common(1)[0][0]

    # Return the candidate TEXT corresponding to the winning canon key
    for i, raw, canon in triples:
        if canon == winner_canon:
            return x_list[i]

    # should never hit, but safe fallback
    return x_list[triples[0][0]]


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
