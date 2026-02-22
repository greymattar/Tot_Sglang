# tot_harness/grading/robust_grader.py
import re
import regex
import multiprocessing
from math import isclose
from typing import Union

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy

HASH_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)
# recursive boxed; good
BOX_RE  = regex.compile(r"\\boxed\{((?:[^{}]|(?R))*)\}")

FINAL_CUE_RE = re.compile(
    r"(final\s+answer\s+is|therefore\s*,?\s*the\s+final\s+answer\s+is|answer\s*[:=])",
    re.IGNORECASE
)



def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\\n", "\n")
    # 1) strongest: #### line
    m = HASH_RE.search(text)
    if m:
        ans = m.group(1).strip()
        ans = ans.split("\n", 1)[0].strip()
        return ans.rstrip(".")

    # 2) prefer boxed that appears AFTER a "final answer" cue
    cue = FINAL_CUE_RE.search(text)
    if cue:
        tail = text[cue.end():]
        boxed = [b.strip() for b in BOX_RE.findall(tail) if b.strip()]
        if boxed:
            return boxed[-1].strip().rstrip(" .;,")
        # if model used cue but no boxed, take a short tail line
        tail_line = tail.strip().splitlines()[0] if tail.strip() else ""
        return tail_line.strip().rstrip(" .;,")

    # 3) fallback: boxed near the end only (avoid intermediate boxed steps)
    # take last N characters and look for boxed there
    tail = "\n".join(text.splitlines()[-8:])  # tune: can use char also 
    boxed = [b.strip() for b in BOX_RE.findall(tail) if b.strip()]
    if boxed:
        return boxed[-1].strip().rstrip(" .;,")

    return ""

def clean_latex(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\left|\\right|\\displaystyle', '', s)
    s = re.sub(r'\\,|\\!|\\;|\\:|~', '', s)
    s = s.replace("π", r"\pi")
    s = re.sub(r"\s+", "", s)
    # strip enclosing punctuation
    s = s.strip().strip(".")
    return s

def parse_numeric_value(value):
    value = regex.sub(",", "", str(value))
    try:
        return float(value)
    except:
        if value.endswith("%"):
            value = value.rstrip("%\\")
            try:
                return float(value) / 100
            except:
                pass
    return None

def is_numeric(value):
    return parse_numeric_value(value) is not None

def numeric_equal(a: float, b: float, tolerance=1e-4) -> bool:
    return isclose(a, b, rel_tol=tolerance)

def symbolic_equal(a_expr, b_expr) -> bool:
    def try_parsing(expr):
        for parser in [parse_latex, parse_expr, latex2sympy]:
            try:
                return parser(expr.replace("\\\\", "\\"))
            except:
                continue
        return expr

    a, b = try_parsing(a_expr), try_parsing(b_expr)

    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass
    try:
        if hasattr(a, "equals") and a.equals(b):
            return True
        if simplify(a - b) == 0:
            return True
    except:
        pass
    try:
        return numeric_equal(float(N(a)), float(N(b)))
    except:
        pass
    return False

def symbolic_equal_process(a, b, output_queue):
    output_queue.put(symbolic_equal(a, b))

def call_with_timeout(func, *args, timeout=1):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=func, args=args + (q,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    return q.get()

def math_equal(
    pred: Union[bool, float, str],
    ref: Union[float, str],
    allow_percentage: bool = True,
    allow_close_match: bool = True,
    check_timeout: bool = True,
) -> bool:
    if pred is None or ref is None:
        return False

    pred = clean_latex(str(pred))
    ref  = clean_latex(str(ref))

    if pred.lower() == ref.lower():
        return True

    # numeric path
    try:
        if is_numeric(pred) and is_numeric(ref):
            pred_num, ref_num = parse_numeric_value(pred), parse_numeric_value(ref)
            comps = [ref_num / 100, ref_num, ref_num * 100] if allow_percentage else [ref_num]
            return any((numeric_equal(pred_num, c) if allow_close_match else pred_num == c) for c in comps)
    except:
        pass

    # symbolic path (timeout protected)
    if check_timeout:
        return call_with_timeout(symbolic_equal_process, pred, ref, timeout=1)
    return symbolic_equal(pred, ref)

