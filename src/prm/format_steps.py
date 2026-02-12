import re

def to_step_tagged(text: str, step_tag: str = "ки") -> str:
    """
    Very simple heuristic:
    - split on newlines or 'Step' markers
    - attach `step_tag` after each non-empty step line
    This mirrors the Math-Shepherd requirement that scores are read at each tag.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    out = []
    for ln in lines:
        # avoid double tagging
        if step_tag in ln:
            out.append(ln)
        else:
            out.append(f"{ln} {step_tag}")
    return "\n".join(out)

def extract_final_answer_line(text: str) -> str:
    if not text: return ""
    
    # 1. Try finding the standard "####" marker
    # We capture everything (.+) until the end of the string
    m = re.search(r"####\s*(.+)$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    
    # 2. Fallback: If no marker, maybe return the last non-empty line?
    # (Optional, but risky for MATH. Safer to return "" if format isn't followed)
    return ""


