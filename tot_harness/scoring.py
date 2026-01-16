import re

def extract_gsm8k_final(answer_text: str) -> str:
    # GSM8K answers usually end with "#### <number>"
    m = re.search(r"####\s*([-\d\.,]+)", answer_text)
    if m:
        return m.group(1).replace(",", "").strip()
    # fallback: last number-like token
    nums = re.findall(r"[-]?\d+(?:\.\d+)?", answer_text.replace(",", ""))
    return nums[-1].strip() if nums else ""

def score_gsm8k_exact(pred_text: str, gold_answer: str) -> float:
    pred = extract_gsm8k_final(pred_text)
    gold = extract_gsm8k_final(gold_answer)
    return 1.0 if pred != "" and pred == gold else 0.0
