import json, os, hashlib
from datasets import load_dataset

EXP_ROOT = os.environ.get("EXP_ROOT", ".")
OUT_PATH = os.path.join(EXP_ROOT, "data/gsm8k_test_first50.jsonl")

def sha(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()[:16]

ds = load_dataset("gsm8k", "main", split="test")

N = 50
rows = []
for i in range(N):
    ex = ds[i]
    q = ex["question"]
    a = ex["answer"]
    rows.append({
        "idx": i,
        "id": f"gsm8k_test_{i:05d}_{sha(q)}",
        "question": q,
        "answer": a,
    })

with open(OUT_PATH, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print("Wrote:", OUT_PATH)
print("First id:", rows[0]["id"])
