import os
from src.prm.math_shepherd import MathShepherdPRM
from src.prm.format_steps import to_step_tagged

question = "Janet’s ducks lay 16 eggs per day. She eats 3 and bakes 4. She sells the rest for $2 each. How much per day?"

good_raw = "16 - 3 = 13\n13 - 4 = 9\n9 * 2 = 18\n#### 18"
bad_raw  = "16 - 3 = 13\n13 - 4 = 9\n9 * 2 = 17\n#### 17"

good = to_step_tagged(good_raw, "ки")
bad  = to_step_tagged(bad_raw, "ки")

prm = MathShepherdPRM("peiyi9979/math-shepherd-mistral-7b-prm", device="cuda", prm_dtype="float16")
for name, sol in [("good", good), ("bad", bad)]:
    s = prm.step_scores(question, sol)
    print(name, "nsteps=", len(s), "scores=", s.tolist(), "agg(min)=", prm.aggregate(s, "min"))
