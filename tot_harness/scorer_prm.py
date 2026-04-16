# tot_harness/scorer_prm.py

from src.prm.math_shepherd import MathShepherdPRM
from src.prm.format_steps import to_step_tagged

class PRMScorer:
    def __init__(
        self,
        prm_model_id: str,
        device: str,
        prm_dtype: str,
        step_tag: str,
        good_token: str,
        bad_token: str,
        aggregation: str,
        mini_step: bool = False,
    ):
        self.mini_step = mini_step
        self.prm = MathShepherdPRM(
            prm_model_id,
            device=device,
            prm_dtype=prm_dtype,
            step_tag=step_tag,
            good_token=good_token,
            bad_token=bad_token,
            mini_step=mini_step,
        )
        self.aggregation = aggregation
        self.step_tag = step_tag

    def score(self, question: str, completion: str) -> float:
        if self.mini_step:
            # SSDP mini-step style: no explicit tagging, dense per-token scores
            scores = self.prm.step_scores(question, completion)
        else:
            # Classic Math-Shepherd step PRM
            tagged = to_step_tagged(completion, self.step_tag)
            scores = self.prm.step_scores(question, tagged)

        return self.prm.aggregate(scores, self.aggregation)
