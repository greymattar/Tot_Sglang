# src/prm/qwen_prm.py

import torch
from transformers import AutoTokenizer, AutoModel

class QwenPRM:
    """
    Qwen2.5-Math-PRM-style process reward model.

    Interface matches MathShepherdPRM:
      - __init__(model_id, device, prm_dtype, step_tag, good_token, bad_token, mini_step)
      - step_scores(question, solution) -> Tensor in [0,1], one score per step
      - aggregate(scores, how)
    good_token / bad_token / mini_step are accepted for signature compatibility.
    """
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        prm_dtype: str = "bfloat16",
        step_tag: str = "\n\n",
        good_token: str = "+",
        bad_token: str = "-",
        mini_step: bool = False,
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.step_tag = step_tag
        self.mini_step = mini_step

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, prm_dtype),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        # ADJUST FOR YOUR PRM: the token that marks the end of a step.
        # Qwen2.5-Math-PRM-7B uses the special token "<extra_0>".
        self.step_sep_id = self.tokenizer.encode("<extra_0>")[0]

    @torch.no_grad()
    def step_scores(self, question: str, solution: str) -> torch.Tensor:
        # Split completion into steps and join with the separator token the PRM expects.
        steps = [s.strip() for s in solution.split(self.step_tag) if s.strip()]
        sep = "<extra_0>"  # ADJUST: must match self.step_sep_id
        convo = [
            {"role": "system", "content": "Please reason step by step."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": sep.join(steps) + sep},
        ]
        text = self.tokenizer.apply_chat_template(convo, tokenize=False)
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)

        outputs = self.model(input_ids=input_ids, use_cache=False)
        # ADJUST FOR YOUR PRM: how the per-step reward is read.
        # Qwen2.5-Math-PRM returns a 2-class logit; take P(positive) at each sep token.
        logits = outputs[0]                      # [1, T, 2] for this model family
        probs = logits.softmax(dim=-1)[0, :, 1]  # P(good) per token
        mask = (input_ids[0] == self.step_sep_id)
        scores = probs[mask]
        return scores.detach().float().cpu()

    def aggregate(self, scores: torch.Tensor, how: str = "min") -> float:
        if scores.numel() == 0:
            return float("nan")
        if how == "min":
            return float(scores.min().item())
        if how == "last":
            return float(scores[-1].item())
        if how == "mean":
            return float(scores.mean().item())
        raise ValueError(f"Unknown aggregation: {how}")
