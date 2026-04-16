# math_shepherd.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class MathShepherdPRM:
    """
    Math-Shepherd-style PRM.

    Two modes:
      - step mode (mini_step=False): score only at explicit step tags `step_tag`
      - mini-step mode (mini_step=True): dense token-level scores over the whole prefix,
        like SSDP's MistralPRM with config.mini_step = True.
    """
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        prm_dtype: str = "float16",
        step_tag: str = " ки",            # NOTE: leading space to match SSDP
        good_token: str = "+",
        bad_token: str = "-",
        mini_step: bool = False,
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.step_tag = step_tag
        self.mini_step = mini_step

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, prm_dtype),
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        # token ids used for scoring (same trick as SSDP)
        self.candidate_token_ids = self.tokenizer.encode(f"{good_token} {bad_token}")[1:]
        self.step_tag_id = self.tokenizer.encode(step_tag)[-1]

    @torch.no_grad()
    def step_scores(self, question: str, solution: str) -> torch.Tensor:
        """
        Returns scores in [0,1].

        - If mini_step == False:
            one score per *step tag* (classic Math-Shepherd).
        - If mini_step == True:
            one score per *token position* (SSDP mini-step).
        """
        if self.mini_step:
            # ---- SSDP-style mini_step = True ----
            # 1) build input (question + solution)
            inp = f"{question} {solution}{self.step_tag}"
            inputs = self.tokenizer(inp, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)

            # 2) logits restricted to ['+', '-'] candidates
            logits = self.model(input_ids).logits[:, :, self.candidate_token_ids]  # [1, T, 2]
            probs_good = logits.softmax(dim=-1)[:, :, 0]                           # [1, T]

            # 3) return dense per-token vector
            scores = probs_good[0]  # shape [T]
            return scores.detach().float().cpu()

        else:
            # ---- classic Math-Shepherd step mode ----
            inp = f"{question} {solution}"
            input_ids = torch.tensor([self.tokenizer.encode(inp)], device=self.device)
            logits = self.model(input_ids).logits[:, :, self.candidate_token_ids]  # [1, T, 2]
            probs_good = logits.softmax(dim=-1)[:, :, 0]                           # [1, T]
            mask = (input_ids == self.step_tag_id)
            return probs_good[mask].detach().float().cpu()

    def aggregate(self, scores: torch.Tensor, how: str = "min") -> float:
        """
        Aggregation is over:
          - step indices (step mode) OR
          - token positions (mini-step mode)
        exactly matching SSDP semantics in the latter case.
        """
        if scores.numel() == 0:
            return float("nan")
        if how == "min":
            return float(scores.min().item())
        if how == "last":
            return float(scores[-1].item())
        if how == "mean":
            return float(scores.mean().item())
        raise ValueError(f"Unknown aggregation: {how}")
