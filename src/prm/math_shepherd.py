import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class MathShepherdPRM:
    """
    Implements the official math-shepherd-mistral-7b-prm scoring:
    - append step_tag 'ки' at the end of each step
    - score(step) = P('+' | context_at_step_tag) where candidates are ['+','-']
    """
    def __init__(self, model_id: str, device: str = "cuda",
                 prm_dtype: str = "float16",
                 step_tag: str = "ки", good_token: str = "+", bad_token: str = "-"):
        self.model_id = model_id
        self.device = torch.device(device)
        self.step_tag = step_tag

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, prm_dtype),
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        # token ids used for scoring
        self.candidate_token_ids = self.tokenizer.encode(f"{good_token} {bad_token}")[1:]
        self.step_tag_id = self.tokenizer.encode(step_tag)[-1]

    @torch.no_grad()
    def step_scores(self, question: str, solution_with_steps_and_tags: str) -> torch.Tensor:
        """
        Returns a 1D tensor of per-step scores in [0,1], one score per step-tag occurrence.
        """
        inp = f"{question} {solution_with_steps_and_tags}"
        input_ids = torch.tensor([self.tokenizer.encode(inp)], device=self.device)
        logits = self.model(input_ids).logits[:, :, self.candidate_token_ids]  # [1, T, 2]
        probs_good = logits.softmax(dim=-1)[:, :, 0]                            # [1, T]
        mask = (input_ids == self.step_tag_id)
        return probs_good[mask].detach().float().cpu()

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
