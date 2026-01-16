from dataclasses import dataclass
from transformers import AutoTokenizer

@dataclass
class TokenCounter:
    tokenizer_name: str
    def __post_init__(self):
        self.tok = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)

    def count_prompt_tokens(self, prompt: str) -> int:
        ids = self.tok(prompt, add_special_tokens=False).input_ids
        return len(ids)

    def count_generated_tokens_delta(self, prompt: str, completion: str) -> int:
        # Robust definition:
        # generated := tokens(prompt+completion) - tokens(prompt)
        # (works even if completion has leading spaces/newlines)
        full = self.tok(prompt + completion, add_special_tokens=False).input_ids
        base = self.tok(prompt, add_special_tokens=False).input_ids
        return max(0, len(full) - len(base))
