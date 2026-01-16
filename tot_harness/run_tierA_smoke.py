#moved to /scripts
# src/backends/hf_backend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, List
import os
import time
import random

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class GenResult:
    text: str
    time_llm_forward_s: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HFBackend:
    """
    HF backend that can return either:
      - one completion: generate(prompt, sampling, seed)
      - N completions in ONE HF generate call: generate_n(prompt, sampling, seed, n)

     closest match to "dpts: num_beams=4 but do_sample=True" configs,
   
    """

    def __init__(
        self,
        model_id_or_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        hf_home: Optional[str] = None,
        trust_remote_code: bool = False,
        use_fast_tokenizer: bool = False,
    ):
        if hf_home:
            os.environ.setdefault("HF_HOME", hf_home)
            os.environ.setdefault("HF_HUB_CACHE", hf_home)
            os.environ.setdefault("TRANSFORMERS_CACHE", hf_home)

        self.device = device
        self.dtype = self._parse_dtype(dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            use_fast=use_fast_tokenizer,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=self.dtype,
            device_map=None,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()
        torch.set_grad_enabled(False)

    @staticmethod
    def _parse_dtype(dtype: str):
        d = str(dtype).lower()
        if d in ("bf16", "bfloat16"):
            return torch.bfloat16
        if d in ("fp16", "float16", "half"):
            return torch.float16
        if d in ("fp32", "float32"):
            return torch.float32
        raise ValueError(f"Unsupported dtype={dtype}. Use bfloat16/float16/float32.")

    def generate(self, prompt_text: str, sampling: dict, seed: int) -> GenResult:
        # single output (keeps your controller working unchanged)
        return self.generate_n(prompt_text, sampling, seed, n=1)[0]

    def generate_n(self, prompt_text: str, sampling: dict, seed: int, n: int) -> List[GenResult]:
        """
        Return n completions from ONE HF generate call.

        Mapping of your config intent:
          - If do_sample=True -> use sampling, set num_return_sequences=n
          - If do_sample=False and num_beams>1 -> do beam search and return n beams
          - If do_sample=False and num_beams==1 -> greedy, n must be 1 (or we will replicate prompt w/ different seeds, but we avoid that)

        Note: True determinism with do_sample=True is hard across frameworks; we do best-effort seeding.
        """
        n = int(n)
        if n < 1:
            return []

        _set_all_seeds(int(seed))

        max_new_tokens = int(sampling.get("max_new_tokens", 128))
        temperature = float(sampling.get("temperature", 0.0))
        top_p = float(sampling.get("top_p", 1.0))
        do_sample = bool(sampling.get("do_sample", False))
        num_beams = int(sampling.get("num_beams", 1))

        # If temperature is 0, force greedy.
        if temperature <= 0.0:
            do_sample = False

        enc = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        prompt_len = int(input_ids.shape[-1])

        # Decide how many sequences HF will return in one call
        if do_sample:
            # "N samples" from the same prompt in one call
            num_return_sequences = n
            # beam search with sampling is a weird hybrid; for DPTS-like configs,
            # the clean interpretation is: num_beams=1, num_return_sequences=n
            # but we won't override num_beams unless you want strict behavior:
            # set num_beams = 1
            if num_beams < 1:
                num_beams = 1
        else:
            # Non-sampling case: greedy or beam search
            if num_beams == 1 and n > 1:
                # Greedy cannot return multiple distinct sequences.
                # Best defensible behavior: just return 1.
                n = 1
            num_return_sequences = min(n, num_beams)

        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                early_stopping=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=False,
            )
        t1 = time.time()

        # HF returns shape (num_return_sequences, prompt_len + new_len)
        # or (1, ...) depending on settings.
        if out.dim() == 1:
            out = out.unsqueeze(0)

        results: List[GenResult] = []
        per_seq_time = (t1 - t0) / max(1, out.shape[0])

        for row in range(out.shape[0]):
            gen_ids = out[row, prompt_len:]
            completion_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            meta = {
                "prompt_tokens": prompt_len,
                "completion_tokens": int(gen_ids.numel()),
                "num_beams": num_beams,
                "do_sample": do_sample,
                "num_return_sequences": int(out.shape[0]),
            }
            results.append(GenResult(text=completion_text, time_llm_forward_s=per_seq_time, meta=meta))

        return results
