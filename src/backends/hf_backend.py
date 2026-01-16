
# src/backends/hf_backend.py
# Minimal HF "generate" backend that matches controller.py expectations:
#   res = backend.generate(prompt_text, sampling, seed)
#   res.text (completion only, NOT prompt+completion)
#   res.time_llm_forward_s (best-effort timing)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict
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
    HuggingFace generate() backend that returns ONLY the new completion text.

    """

    def __init__(
        self,
        model_id_or_path: str,
        device: str = "cuda",
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

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            use_fast=use_fast_tokenizer,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            # many Llama-like tokenizers have no pad token; use EOS as pad
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=self.dtype,
            device_map=None,  # single GPU
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()

        # Slight speed win; safe for inference
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
        
        _set_all_seeds(int(seed))

        max_new_tokens = int(sampling.get("max_new_tokens", 128))
        temperature = float(sampling.get("temperature", 0.0))
        top_p = float(sampling.get("top_p", 1.0))
        do_sample = bool(sampling.get("do_sample", False))
        num_beams = int(sampling.get("num_beams", 1))

        # If temperature is 0, force greedy (sampling off).
        if temperature <= 0.0:
            do_sample = False

        enc = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        prompt_len = int(input_ids.shape[-1])

        t0 = time.time()
        with torch.inference_mode():
            out_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                num_beams=num_beams,
                early_stopping=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=False,
            )
        t1 = time.time()

        # out_ids: (1, prompt_len + new_len)
        gen_ids = out_ids[0, prompt_len:]
        completion_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        meta = {
            "prompt_tokens": prompt_len,
            "completion_tokens": int(gen_ids.numel()),
        }
        return GenResult(text=completion_text, time_llm_forward_s=(t1 - t0), meta=meta)


