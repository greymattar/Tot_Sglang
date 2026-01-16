###not using this
import time, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .backend_base import LLMBackend, GenResult

class HFBackend(LLMBackend):
    def __init__(self, model_name: str, dtype: str = "float16", device: str = "cuda:0"):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=None,
        ).to(device)
        self.device = device
        self.model.eval()

    def generate(self, prompt: str, sampling: dict, seed: int) -> GenResult:
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        inputs = self.tok(prompt, return_tensors="pt").to(self.device)

        gen_kwargs = dict(
            do_sample=True,
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            max_new_tokens=int(sampling["max_new_tokens"]),
            pad_token_id=self.tok.eos_token_id,
        )

        t0 = time.time()
        out = self.model.generate(**inputs, **gen_kwargs)
        t1 = time.time()

        # decode only newly generated portion for cleanliness
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)

        return GenResult(text=text, time_llm_forward_s=(t1 - t0))
