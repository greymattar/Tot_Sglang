import time
import requests
from dataclasses import dataclass
from typing import Dict, Any, List, Union, Optional


@dataclass
class GenResult:
    texts: List[str] # Changed to list
    time_llm_forward_s: float
    token_logprobs: Optional[List[Optional[List[float]]]] = None  # len = n, each is list[float] or None
    tokens: Optional[List[Optional[List[str]]]] = None            # len = n, each is list[str] or None

class VLLMBackend:
    def __init__(self, base_url: str, model: str, api_key: str = "token-abc123"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def generate_batch(self, prompts: List[str], sampling: Dict[str, Any]) -> List[GenResult]:
        """
        Sends a batch of prompts to vLLM in a single request (or parallel requests).
        """
        url = f"{self.base_url}/v1/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # vLLM supports list[str] for prompt
        payload = {
            "model": self.model,
            "prompt": prompts, # <--- SENDING LIST HERE
            "max_tokens": int(sampling.get("max_new_tokens", 100)),
            "temperature": float(sampling.get("temperature", 0.7)),
            "n": int(sampling.get("n", 1)),
        }

        if sampling.get("return_logprobs", False):
            payload["logprobs"] = int(sampling.get("logprobs_k", 1))

        t0 = time.time()
        #print("[DEBUG] POST", url, "n=", payload.get("n"), "logprobs=", payload.get("logprobs", None), "prompts=", len(prompts))
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=600)
            r.raise_for_status()
            data = r.json()

        except requests.HTTPError as e:
            print(f"Request Failed: {e}")
            try:
                print("Status:", r.status_code)
                print("Resp text:", r.text[:500])
            except Exception:
                pass
            return []
        except Exception as e:
            print(f"Request Failed (non-HTTP): {e}")
            return []
        t1 = time.time()


        
        choices = data["choices"]
        n_completions = payload["n"]
        
        results = []
        
        
        for i in range(0, len(choices), n_completions):
            chunk = choices[i : i + n_completions]
            texts = [c["text"] for c in chunk]
            chunk_token_logprobs = []
            chunk_tokens = []

            for c in chunk:
                lp = c.get("logprobs")
                if lp is None:
                    chunk_token_logprobs.append(None)
                    chunk_tokens.append(None)
                    continue
                tlp = lp.get("token_logprobs")
                toks = lp.get("tokens")
                chunk_token_logprobs.append(tlp if isinstance(tlp, list) else None)
                chunk_tokens.append(toks if isinstance(toks, list) else None)

            results.append(GenResult(texts=texts, time_llm_forward_s=(t1-t0), token_logprobs=chunk_token_logprobs,tokens=chunk_tokens,))
            
        return results
