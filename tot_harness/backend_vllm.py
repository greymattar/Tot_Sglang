import time
import requests
from dataclasses import dataclass
from typing import Dict, Any, List, Union

@dataclass
class GenResult:
    texts: List[str] # Changed to list
    time_llm_forward_s: float

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

        t0 = time.time()
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=600)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"Request Failed: {e}")
            return []
        t1 = time.time()


        
        choices = data["choices"]
        n_completions = payload["n"]
        
        results = []
        
        
        for i in range(0, len(choices), n_completions):
            chunk = choices[i : i + n_completions]
            texts = [c["text"] for c in chunk]
            results.append(GenResult(texts=texts, time_llm_forward_s=(t1-t0)))
            
        return results
