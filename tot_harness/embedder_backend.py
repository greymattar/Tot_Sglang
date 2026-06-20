import time
import requests
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EmbedResult:
    vectors: List[List[float]]
    time_s: float


class VLLMEmbedder:
    """
    OpenAI-compatible embeddings client. Works with vLLM's /v1/embeddings.
    """
    def __init__(self, base_url: str, model: str, api_key: str = "token-abc123", timeout_s: int = 600):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def embed(self, texts: List[str]) -> EmbedResult:
        url = f"{self.base_url}/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
        }

        t0 = time.time()
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        t1 = time.time()

        # OpenAI format: data["data"] is list of {"embedding": [...], "index": i}
        items = data.get("data", [])
        # Ensure stable ordering by index
        items_sorted = sorted(items, key=lambda x: x.get("index", 0))
        vecs = [it["embedding"] for it in items_sorted]

        return EmbedResult(vectors=vecs, time_s=(t1 - t0))
