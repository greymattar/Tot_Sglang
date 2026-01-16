from dataclasses import dataclass
import time
import requests


@dataclass
class GenResult:
    text: str
    time_llm_forward_s: float | None = None
    meta: dict | None = None


class SGLangBackend:
    def __init__(self, base_url: str, timeout_s: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _map_params(self, sampling: dict, seed: int) -> dict:
        return {
            "max_new_tokens": int(sampling["max_new_tokens"]),
            "temperature": float(sampling.get("temperature", 0.0)),
            "top_p": float(sampling.get("top_p", 1.0)),
            "sampling_seed": int(seed),
        }

    def generate_batch(self, prompts: list[str], sampling_params_list: list[dict]) -> list[GenResult]:
        assert len(prompts) == len(sampling_params_list)

        payload = {"text": prompts, "sampling_params": sampling_params_list, "stream": False}

        t0 = time.time()
        r = requests.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        t1 = time.time()
        out = r.json()



        dt = t1 - t0
        n = len(prompts)
        per_item = dt / max(1, n)

        if isinstance(out, dict):
            texts = out.get("text", [])
            if isinstance(texts, str):
                texts = [texts]
            metas = out.get("meta_info", [{}] * len(texts))
            if isinstance(metas, dict):
                metas = [metas]
            if len(metas) < len(texts):
                metas = list(metas) + ([{}] * (len(texts) - len(metas)))
            return [GenResult(text=texts[i], time_llm_forward_s=per_item, meta=metas[i]) for i in range(len(texts))]

        if isinstance(out, list):
            results = []
            for i, item in enumerate(out):
                if isinstance(item, dict):
                    txt = item.get("text", "")
                    meta = item.get("meta_info", None)
                else:
                    txt = str(item)
                    meta = None
                results.append(GenResult(text=txt, time_llm_forward_s=per_item, meta=meta))
            return results
        return [GenResult(text="", time_llm_forward_s=per_item, meta={"raw": out}) for _ in range(n)]



    def generate_n(self, prompt_text: str, sampling: dict, seed: int, n: int) -> list[GenResult]:
        n = int(n)
        params_list = [self._map_params(sampling, seed + j) for j in range(n)]
        prompts = [prompt_text] * n
        return self.generate_batch(prompts, params_list)

    def generate(self, prompt_text: str, sampling: dict, seed: int) -> GenResult:
        return self.generate_n(prompt_text, sampling, seed, n=1)[0]

