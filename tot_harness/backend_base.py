from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class GenResult:
    text: str
    
    time_llm_forward_s: float | None = None

class LLMBackend:
    def generate(self, prompt: str, sampling: Dict[str, Any], seed: int) -> GenResult:
        raise NotImplementedError
