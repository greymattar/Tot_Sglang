import requests
import time
import json
from typing import List, Dict, Any

class VLLMBackendFixed:
    """Fixed backend that properly handles multiple completions"""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/v1/completions"  # Use completions endpoint
        
    def generate_batch(self, prompts: List[str], sampling: Dict[str, Any]) -> List:
        """Generate multiple completions per prompt using vLLM's native API"""
        
        results = []
        n_completions = sampling.get('n', 4)
        
        print(f"[VLLM] Generating {n_completions} completions per prompt for {len(prompts)} prompts")
        
        for prompt in prompts:
            try:
                # Prepare request for vLLM's completions endpoint
                # This is more reliable than chat completions for multiple outputs
                request_data = {
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": sampling.get('max_tokens', 32),
                    "temperature": sampling.get('temperature', 1.0),
                    "top_p": sampling.get('top_p', 0.9),
                    "n": n_completions,  # Number of completions
                    "best_of": n_completions,  # vLLM specific parameter
                    "use_beam_search": False,
                    "stream": False
                }
                
                start_time = time.perf_counter()
                
                response = requests.post(
                    self.api_url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(request_data),
                    timeout=60
                )
                
                if response.status_code != 200:
                    print(f"[ERROR] API error: {response.status_code} - {response.text}")
                    raise Exception(f"API error: {response.status_code}")
                
                response_data = response.json()
                e2e_latency = time.perf_counter() - start_time
                
                # Extract all completions
                completions = []
                if 'choices' in response_data:
                    for choice in response_data['choices']:
                        completions.append(choice['text'])
                
                print(f"[VLLM] Got {len(completions)} completions")
                
                # Create result object
                result_obj = type('Result', (), {})()
                result_obj.texts = completions
                result_obj.time_llm_forward_s = e2e_latency
                
                results.append(result_obj)
                
            except Exception as e:
                print(f"[ERROR] Failed to generate: {e}")
                # Return empty result
                result_obj = type('Result', (), {})()
                result_obj.texts = []
                result_obj.time_llm_forward_s = 0
                results.append(result_obj)
        
        return results
