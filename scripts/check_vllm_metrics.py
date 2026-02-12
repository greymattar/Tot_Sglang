import requests
import re

def get_vllm_metrics(url="http://127.0.0.1:18000"):
    try:
        response = requests.get(f"{url}/metrics", timeout=5)
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return ""

def categorize_metrics(metrics_text):
    categories = {
        "cache": [],
        "timing": [],
        "memory": [],
        "throughput": [],
        "queue": [],
        "other": []
    }
    
    for line in metrics_text.split('\n'):
        if line.startswith('vllm:'):
            # Parse metric
            if 'cache' in line.lower():
                categories["cache"].append(line)
            elif any(time_word in line.lower() for time_word in ['time', 'latency', 'duration']):
                categories["timing"].append(line)
            elif any(mem_word in line.lower() for mem_word in ['memory', 'gpu', 'cuda']):
                categories["memory"].append(line)
            elif any(queue_word in line.lower() for queue_word in ['queue', 'waiting', 'pending']):
                categories["queue"].append(line)
            elif any(tput_word in line.lower() for tput_word in ['throughput', 'tokens_per_second', 'requests']):
                categories["throughput"].append(line)
            else:
                categories["other"].append(line)
    
    return categories

def main():
    metrics = get_vllm_metrics()
    if not metrics:
        print("Could not fetch metrics")
        return
    
    categories = categorize_metrics(metrics)
    
    print("=" * 60)
    print("VLLM METRICS ANALYSIS")
    print("=" * 60)
    
    for category, lines in categories.items():
        print(f"\n{category.upper()} ({len(lines)} metrics):")
        print("-" * 40)
        for line in lines[:10]:  # Show first 10 of each category
            print(f"  {line}")
        if len(lines) > 10:
            print(f"  ... and {len(lines) - 10} more")
    
    # Look specifically for cache metrics
    print("\n" + "=" * 60)
    print("CACHE-SPECIFIC METRICS:")
    print("=" * 60)
    cache_lines = [l for l in metrics.split('\n') if 'cache' in l.lower() and 'vllm:' in l]
    for line in cache_lines:
        print(f"  {line}")
    
    if not cache_lines:
        print("  No cache-specific metrics found!")
        print("  Note: You might need to enable cache metrics in vLLM config")

if __name__ == "__main__":
    main()
