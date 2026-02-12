#!/usr/bin/env python3
import requests
import time
import json
from datetime import datetime
import csv

def monitor_vllm_continuously(metrics_url="http://127.0.0.1:18000/metrics", 
                              output_file="vllm_monitor.csv",
                              interval=0.5):
    """Monitor vLLM metrics in real-time"""
    
    print(f"Starting vLLM monitoring. Output: {output_file}")
    print("Press Ctrl+C to stop.")
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = None
        
        while True:
            try:
                response = requests.get(metrics_url, timeout=2)
                metrics_text = response.text
                
                # Parse key metrics
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'unix_time': time.time()
                }
                
                for line in metrics_text.split('\n'):
                    line = line.strip()
                    if not line.startswith('vllm:'):
                        continue
                    
                    # Simple parsing
                    if '{' in line and '}' in line:
                        try:
                            name_part = line.split('{')[0]
                            value_part = line.split('}')[-1].strip()
                            value = float(value_part)
                            metrics[name_part] = value
                        except:
                            pass
                
                # Initialize CSV writer with headers
                if writer is None:
                    headers = list(metrics.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=headers)
                    writer.writeheader()
                
                writer.writerow(metrics)
                csvfile.flush()
                
                # Display key metrics
                print(f"\r[KV Cache: {metrics.get('vllm:kv_cache_usage_perc', 0):.1f}% | "
                      f"Preemptions: {metrics.get('vllm:num_preemptions_total', 0):.0f} | "
                      f"Running: {metrics.get('vllm:num_requests_running', 0):.0f}]", 
                      end='', flush=True)
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n\nMonitoring stopped.")
                break
            except Exception as e:
                print(f"\nError: {e}")
                time.sleep(1)

if __name__ == "__main__":
    monitor_vllm_continuously()
