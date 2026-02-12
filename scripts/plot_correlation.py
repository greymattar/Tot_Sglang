import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

# FILES PRODUCED BY YOUR EXPERIMENT
CLIENT_LOG = "/nfsasync/atkia/Tot_Sglang/runlogs/batched_trace_20260120_163942.jsonl" 
SERVER_LOG = "/nfsasync/atkia/Tot_Sglang/runlogs/vllm_monitor_20260120_163915.csv"    

def plot_correlation():
    # 1. Load Client Data (Staleness)
    client_data = []
    with open(CLIENT_LOG, 'r') as f:
        for line in f:
            try:
                d = json.loads(line)
                if d["event"] == "expand" and "timestamp" in d:
                    client_data.append(d)
            except: pass
    
    df_client = pd.DataFrame(client_data)
    if df_client.empty:
        print("No client data found!")
        return
        
    # Normalize Time (Start at 0)
    start_time = df_client['timestamp'].min()
    df_client['time_rel'] = df_client['timestamp'] - start_time

    # 2. Load Server Data (Preemptions)
    df_server = pd.read_csv(SERVER_LOG)
    df_server['time_rel'] = df_server['timestamp_unix'] - start_time
    
    # Filter to relevant timeframe
    df_server = df_server[df_server['time_rel'] >= 0]
    df_server = df_server[df_server['time_rel'] <= df_client['time_rel'].max() + 5]

    # Calculate Preemption DELTA (Rate) instead of just Total
    # This shows "Preemptions happening NOW" rather than "History"
    df_server['preemption_rate'] = df_server['vllm:num_preemptions_total'].diff().fillna(0)

    # 3. Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.set_theme(style="white")

    # --- LEFT AXIS: Staleness (The Cause) ---
    # We use a scatter plot. Higher dots = Older nodes being accessed.
    sns.scatterplot(data=df_client, x='time_rel', y='staleness', ax=ax1, 
                    color='blue', alpha=0.5, s=20, label='Node Staleness')
    ax1.set_xlabel("Time (seconds)", fontsize=12)
    ax1.set_ylabel("Node Staleness (Iterations)", color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Add a smooth trendline for staleness
    sns.regplot(data=df_client, x='time_rel', y='staleness', scatter=False, 
                ax=ax1, color='blue', line_kws={'alpha':0.3})

    # --- RIGHT AXIS: Preemptions (The Effect) ---
    ax2 = ax1.twinx()
    
    # Plot Cumulative Preemptions (The steepness of slope indicates crash intensity)
    sns.lineplot(data=df_server, x='time_rel', y='vllm:num_preemptions_total', ax=ax2, 
                 color='red', linewidth=3, label='Total Preemptions')
    
    ax2.set_ylabel("Cumulative Preemptions", color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')

    # 4. Highlight Correlation
    plt.title("Correlation: Accessing Stale Nodes Triggers Preemptions", fontsize=14, fontweight='bold')
    
    # Add arrows or annotations if needed (optional)
    # peak_staleness = df_client['staleness'].max()
    # peak_time = df_client.loc[df_client['staleness'].idxmax(), 'time_rel']
    # ax1.annotate('Peak Staleness', xy=(peak_time, peak_staleness), xytext=(peak_time+10, peak_staleness),
    #              arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig("proof_of_failure.pdf")
    print("Graph saved to proof_of_failure.pdf")

if __name__ == "__main__":
    plot_correlation()
