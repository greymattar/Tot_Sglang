#!/bin/bash

# CONFIGURATION
VLLM_URL="http://127.0.0.1:18000"  
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="runlogs/vllm_monitor_${TIMESTAMP}.csv"

# 1. Start the Monitor in the Background
echo "Starting vLLM Monitor..."
python scripts/monitor_vllm.py --url $VLLM_URL --output $LOG_FILE &
MONITOR_PID=$! # Save the Process ID (PID) of the monitor

# 2. Wait a second to ensure it started
sleep 1

# 3. Run the Main Experiment
echo "Starting ToT Experiment..."
# Pass your arguments to the python script here
python scripts/run_tierA_batched.py --vllm_url $VLLM_URL --batch_problems 20 --batch_width 4

# 4. Cleanup: Kill the Monitor when experiment finishes
echo "Experiment finished. Stopping Monitor (PID: $MONITOR_PID)..."
kill $MONITOR_PID

echo "Done. Monitor log saved to: $LOG_FILE"
