#!/bin/bash
# Auto Checkpoint Eval Monitor
# Watches for new checkpoints and runs eval automatically
#
# Usage: ./auto_checkpoint_eval.sh [track] [eval_interval]
# Example: ./auto_checkpoint_eval.sh 2 4000

set -e

TRACK=${1:-2}
EVAL_INTERVAL=${2:-4000}
EXP_DIR="/root/zind/Experiments/exp013_full_pipeline"
VLLM_PORT=8002

# Track-specific config
case $TRACK in
    1)
        BASE_MODEL="/root/Qwen3-32B"
        ADAPTER_PREFIX="track1_32b"
        GPU_MEM="0.3"
        ;;
    2)
        BASE_MODEL="/root/Qwen2.5-7B-Instruct"
        ADAPTER_PREFIX="track2_7b"
        GPU_MEM="0.4"
        ;;
    3)
        BASE_MODEL="/root/Qwen2.5-1.5B-Instruct"
        ADAPTER_PREFIX="track3_1.5b"
        GPU_MEM="0.4"
        ;;
    *)
        echo "Unknown track: $TRACK"
        exit 1
        ;;
esac

PROGRESS_DIR="$EXP_DIR/adapters/${ADAPTER_PREFIX}_progress"
EVAL_LOG="$EXP_DIR/auto_eval_track${TRACK}.log"
LAST_EVAL_FILE="$EXP_DIR/.last_eval_step_track${TRACK}"

echo "========================================"
echo "Auto Checkpoint Eval Monitor"
echo "========================================"
echo "Track: $TRACK ($ADAPTER_PREFIX)"
echo "Base Model: $BASE_MODEL"
echo "Progress Dir: $PROGRESS_DIR"
echo "Eval Interval: every $EVAL_INTERVAL steps"
echo "Log: $EVAL_LOG"
echo "========================================"

# Initialize last eval step
if [ -f "$LAST_EVAL_FILE" ]; then
    LAST_EVAL_STEP=$(cat "$LAST_EVAL_FILE")
else
    LAST_EVAL_STEP=0
fi
echo "Last evaluated step: $LAST_EVAL_STEP"

# Function to get latest checkpoint step
get_latest_checkpoint() {
    if [ -d "$PROGRESS_DIR" ]; then
        ls -1d "$PROGRESS_DIR"/checkpoint-* 2>/dev/null | \
            sed 's/.*checkpoint-//' | sort -n | tail -1
    else
        echo "0"
    fi
}

# Function to run eval on a checkpoint
run_eval() {
    local STEP=$1
    local CKPT_PATH="$PROGRESS_DIR/checkpoint-$STEP"
    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
    echo "[$(date)] Starting eval for checkpoint-$STEP" | tee -a "$EVAL_LOG"
    
    # Cleanup any existing vLLM
    docker rm -f vllm_auto_eval 2>/dev/null || true
    
    # Start vLLM
    echo "[$(date)] Starting vLLM..." | tee -a "$EVAL_LOG"
    docker run -d --gpus all --name vllm_auto_eval \
        -v "$BASE_MODEL":/model \
        -v "$CKPT_PATH":/adapter \
        -p $VLLM_PORT:8000 vllm/vllm-openai:latest \
        --model /model --enable-lora --lora-modules ckpt_eval=/adapter \
        --max-model-len 8000 --max-lora-rank 128 \
        --gpu-memory-utilization $GPU_MEM --host 0.0.0.0 2>&1 | tee -a "$EVAL_LOG"
    
    # Wait for vLLM to be ready (max 3 min)
    echo "[$(date)] Waiting for vLLM..." | tee -a "$EVAL_LOG"
    for i in {1..90}; do
        if curl -s "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
            echo "[$(date)] vLLM ready!" | tee -a "$EVAL_LOG"
            break
        fi
        sleep 2
    done
    
    # Run eval
    local OUTPUT_FILE="$EXP_DIR/results/eval_${ADAPTER_PREFIX}_step${STEP}_${TIMESTAMP}.json"
    echo "[$(date)] Running eval..." | tee -a "$EVAL_LOG"
    cd "$EXP_DIR"
    python3 scripts/step_5_eval_score.py \
        --model ckpt_eval \
        --port $VLLM_PORT \
        --output "$OUTPUT_FILE" 2>&1 | tee -a "$EVAL_LOG"
    
    # Extract accuracy from output
    ACCURACY=$(python3 -c "import json; d=json.load(open('$OUTPUT_FILE')); print(f'{d[\"accuracy\"]:.2f}%')" 2>/dev/null || echo "N/A")
    echo "[$(date)] Checkpoint-$STEP Accuracy: $ACCURACY" | tee -a "$EVAL_LOG"
    
    # Cleanup vLLM
    docker rm -f vllm_auto_eval 2>/dev/null || true
    
    # Update last eval step
    echo "$STEP" > "$LAST_EVAL_FILE"
}

# Main monitor loop
echo "[$(date)] Starting monitor loop..." | tee -a "$EVAL_LOG"
while true; do
    CURRENT_STEP=$(get_latest_checkpoint)
    
    if [ -z "$CURRENT_STEP" ] || [ "$CURRENT_STEP" = "0" ]; then
        echo "[$(date)] No checkpoints found yet, waiting..."
        sleep 60
        continue
    fi
    
    # Calculate next eval step
    NEXT_EVAL_STEP=$(( (LAST_EVAL_STEP / EVAL_INTERVAL + 1) * EVAL_INTERVAL ))
    
    if [ "$CURRENT_STEP" -ge "$NEXT_EVAL_STEP" ]; then
        # Find the actual checkpoint at or after NEXT_EVAL_STEP
        EVAL_STEP=$(ls -1d "$PROGRESS_DIR"/checkpoint-* 2>/dev/null | \
            sed 's/.*checkpoint-//' | sort -n | \
            awk -v target="$NEXT_EVAL_STEP" '$1 >= target {print; exit}')
        
        if [ -n "$EVAL_STEP" ]; then
            echo "[$(date)] New checkpoint ready: $EVAL_STEP (next eval target was $NEXT_EVAL_STEP)"
            run_eval "$EVAL_STEP"
            LAST_EVAL_STEP=$EVAL_STEP
        fi
    else
        echo "[$(date)] Current: $CURRENT_STEP, Next eval at: $NEXT_EVAL_STEP, waiting..." 
    fi
    
    # Check if training is still running
    if ! pgrep -f "step_4_1_training_model.py.*--track $TRACK" > /dev/null 2>&1; then
        echo "[$(date)] Training appears to have finished. Running final eval..." | tee -a "$EVAL_LOG"
        FINAL_STEP=$(get_latest_checkpoint)
        if [ "$FINAL_STEP" -gt "$LAST_EVAL_STEP" ]; then
            run_eval "$FINAL_STEP"
        fi
        echo "[$(date)] Monitor complete." | tee -a "$EVAL_LOG"
        exit 0
    fi
    
    sleep 60
done
