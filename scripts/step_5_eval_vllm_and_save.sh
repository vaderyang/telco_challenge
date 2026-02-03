#!/bin/bash
# ============================================================================
# Step 5: Evaluation using vLLM (Fast)
# ============================================================================
# Starts vLLM with the trained adapter, runs evaluation, and saves results.
#
# Usage:
#   ./step_5_eval_vllm_and_save.sh track2_7b           # Evaluate specific track
#   ./step_5_eval_vllm_and_save.sh                     # Auto-detect latest adapter
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXP_DIR"

# Configuration
VLLM_PORT=8001
GPU_MEMORY_UTIL=0.4

# Parse arguments
TRACK_NAME="${1:-}"

echo "============================================================================"
echo "Step 5: Evaluation using vLLM"
echo "============================================================================"

# Determine track to evaluate
if [ -z "$TRACK_NAME" ]; then
    # Auto-detect: find most recently modified adapter
    LATEST_ADAPTER=$(ls -t adapters/*/training_info.json 2>/dev/null | head -1)
    if [ -z "$LATEST_ADAPTER" ]; then
        echo "Error: No adapters found in adapters/"
        exit 1
    fi
    TRACK_NAME=$(dirname "$LATEST_ADAPTER" | xargs basename)
    echo "Auto-detected adapter: $TRACK_NAME"
fi

ADAPTER_PATH="$EXP_DIR/adapters/$TRACK_NAME"
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "Error: Adapter not found: $ADAPTER_PATH"
    exit 1
fi

# Determine base model from training info
BASE_MODEL=$(python3 -c "import json; print(json.load(open('$ADAPTER_PATH/training_info.json'))['base_model'])")
echo "Base model: $BASE_MODEL"
echo "Adapter: $ADAPTER_PATH"

# Stop any existing vLLM container on this port
echo ""
echo "[1] Stopping any existing vLLM container..."
docker stop vllm_eval 2>/dev/null || true
docker rm vllm_eval 2>/dev/null || true

# Start vLLM with adapter
echo ""
echo "[2] Starting vLLM with adapter..."

# Determine max-model-len based on track (32B needs 16000 for long questions)
if echo "$TRACK_NAME" | grep -qi "32b\|track1"; then
    MAX_MODEL_LEN=16000
    GPU_MEMORY_UTIL=0.85  # 32B needs more memory
    echo "    Using max-model-len=$MAX_MODEL_LEN for 32B model"
else
    MAX_MODEL_LEN=8000
fi

docker run -d --gpus all --name vllm_eval \
    -v "$BASE_MODEL":/model \
    -v "$ADAPTER_PATH":/adapter \
    -p $VLLM_PORT:8000 \
    vllm/vllm-openai:latest \
    --model /model \
    --enable-lora \
    --lora-modules "$TRACK_NAME=/adapter" \
    --max-model-len $MAX_MODEL_LEN \
    --max-lora-rank 128 \
    --gpu-memory-utilization $GPU_MEMORY_UTIL

echo "    Waiting for vLLM to start..."
for i in {1..120}; do
    if curl -s "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
        echo "    vLLM ready after ${i}s"
        break
    fi
    sleep 1
done

# Test connection
TEST_RESP=$(curl -s "http://localhost:$VLLM_PORT/v1/models" || echo "error")
if echo "$TEST_RESP" | grep -q "error"; then
    echo "Error: vLLM failed to start"
    docker logs vllm_eval
    exit 1
fi
echo "    vLLM is running: $(echo "$TEST_RESP" | python3 -c 'import sys,json; d=json.load(sys.stdin); print([m["id"] for m in d["data"]])')"

# Run evaluation using Tagged Data as ground truth
echo ""
echo "[3] Running evaluation (using Tagged Data as ground truth)..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="results/eval_${TRACK_NAME}_${TIMESTAMP}.json"
mkdir -p results

python3 scripts/step_5_eval_score.py \
    --model "$TRACK_NAME" \
    --port $VLLM_PORT \
    --output "$RESULT_FILE" \
    --workers 16

# Show results
echo ""
echo "[4] Results saved to: $RESULT_FILE"
if [ -f "$RESULT_FILE" ]; then
    python3 -c "import json; r=json.load(open('$RESULT_FILE')); print(f\"Accuracy: {r['correct']}/{r['total']} ({r['accuracy']:.2f}%)\")"
fi

# Cleanup
echo ""
echo "[5] Stopping vLLM container..."
docker stop vllm_eval
docker rm vllm_eval

echo ""
echo "============================================================================"
echo "EVALUATION COMPLETE"
echo "============================================================================"
echo "Track: $TRACK_NAME"
echo "Results: $RESULT_FILE"
echo "============================================================================"
