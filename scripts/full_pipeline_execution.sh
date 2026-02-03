#!/bin/bash
# ============================================================================
# Exp013: Full Pipeline Execution Script
# ============================================================================
# Runs the complete pipeline from data preparation to submission generation.
#
# Usage:
#   ./full_pipeline_execution.sh               # Run everything
#   ./full_pipeline_execution.sh --skip-data   # Skip data prep, start from training
#   ./full_pipeline_execution.sh --track 3     # Only train specific track
#   ./full_pipeline_execution.sh --smoke-test  # Quick test: Track2, ~100 steps, skip data
#
# ============================================================================

set -e  # Exit on error

# Navigate to experiment directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXP_DIR"

echo "============================================================================"
echo "Exp013: Full Pipeline Execution"
echo "============================================================================"
echo "Experiment dir: $EXP_DIR"
echo "Start time: $(date)"
echo "============================================================================"

# Parse arguments first (before prereq checks that depend on them)
SKIP_DATA=false
TRACK_ONLY=""
SKIP_TRAIN=false
SMOKE_TEST=false
EVAL_ONLY=false
VLLM_PORT=8001

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --eval-only)
            # Skip data prep and training, run eval + submission
            EVAL_ONLY=true
            SKIP_DATA=true
            SKIP_TRAIN=true
            shift
            ;;
        --track)
            TRACK_ONLY="$2"
            shift 2
            ;;
        --port)
            VLLM_PORT="$2"
            shift 2
            ;;
        --smoke-test)
            # Quick training validation: Track 2, 100 steps, skip data prep
            SMOKE_TEST=true
            SKIP_DATA=true
            TRACK_ONLY="2"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --skip-data    Skip data preparation (Step 1-3)"
            echo "  --skip-train   Skip training (Step 4)"
            echo "  --eval-only    Skip data+train, run eval+submission only"
            echo "  --track N      Only process track N (1, 2, or 3)"
            echo "  --port PORT    vLLM port (default: 8001)"
            echo "  --smoke-test   Quick test: Track 2, ~100 steps"
            exit 1
            ;;
    esac
done

# ============================================================================
# Prerequisite Checks
# ============================================================================
echo ""
echo "[Prereq] Checking dependencies..."

# Check Python3
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.8+."
    exit 1
fi

# Check required Python packages
MISSING_PKGS=""
python3 -c "import pandas" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS pandas"
python3 -c "import numpy" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS numpy"
python3 -c "import sklearn" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS scikit-learn"
python3 -c "import requests" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS requests"
python3 -c "import aiohttp" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS aiohttp"

if [ -n "$MISSING_PKGS" ]; then
    echo "[Prereq] Installing missing packages:$MISSING_PKGS"
    pip3 install -q $MISSING_PKGS || pip install -q $MISSING_PKGS
fi

# Check for training dependencies (only if training will run)
if [ "$SKIP_TRAIN" = false ]; then
    TRAIN_MISSING=""
    python3 -c "import torch" 2>/dev/null || TRAIN_MISSING="$TRAIN_MISSING torch"
    python3 -c "import transformers" 2>/dev/null || TRAIN_MISSING="$TRAIN_MISSING transformers"
    python3 -c "import datasets" 2>/dev/null || TRAIN_MISSING="$TRAIN_MISSING datasets"
    python3 -c "import trl" 2>/dev/null || TRAIN_MISSING="$TRAIN_MISSING trl"
    python3 -c "import peft" 2>/dev/null || TRAIN_MISSING="$TRAIN_MISSING peft"
    
    if [ -n "$TRAIN_MISSING" ]; then
        echo "WARNING: Missing training packages:$TRAIN_MISSING"
        echo "         Install with: pip install$TRAIN_MISSING"
        echo "         Or use --skip-train to skip training phase."
    fi
fi

# Check for required data files
if [ "$SKIP_DATA" = false ]; then
    if [ ! -f "challenge_data/phase_2_test.csv" ]; then
        echo "ERROR: challenge_data/phase_2_test.csv not found!"
        echo "       Please ensure challenge data is in place."
        exit 1
    fi
    if [ ! -f "challenge_data/train.csv" ]; then
        echo "ERROR: challenge_data/train.csv not found!"
        exit 1
    fi
fi

echo "[Prereq] All checks passed ✓"

# ============================================================================
# Phase 1: Data Preparation
# ============================================================================

if [ "$SKIP_DATA" = false ]; then
    echo ""
    echo "============================================================================"
    echo "PHASE 1: Data Preparation"
    echo "============================================================================"
    
    echo ""
    echo "[Step 1] Preprocessing & Tagging..."
    python3 scripts/step_1_preprocess_tagging.py
    
    echo ""
    echo "[Step 1.5] Generating synthetic Q&A for non-5G questions..."
    echo "(Requires vLLM with Qwen3-32B - will skip if not available)"
    python3 scripts/step_1.5_synthetic_question_answer.py || echo "    Skipped - vLLM not available"
    
    echo ""
    echo "[Step 2] Generating enriched training dataset..."
    python3 scripts/step_2_data_enrichment.py
    
    echo ""
    echo "[Step 3] Verifying training data..."
    python3 scripts/step_3_data_verification.py
    
    echo ""
    echo "Phase 1 Complete. Training data ready."
    echo "============================================================================"
fi

# ============================================================================
# Phase 2: Training (Track1 -> Track2 -> Track3)
# ============================================================================

if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "============================================================================"
    echo "PHASE 2: Model Training"
    echo "============================================================================"
    
    # Training uses track-specific epochs from config:
    # Track1 (32B): 2 epochs, Track2 (7B): 4 epochs, Track3 (1.5B): 25 epochs
    
    # Train Track2 (7B) first
    if [ -z "$TRACK_ONLY" ] || [ "$TRACK_ONLY" = "2" ]; then
        echo ""
        if [ "$SMOKE_TEST" = true ]; then
            echo "[Step 4] SMOKE TEST: Training Track2 (Qwen2.5-7B-Instruct) for ~100 steps..."
            echo "Start: $(date)"
            python3 scripts/step_4_training.py \
                --track 2 \
                --epochs 0.03 \
                --data_path data/training_dataset.jsonl
            echo "Track2 Smoke Test Complete: $(date)"
        else
            echo "[Step 4] Training Track2 (Qwen2.5-7B-Instruct) - using default 4 epochs..."
            echo "Start: $(date)"
            python3 scripts/step_4_training.py \
                --track 2 \
                --data_path data/training_dataset.jsonl
            echo "Track2 Complete: $(date)"
        fi
    fi
    
    # Train Track1 (32B)
    if [ -z "$TRACK_ONLY" ] || [ "$TRACK_ONLY" = "1" ]; then
        echo ""
        echo "[Step 4] Training Track1 (Qwen3-32B) - using default 2 epochs..."
        echo "Start: $(date)"
        python3 scripts/step_4_training.py \
            --track 1 \
            --data_path data/training_dataset.jsonl
        echo "Track1 Complete: $(date)"
    fi
    
    # Train Track3 (1.5B)
    if [ -z "$TRACK_ONLY" ] || [ "$TRACK_ONLY" = "3" ]; then
        echo ""
        echo "[Step 4] Training Track3 (Qwen2.5-1.5B-Instruct) - using default 25 epochs..."
        echo "Start: $(date)"
        python3 scripts/step_4_training.py \
            --track 3 \
            --data_path data/training_dataset.jsonl
        echo "Track3 Complete: $(date)"
    fi
    
    echo ""
    echo "Phase 2 Complete. All tracks trained."
    echo "============================================================================"
fi

# ============================================================================
# Phase 3: Evaluation (using vLLM - Fast)
# ============================================================================

echo ""
echo "============================================================================"
echo "PHASE 3: Evaluation (vLLM)"
echo "============================================================================"

# Evaluate trained adapters using vLLM
for TRACK in 1 2 3; do
    TRACK_NAMES=("" "track1_32b" "track2_7b" "track3_1.5b")
    TRACK_NAME="${TRACK_NAMES[$TRACK]}"
    
    # Find latest adapter with timestamp (pattern: track*_YYYYMMDD_HHMMSS)
    LATEST_ADAPTER=$(ls -td adapters/${TRACK_NAME}_* 2>/dev/null | head -1)
    
    if [ -z "$TRACK_ONLY" ] || [ "$TRACK_ONLY" = "$TRACK" ]; then
        if [ -n "$LATEST_ADAPTER" ] && [ -d "$LATEST_ADAPTER" ]; then
            ADAPTER_NAME=$(basename "$LATEST_ADAPTER")
            echo ""
            echo "[Step 5] Evaluating $ADAPTER_NAME with vLLM..."
            bash scripts/step_5_eval_vllm_and_save.sh "$ADAPTER_NAME"
        else
            echo ""
            echo "[Step 5] Skipping $TRACK_NAME - adapter not found"
        fi
    fi
done

echo ""
echo "Phase 3 Complete. Evaluation results saved to results/"
echo "============================================================================"

# ============================================================================
# Phase 4: Submission Generation
# ============================================================================

echo ""
echo "============================================================================"
echo "PHASE 4: Submission Generation"
echo "============================================================================"

# Find latest eval results for each track
TRACK1_EVALS=""
TRACK2_EVALS=""
TRACK3_EVALS=""

echo ""
echo "[Step 6] Finding latest eval results..."

# For each track, find the 4 most recent run results (or single result)
for TRACK in 1 2 3; do
    if [ -n "$TRACK_ONLY" ] && [ "$TRACK_ONLY" != "$TRACK" ]; then
        continue
    fi
    
    case $TRACK in
        1) PATTERN="eval_*32b*" ;;
        2) PATTERN="eval_*7b*" ;;
        3) PATTERN="eval_*1.5b*" ;;
    esac
    
    # Find latest 4 run files or single eval
    LATEST_EVALS=$(ls -t results/$PATTERN 2>/dev/null | head -4 | tr '\n' ',' | sed 's/,$//')
    
    if [ -n "$LATEST_EVALS" ]; then
        echo "  Track $TRACK: $LATEST_EVALS"
        case $TRACK in
            1) TRACK1_EVALS="$LATEST_EVALS" ;;
            2) TRACK2_EVALS="$LATEST_EVALS" ;;
            3) TRACK3_EVALS="$LATEST_EVALS" ;;
        esac
    else
        echo "  Track $TRACK: No eval results found"
    fi
done

# Build submission command
echo ""
echo "[Step 6] Generating submission CSV..."

SUBMISSION_CMD="python3 scripts/step_6_submission_generation.py"

# Check if we have 4-run results (filenames contain 'run')
if echo "$TRACK1_EVALS" | grep -q "run"; then
    [ -n "$TRACK1_EVALS" ] && SUBMISSION_CMD="$SUBMISSION_CMD --track1-runs $TRACK1_EVALS"
else
    FIRST_EVAL=$(echo "$TRACK1_EVALS" | cut -d',' -f1)
    [ -n "$FIRST_EVAL" ] && SUBMISSION_CMD="$SUBMISSION_CMD --track1-eval $(basename $FIRST_EVAL)"
fi

if echo "$TRACK2_EVALS" | grep -q "run"; then
    [ -n "$TRACK2_EVALS" ] && SUBMISSION_CMD="$SUBMISSION_CMD --track2-runs $TRACK2_EVALS"
else
    FIRST_EVAL=$(echo "$TRACK2_EVALS" | cut -d',' -f1)
    [ -n "$FIRST_EVAL" ] && SUBMISSION_CMD="$SUBMISSION_CMD --track2-eval $(basename $FIRST_EVAL)"
fi

if echo "$TRACK3_EVALS" | grep -q "run"; then
    [ -n "$TRACK3_EVALS" ] && SUBMISSION_CMD="$SUBMISSION_CMD --track3-runs $TRACK3_EVALS"
else
    FIRST_EVAL=$(echo "$TRACK3_EVALS" | cut -d',' -f1)
    [ -n "$FIRST_EVAL" ] && SUBMISSION_CMD="$SUBMISSION_CMD --track3-eval $(basename $FIRST_EVAL)"
fi

echo "Running: $SUBMISSION_CMD"
$SUBMISSION_CMD

echo ""
echo "Phase 4 Complete. Submission saved to submissions/"
echo "============================================================================"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================================"
echo "PIPELINE COMPLETE"
echo "============================================================================"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - Training data: data/training_dataset.jsonl"
echo "  - Adapters: adapters/track{1,2,3}_*/"
echo "  - Eval results: results/"
echo "  - Submissions: submissions/"
echo ""
ls -lt submissions/*.csv 2>/dev/null | head -3
echo "============================================================================"
