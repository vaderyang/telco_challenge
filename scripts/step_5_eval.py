#!/usr/bin/env python3
"""
Step 5: Evaluation

Two modes:
1. validate: 90/10 split on training data to measure model accuracy
2. inference: Use phase_2_test.csv to generate predictions for submission

Supports 4 independent evaluation runs for submission (run_id 1-4).

Usage:
    python3 step_5_eval.py --model track2_7b --port 8001 --mode validate
    python3 step_5_eval.py --model track2_7b --port 8001 --mode inference --run_all
"""

# Auto-install missing dependencies
import subprocess
import sys

def _ensure_deps(packages):
    """Check and install missing packages."""
    for pkg in packages:
        pkg_name = pkg.split('>=')[0].split('==')[0].strip()
        try:
            __import__(pkg_name)
        except ImportError:
            print(f"[auto-install] Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

_ensure_deps(['requests'])

import argparse
import csv
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests


def extract_answer(text):
    """Extract answer from model output."""
    if not text:
        return None
    
    # Match boxed{...} with any number of backslashes
    all_boxed = re.findall(r'\\*boxed\{([^}]+)\}', text)
    
    # From the end, find first short code
    for item in reversed(all_boxed):
        item = item.strip()
        if re.match(r'^[A-Za-z]?\d+$', item) or re.match(r'^[A-Za-z]$', item):
            return normalize_answer(item)
    
    # Fallback
    direct_match = re.findall(r'\\*boxed\{([A-Za-z]?\d+|[A-Za-z])\}', text)
    if direct_match:
        return normalize_answer(direct_match[-1])
    
    return None


def normalize_answer(ans):
    """Normalize answer: C1, M5, P6, etc. -> numeric only."""
    if not ans:
        return ''
    ans = ans.strip()
    match = re.match(r'^[A-Za-z]?(\d+)$', ans)
    if match:
        return match.group(1)
    return ans.upper()


# System prompt matching training data
SYSTEM_PROMPT = "You are a 5G network expert. Analyze the data and identify the root cause."


# Temperature settings for 4 independent runs
RUN_TEMPERATURES = {
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
}


def query_vllm(prompt, model, port, temperature=0.0, seed=None):
    """Query vLLM server with configurable temperature and seed."""
    try:
        # For Qwen3-32B models, add /no_think to disable thinking mode
        user_content = prompt
        if '32b' in model.lower() or '32B' in model:
            user_content = prompt + " /no_think"
        
        request_body = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_content}
            ],
            'max_tokens': 4096,
            'temperature': temperature
        }
        if seed is not None:
            request_body['seed'] = seed
        
        resp = requests.post(
            f'http://localhost:{port}/v1/chat/completions',
            json=request_body,
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        return f'ERROR: {e}'


def load_training_data(exp_dir):
    """Load training_dataset.jsonl for validation mode."""
    training_path = exp_dir / 'data/training_dataset.jsonl'
    records = []
    if training_path.exists():
        with open(training_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def load_phase2_questions(exp_dir):
    """Load phase_2_test.csv for inference mode."""
    phase2_path = exp_dir / 'challenge_data/phase_2_test.csv'
    questions = {}
    if phase2_path.exists():
        with open(phase2_path) as f:
            for row in csv.DictReader(f):
                questions[row['ID']] = row['question']
    return questions


def run_validate_mode(args, exp_dir):
    """90/10 split validation on training data."""
    print("\n[1] Loading training data...")
    records = load_training_data(exp_dir)
    print(f"    Total training samples: {len(records)}")
    
    if not records:
        print("    ERROR: No training data found.")
        return 1
    
    # 90/10 split
    random.seed(42)
    random.shuffle(records)
    split_idx = int(len(records) * 0.9)
    eval_records = records[split_idx:]
    print(f"    Using 10% for validation: {len(eval_records)} samples")
    
    if args.max_samples:
        eval_records = eval_records[:args.max_samples]
        print(f"    Limited to: {len(eval_records)} samples")
    
    # Test vLLM connection
    print(f"\n[2] Testing vLLM connection (port {args.port})...")
    test_resp = query_vllm("What is 2+2? Answer in \\boxed{}", args.model, args.port)
    if 'ERROR' in test_resp:
        print(f"    Error: {test_resp}")
        return 1
    print(f"    Connection OK")
    
    # Evaluate
    print(f"\n[3] Evaluating {len(eval_records)} samples...")
    
    def process_record(rec):
        # Find user message
        user_msg = None
        for msg in rec.get('input', []):
            if msg.get('role') == 'user':
                user_msg = msg.get('content', '')
                break
        if not user_msg:
            return None, None, None, False
        
        # Get expected answer
        output = rec.get('output', [{}])[0].get('content', '')
        expected = extract_answer(output)
        expected_norm = normalize_answer(expected) if expected else ''
        
        # Query model
        response = query_vllm(user_msg, args.model, args.port)
        pred = extract_answer(response)
        pred_norm = normalize_answer(pred) if pred else ''
        
        is_correct = pred_norm == expected_norm
        return rec.get('id', ''), pred_norm, expected_norm, is_correct
    
    results = []
    correct = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_record, rec) for rec in eval_records]
        for i, future in enumerate(as_completed(futures)):
            qid, pred, expected, is_correct = future.result()
            if qid is not None:
                results.append({
                    'id': qid,
                    'pred': pred,
                    'expected': expected,
                    'correct': is_correct,
                })
                if is_correct:
                    correct += 1
            
            if (i + 1) % 50 == 0:
                acc = correct / (i + 1) * 100
                print(f"    [{i+1}/{len(eval_records)}] Accuracy: {correct}/{i+1} ({acc:.1f}%)")
    
    accuracy = correct / len(results) * 100 if results else 0
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = exp_dir / f'results/validate_{args.model}_{timestamp}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        'mode': 'validate',
        'model': args.model,
        'timestamp': timestamp,
        'split': '90/10',
        'total': len(results),
        'correct': correct,
        'accuracy': accuracy,
        'details': results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    # Summary
    print(f"\n{'=' * 70}")
    print("VALIDATION RESULTS (90/10 Split)")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Samples: {len(results)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results: {output_path}")
    print(f"{'=' * 70}")
    
    return 0


def run_inference_mode(args, exp_dir):
    """Inference on phase_2_test.csv for submission."""
    print("\n[1] Loading Phase 2 test questions...")
    questions = load_phase2_questions(exp_dir)
    print(f"    Total questions: {len(questions)}")
    
    if not questions:
        print("    ERROR: No Phase 2 test questions found.")
        return 1
    
    question_ids = list(questions.keys())
    if args.max_samples:
        question_ids = question_ids[:args.max_samples]
        print(f"    Limited to: {len(question_ids)} samples")
    
    # Test vLLM connection
    print(f"\n[2] Testing vLLM connection (port {args.port})...")
    test_resp = query_vllm("What is 2+2? Answer in \\boxed{}", args.model, args.port)
    if 'ERROR' in test_resp:
        print(f"    Error: {test_resp}")
        return 1
    print(f"    Connection OK")
    
    # Determine runs
    if args.run_all:
        run_ids = [1, 2, 3, 4]
    elif args.run_id:
        run_ids = [args.run_id]
    else:
        run_ids = [1]  # Default to single run
    
    print(f"\n[3] Running {len(run_ids)} inference run(s)...")
    
    all_results = []
    for run_id in run_ids:
        temperature = RUN_TEMPERATURES.get(run_id, 0.0)
        seed = 42 + run_id  # Different seed for each run
        
        print(f"\n{'=' * 70}")
        print(f"Run {run_id}/{len(run_ids)} - Temperature: {temperature}, Seed: {seed}")
        print(f"{'=' * 70}")
        
        def process_question(qid):
            question = questions.get(qid, '')
            response = query_vllm(question, args.model, args.port, temperature=temperature, seed=seed)
            pred = extract_answer(response)
            pred_norm = normalize_answer(pred) if pred else ''
            return qid, pred_norm, response
        
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_question, qid): qid for qid in question_ids}
            for i, future in enumerate(as_completed(futures)):
                qid, pred, response = future.result()
                results.append({
                    'id': qid,
                    'pred': pred,
                    'response': response[:500] if response else '',  # Truncate for storage
                })
                
                if (i + 1) % 100 == 0:
                    print(f"    Run {run_id} [{i+1}/{len(question_ids)}] completed")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = exp_dir / f'results/inference_{args.model}_run{run_id}_{timestamp}.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_data = {
            'mode': 'inference',
            'model': args.model,
            'run_id': run_id,
            'temperature': temperature,
            'seed': seed,
            'timestamp': timestamp,
            'total': len(results),
            'details': results,
        }
        
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"    Run {run_id} completed: {len(results)} predictions")
        print(f"    Saved: {output_path.name}")
        all_results.append((run_id, output_path))
    
    # Summary
    print(f"\n{'=' * 70}")
    print("INFERENCE RESULTS")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Questions: {len(question_ids)}")
    for run_id, output_path in all_results:
        print(f"  Run {run_id}: {output_path.name}")
    print(f"\nUse step_6_submission_generation.py to create submission CSV.")
    print(f"{'=' * 70}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='Step 5: Evaluation')
    parser.add_argument('--model', required=True, help='Model name in vLLM')
    parser.add_argument('--port', type=int, default=8001, help='vLLM port')
    parser.add_argument('--mode', choices=['validate', 'inference'], default='inference',
                        help='Evaluation mode: validate (90/10 split) or inference (phase_2_test)')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to evaluate')
    parser.add_argument('--workers', type=int, default=16, help='Parallel workers')
    parser.add_argument('--run_id', type=int, choices=[1, 2, 3, 4], default=None,
                        help='Run ID (1-4) for inference mode')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all 4 inference runs (inference mode only)')
    args = parser.parse_args()
    
    exp_dir = Path(__file__).parent.parent
    
    print("=" * 70)
    print(f"Step 5: Evaluation ({args.mode} mode)")
    print(f"Model: {args.model}")
    print("=" * 70)
    
    if args.mode == 'validate':
        return run_validate_mode(args, exp_dir)
    else:
        return run_inference_mode(args, exp_dir)


if __name__ == '__main__':
    sys.exit(main())
