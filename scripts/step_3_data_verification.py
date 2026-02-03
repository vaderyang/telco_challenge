#!/usr/bin/env python3
"""
Verify training data quality:
1. No Phase 2 question leakage into training set
2. No Phase 1 question leakage (or report if expected)
3. Training data format validation

Usage:
    python3 step_3_data_verification.py [--data DATA]
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def load_jsonl(path: Path) -> list:
    """Load JSONL file."""
    records = []
    if not path.exists():
        return records
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description='Step 3: Training Data Verification')
    parser.add_argument('--data', default='../data/training_dataset.jsonl',
                        help='Path to training dataset')
    parser.add_argument('--phase2', default='../challenge_data/phase_2_test.csv',
                        help='Path to phase_2_test.csv')
    parser.add_argument('--phase1', default='../challenge_data/phase_1_test.csv',
                        help='Path to phase_1_test.csv')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    print("=" * 70)
    print("Step 3: Training Data Verification")
    print("=" * 70)
    
    # Load training data
    data_path = script_dir / args.data
    if not data_path.exists():
        print(f"\n[ERROR] Training data not found: {data_path}")
        print("        Run step_2_data_enrichment.py first.")
        return 1
    
    print(f"\n[1] Loading training data: {data_path}")
    training_data = load_jsonl(data_path)
    print(f"    Total samples: {len(training_data)}")
    
    # Load Phase 2 questions (hash for comparison)
    print(f"\n[2] Loading Phase 2 test questions...")
    phase2_path = script_dir / args.phase2
    phase2_hashes = {}
    if phase2_path.exists():
        with open(phase2_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_hash = hashlib.md5(row['question'].encode()).hexdigest()
                phase2_hashes[q_hash] = row['ID']
        print(f"    Phase 2 questions loaded: {len(phase2_hashes)}")
    else:
        print(f"    [WARNING] Phase 2 test file not found: {phase2_path}")
    
    # Load Phase 1 questions
    print(f"\n[3] Loading Phase 1 test questions...")
    phase1_path = script_dir / args.phase1
    phase1_hashes = {}
    if phase1_path.exists():
        with open(phase1_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_hash = hashlib.md5(row['question'].encode()).hexdigest()
                phase1_hashes[q_hash] = row['ID']
        print(f"    Phase 1 questions loaded: {len(phase1_hashes)}")
    else:
        print(f"    [WARNING] Phase 1 test file not found: {phase1_path}")
    
    # Check for leakage
    print(f"\n[4] Checking for data leakage...")
    phase2_leaks = 0
    phase1_leaks = 0
    format_errors = 0
    
    for rec in training_data:
        # Find user message
        user_msg = None
        for msg in rec.get('input', []):
            if msg.get('role') == 'user':
                user_msg = msg.get('content', '')
                break
        
        if not user_msg:
            format_errors += 1
            continue
        
        q_hash = hashlib.md5(user_msg.encode()).hexdigest()
        
        if q_hash in phase2_hashes:
            phase2_leaks += 1
        if q_hash in phase1_hashes:
            phase1_leaks += 1
    
    # Results
    print(f"\n{'=' * 70}")
    print("VERIFICATION RESULTS")
    print(f"{'=' * 70}")
    
    # Phase 2 leakage check
    if phase2_leaks > 0:
        print(f"\n❌ [CRITICAL] Phase 2 EXACT match: {phase2_leaks} samples found!")
        print(f"   This is a data leakage issue and must be fixed.")
    else:
        print(f"\n✅ [PASS] Phase 2 leakage check: No matches found")
    
    # Phase 1 leakage check (may be expected if using phase1 for training)
    if phase1_leaks > 0:
        print(f"\n⚠️  [INFO] Phase 1 matches: {phase1_leaks} samples")
        print(f"   This may be expected if Phase 1 data is used for training.")
    else:
        print(f"\n✅ [PASS] Phase 1 leakage check: No exact matches")
    
    # Format validation
    if format_errors > 0:
        print(f"\n⚠️  [WARNING] Format errors: {format_errors} samples missing user message")
    else:
        print(f"\n✅ [PASS] Format validation: All samples have user message")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Training samples: {len(training_data)}")
    print(f"Phase 2 leaks: {phase2_leaks}")
    print(f"Phase 1 overlaps: {phase1_leaks}")
    print(f"Format errors: {format_errors}")
    
    if phase2_leaks == 0:
        print("\n✅ Training data verification PASSED")
        return 0
    else:
        print("\n❌ Training data verification FAILED")
        return 1


if __name__ == '__main__':
    exit(main())
