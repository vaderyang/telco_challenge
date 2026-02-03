#!/usr/bin/env python3
"""
Step 1: Preprocessing & Tagging

Classify phase_2_test.csv questions into categories (in memory):
- Category A (600Mbps): 5G throughput below 600Mbps questions (ML/Rule tagged)
- Category B (100Mbps): 5G throughput below 100Mbps questions
- Category C (non-5G): All other questions (math, history, etc.)

Output: tagged_question_answer.jsonl (for enhanced training data enrichment)

Usage:
    python3 step_1_preprocess_tagging.py [--input INPUT] [--output OUTPUT]
"""

import argparse
import csv
import json
import re
from pathlib import Path


def classify_question(question: str) -> str:
    """Classify a question into category A, B, or C.
    
    Returns:
        'A' for 600Mbps 5G questions
        'B' for 100Mbps 5G questions
        'C' for non-5G/other questions
    """
    q_lower = question.lower()
    
    # Check for 600Mbps 5G pattern (Category A)
    if '600mbps' in q_lower or '600 mbps' in q_lower:
        if '5g' in q_lower or 'nr' in q_lower or 'gnodeb' in q_lower:
            return 'A'
    
    # Check for 100Mbps pattern (Category B)
    if '100mbps' in q_lower or '100 mbps' in q_lower:
        if '5g' in q_lower or 'nr' in q_lower or 'gnodeb' in q_lower or 'throughput' in q_lower:
            return 'B'
    
    # Default: Category C (non-5G / general questions)
    return 'C'


def load_phase2_with_categories(input_path: Path) -> dict:
    """Load phase_2_test.csv and classify questions in memory.
    
    Returns:
        dict with keys 'A', 'B', 'C' containing lists of question dicts
    """
    rows_by_category = {'A': [], 'B': [], 'C': []}
    
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = classify_question(row['question'])
            rows_by_category[category].append(row)
    
    return rows_by_category


def main():
    parser = argparse.ArgumentParser(description='Step 1: Preprocessing & Tagging')
    parser.add_argument('--input', default='../challenge_data/phase_2_test.csv',
                        help='Path to phase_2_test.csv')
    parser.add_argument('--output', default='../data/tagged_question_answer.jsonl',
                        help='Output JSONL file for tagged data')
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input
    output_dir = (script_dir / args.output).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Step 1: Preprocessing & Tagging")
    print("=" * 70)
    
    # Read and classify (in memory, no CSV output)
    print(f"\n[1] Reading {input_path}...")
    rows_by_category = load_phase2_with_categories(input_path)
    
    total = sum(len(rows) for rows in rows_by_category.values())
    print(f"    Total questions: {total}")
    
    # Summary
    print(f"\n[2] Category classification (in memory):")
    for cat in ['A', 'B', 'C']:
        pct = len(rows_by_category[cat]) / total * 100 if total > 0 else 0
        print(f"    Category {cat}: {len(rows_by_category[cat])} questions ({pct:.1f}%)")
    
    # Note: This script now only does classification.
    # The actual tagging (generating answers) should be done by training a model
    # or using external inference. For now, output an empty placeholder file.
    output_path = script_dir / args.output
    
    # If tagged data already exists, keep it; otherwise note it's empty
    if output_path.exists():
        print(f"\n[3] Tagged data already exists: {output_path}")
        with open(output_path) as f:
            count = sum(1 for line in f if line.strip())
        print(f"    Existing samples: {count}")
    else:
        print(f"\n[3] No tagged data yet. Use inference to generate tagged_question_answer.jsonl")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Category A (600Mbps 5G): {len(rows_by_category['A'])} questions")
    print(f"Category B (100Mbps 5G): {len(rows_by_category['B'])} questions")
    print(f"Category C (Non-5G/Other): {len(rows_by_category['C'])} questions")
    print(f"Total: {total} questions")
    print(f"{'=' * 70}")
    
    # Show sample from each category
    print("\n[4] Sample questions from each category:")
    for cat in ['A', 'B', 'C']:
        rows = rows_by_category[cat]
        if rows:
            sample = rows[0]
            q_preview = sample['question'][:200].replace('\n', ' ')
            print(f"\n    Category {cat} sample (ID: {sample['ID']}):")
            print(f"    {q_preview}...")


if __name__ == '__main__':
    main()
