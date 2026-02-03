#!/usr/bin/env python3
"""
Step 6: Submission Generation

Build submission CSV from inference results or pre-generated answers.
Supports 4 independent evaluation runs for each question.

Features:
- List available inference results with --list
- Select specific inference results for each track
- Use 4 independent runs for 4-row submission format
- Generate timestamped submission files

Usage:
    python3 step_6_submission_generation.py --list                           # List results
    python3 step_6_submission_generation.py --track 2                        # Use tagged data for Track 2
    python3 step_6_submission_generation.py --track2-runs run1.json,run2.json,run3.json,run4.json
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path


def extract_answer(text: str) -> str:
    """Extract answer from model output."""
    if not text:
        return ''
    boxed = re.findall(r'\\*boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    return text.strip()


def normalize_answer(ans: str) -> str:
    """Normalize answer for submission."""
    if not ans:
        return ''
    ans = ans.strip()
    match = re.match(r'^[A-Za-z]?(\d+)$', ans)
    if match:
        return match.group(1)
    return ans


def extract_option_prefix(question: str) -> str:
    """Extract the option prefix pattern from question (e.g., 'M', 'P', 'C', '').
    
    Looks for patterns like "M1:", "P1:", "C1:", "1:", etc. in the question.
    Returns the letter prefix used for options.
    """
    # Look for patterns like "\nM1:", "\nP1:", "\n1:", etc.
    patterns = re.findall(r'\n([A-Z]?)(\d+):', question)
    if patterns:
        return patterns[0][0]  # Get the letter prefix from first option
    return ''


def list_eval_results(results_dir: Path):
    """List all available eval results."""
    print("\n" + "=" * 70)
    print("Available Eval Results")
    print("=" * 70)
    
    if not results_dir.exists():
        print("  No results directory found.")
        return []
    
    results = []
    for f in sorted(results_dir.glob("eval_*.json"), reverse=True):
        try:
            with open(f) as fp:
                data = json.load(fp)
            results.append({
                'file': f.name,
                'model': data.get('model', 'unknown'),
                'accuracy': data.get('accuracy', 0),
                'total': data.get('total', 0),
                'correct': data.get('correct', 0),
                'timestamp': data.get('timestamp', ''),
            })
        except Exception:
            pass
    
    if not results:
        print("  No eval results found.")
        return results
    
    # Group by track
    track_results = {'track1_32b': [], 'track2_7b': [], 'track3_1.5b': []}
    for r in results:
        if 'track1' in r['model'] or '32b' in r['model'].lower():
            track_results['track1_32b'].append(r)
        elif 'track2' in r['model'] or '7b' in r['model'].lower():
            track_results['track2_7b'].append(r)
        elif 'track3' in r['model'] or '1.5b' in r['model'].lower():
            track_results['track3_1.5b'].append(r)
    
    for track, items in track_results.items():
        if items:
            print(f"\n  {track}:")
            for r in items[:5]:  # Show last 5
                print(f"    {r['file']}: {r['correct']}/{r['total']} ({r['accuracy']:.1f}%)")
    
    print("\n" + "=" * 70)
    return results


def load_answers_from_eval(eval_path: Path) -> dict:
    """Load answers from eval result JSON."""
    with open(eval_path) as f:
        data = json.load(f)
    
    answers = {}
    for item in data.get('details', []):
        qid = item.get('id', '')
        pred = item.get('pred', '')
        if qid and pred:
            answers[qid] = normalize_answer(pred)
    
    return answers


def load_4_run_answers(run_files: list, results_dir: Path) -> dict:
    """Load answers from 4 eval run files.
    
    Returns: {qid: [answer1, answer2, answer3, answer4]}
    """
    all_answers = {}
    
    for i, run_file in enumerate(run_files):
        run_path = results_dir / run_file if not Path(run_file).is_absolute() else Path(run_file)
        if not run_path.exists():
            run_path = results_dir / f"{run_file}.json"
        
        if not run_path.exists():
            print(f"    Warning: Run file not found: {run_file}")
            continue
        
        run_answers = load_answers_from_eval(run_path)
        print(f"    Run {i+1}: Loaded {len(run_answers)} answers from {run_path.name}")
        
        for qid, answer in run_answers.items():
            if qid not in all_answers:
                all_answers[qid] = ['', '', '', '']
            all_answers[qid][i] = answer
    
    return all_answers


def load_tagged_answers(exp_dir: Path) -> dict:
    """Load answers from tagged_question_answer.jsonl + category_C_answers.csv."""
    answers = {}
    
    # Load tagged (Category A + B)
    tagged_path = exp_dir / 'data/tagged_question_answer.jsonl'
    if tagged_path.exists():
        with open(tagged_path) as f:
            for line in f:
                record = json.loads(line)
                qid = record['id']
                output = record['output'][0]['content']
                ans = extract_answer(output)
                answers[qid] = normalize_answer(ans)
    
    # Load Category C
    category_c_path = exp_dir / 'data/category_C_answers.csv'
    if category_c_path.exists():
        with open(category_c_path) as f:
            for row in csv.DictReader(f):
                if row['answer']:
                    answers[row['ID']] = normalize_answer(row['answer'])
    
    return answers


def main():
    parser = argparse.ArgumentParser(description='Submission Generation')
    parser.add_argument('--list', action='store_true', help='List available eval results')
    parser.add_argument('--track', type=int, choices=[1, 2, 3], default=None,
                        help='Generate for specific track using tagged data')
    parser.add_argument('--track1-eval', type=str, help='Single eval result file for Track 1')
    parser.add_argument('--track2-eval', type=str, help='Single eval result file for Track 2')
    parser.add_argument('--track3-eval', type=str, help='Single eval result file for Track 3')
    parser.add_argument('--track1-runs', type=str, help='4 eval run files for Track 1 (comma-separated)')
    parser.add_argument('--track2-runs', type=str, help='4 eval run files for Track 2 (comma-separated)')
    parser.add_argument('--track3-runs', type=str, help='4 eval run files for Track 3 (comma-separated)')
    parser.add_argument('--output', default=None, help='Output submission CSV')
    args = parser.parse_args()
    
    exp_dir = Path(__file__).parent.parent
    results_dir = exp_dir / 'results'
    
    # Track column mapping
    TRACK_COLUMNS = {
        1: 'Qwen3-32B',
        2: 'Qwen2.5-7B-Instruct',
        3: 'Qwen2.5-1.5B-Instruct',
    }
    
    # List mode
    if args.list:
        list_eval_results(results_dir)
        return 0
    
    print("=" * 70)
    print("Step 6: Submission Generation")
    print("=" * 70)
    
    # Load Phase 2 test IDs and questions (for option prefix extraction)
    phase2_path = exp_dir / 'challenge_data/phase_2_test.csv'
    phase2_ids = []
    phase2_questions = {}
    phase2_prefixes = {}
    with open(phase2_path, 'r') as f:
        for row in csv.DictReader(f):
            qid = row['ID']
            question = row['question']
            phase2_ids.append(qid)
            phase2_questions[qid] = question
            phase2_prefixes[qid] = extract_option_prefix(question)
    print(f"\n[1] Phase 2 questions: {len(phase2_ids)}")
    
    # Determine answer sources for each track
    # track_answers: {track: {qid: answer}} for single eval
    # track_4run_answers: {track: {qid: [ans1, ans2, ans3, ans4]}} for 4-run eval
    track_answers = {1: {}, 2: {}, 3: {}}
    track_4run_answers = {1: {}, 2: {}, 3: {}}
    track_sources = {1: None, 2: None, 3: None}
    use_4runs = {1: False, 2: False, 3: False}
    
    # Load from 4-run eval results if specified
    for track, runs_arg in [(1, args.track1_runs), (2, args.track2_runs), (3, args.track3_runs)]:
        if runs_arg:
            run_files = [f.strip() for f in runs_arg.split(',')]
            if len(run_files) != 4:
                print(f"    Warning: Track {track} requires 4 run files, got {len(run_files)}")
            print(f"    Track {track}: Loading 4 independent runs...")
            track_4run_answers[track] = load_4_run_answers(run_files, results_dir)
            track_sources[track] = f"4-runs:{len(track_4run_answers[track])} questions"
            use_4runs[track] = True
    
    # Load from single eval results if specified
    for track, eval_arg in [(1, args.track1_eval), (2, args.track2_eval), (3, args.track3_eval)]:
        if eval_arg and not use_4runs[track]:
            eval_path = results_dir / eval_arg
            if not eval_path.exists():
                eval_path = results_dir / f"{eval_arg}.json"
            if eval_path.exists():
                track_answers[track] = load_answers_from_eval(eval_path)
                track_sources[track] = f"eval:{eval_path.name}"
                print(f"    Track {track}: Loaded {len(track_answers[track])} from {eval_path.name}")
    
    # Load from tagged data if --track specified
    if args.track:
        tagged_answers = load_tagged_answers(exp_dir)
        track_answers[args.track] = tagged_answers
        track_sources[args.track] = "tagged_data"
        print(f"    Track {args.track}: Loaded {len(tagged_answers)} from tagged data")
    
    # Check that at least one track has answers
    active_tracks = [t for t in [1, 2, 3] if track_answers[t] or track_4run_answers[t]]
    if not active_tracks:
        print("\nError: No answers loaded. Use --track, --trackN-eval, or --trackN-runs options.")
        print("       Run with --list to see available eval results.")
        return 1
    
    print(f"\n[2] Active tracks: {active_tracks}")
    for t in active_tracks:
        mode = "4-runs (independent)" if use_4runs[t] else "single-eval (repeated)"
        print(f"    Track {t}: {mode}")
    
    # Build submission
    print(f"\n[3] Building submission CSV...")
    submission_rows = []
    
    for qid in phase2_ids:
        prefix = phase2_prefixes.get(qid, '')  # Get option prefix for this question
        for i in range(1, 5):  # 4 repeats per question (1-indexed)
            row = {'ID': f"{qid}_{i}"}
            
            for track, col in TRACK_COLUMNS.items():
                if use_4runs[track] and qid in track_4run_answers[track]:
                    # Use independent answer for each row
                    answer = track_4run_answers[track][qid][i - 1]  # i is 1-4, index is 0-3
                    full_answer = f"{prefix}{answer}" if answer else ''
                    row[col] = f"\\\\boxed{{{full_answer}}}"
                elif track_answers[track]:
                    # Repeat same answer for all 4 rows
                    answer = track_answers[track].get(qid, '')
                    full_answer = f"{prefix}{answer}" if answer else ''
                    row[col] = f"\\\\boxed{{{full_answer}}}"
                else:
                    row[col] = 'placeholder'
            
            submission_rows.append(row)
    
    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output:
        output_path = exp_dir / args.output
    else:
        track_suffix = "_".join([f"t{t}" for t in active_tracks])
        run_mode = "4runs" if any(use_4runs.values()) else "single"
        output_path = exp_dir / f"submissions/submission_{track_suffix}_{run_mode}_{timestamp}.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'Qwen3-32B', 'Qwen2.5-7B-Instruct', 'Qwen2.5-1.5B-Instruct'])
        writer.writeheader()
        writer.writerows(submission_rows)
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUBMISSION GENERATED")
    print(f"{'=' * 70}")
    print(f"Output: {output_path}")
    print(f"Total rows: {len(submission_rows)} (= {len(phase2_ids)} questions × 4)")
    print("\nSources:")
    for track in [1, 2, 3]:
        src = track_sources[track] or "placeholder"
        mode = "(4 independent)" if use_4runs[track] else "(repeated)"
        print(f"  Track {track} ({TRACK_COLUMNS[track]}): {src} {mode}")
    print(f"{'=' * 70}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
