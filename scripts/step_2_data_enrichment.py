#!/usr/bin/env python3
"""
Combine multiple data sources and apply augmentation:
1. tagged_question_answer.jsonl (synthetic predictions from step 1)
2. phase_1_test.csv + ground truth labels
3. train.csv original training data
4. Apply shuffle + replace_label augmentation
5. Output training_dataset.jsonl (~20,000 samples)

Usage:
    python3 step_2_data_enrichment.py [--multiply N] [--output OUTPUT]
"""

import argparse
import csv
import json
import os
import random
import re
import string
from pathlib import Path


# System prompt matching exp008 for consistency
SYSTEM_PROMPT = "You are a 5G network expert. Analyze the data and identify the root cause."


# === Regex patterns from step1 ===
CHOICE_LINE_RE = re.compile(
    r"^(?P<prefix>\s*)(?P<label>(?:[A-Za-z]+\d*|\d+))(?P<pre_colon>\s*):(?P<post_colon>\s*)(?P<text>.*?)(?P<suffix>\s*)$"
)


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


def load_train_csv(path: Path) -> list:
    """Load train.csv and convert to JSONL format."""
    records = []
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            answer = row['answer']
            records.append({
                'id': row['ID'],
                'source': 'train',
                'input': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': row['question']}
                ],
                'output': [
                    {'role': 'assistant', 'content': f"\\boxed{{{answer}}}"}
                ]
            })
    return records


def load_phase1_with_truth(test_path: Path, truth_path: Path) -> list:
    """Load phase_1_test with truth and convert to JSONL format."""
    records = []
    
    # Build truth dict
    truth_dict = {}
    with open(truth_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            full_id = row['ID']
            if '_1' in full_id:
                base_id = '_'.join(full_id.split('_')[:-1])
                # Try multiple columns
                ans = row.get('Qwen3-32B', row.get('Qwen2.5-1.5B-Instruct', ''))
                if ans and base_id not in truth_dict:
                    truth_dict[base_id] = ans
    
    # Load test questions
    with open(test_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row['ID']
            if qid in truth_dict:
                answer = truth_dict[qid]
                records.append({
                    'id': qid,
                    'source': 'phase1',
                    'input': [
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': row['question']}
                    ],
                    'output': [
                        {'role': 'assistant', 'content': f"\\boxed{{{answer}}}"}
                    ]
                })
    return records


def _is_choice_label(label: str) -> bool:
    """Check if a label is a valid choice label."""
    if label.isdigit():
        return True
    if label.isalpha() and len(label) <= 2:
        return True
    if re.match(r'^[A-Za-z]+\d+$', label):
        return True
    return False


def _extract_label_prefix(label: str) -> str:
    """Extract prefix from a label like 'C1' -> 'C'."""
    return re.sub(r'\d+$', '', label)


def _shuffle_answer_list(question: str, answer_code: str) -> tuple:
    """Shuffle answer options and update answer label.
    
    Synced from step1_augment_raw_train_data.py
    Returns: (new_question, new_answer_code, raw_answer_text)
    """
    if not answer_code:
        return question, answer_code, ""
    
    lines = question.splitlines(keepends=True)
    parsed = []
    
    for idx, line in enumerate(lines):
        newline = "\n" if line.endswith("\n") else ""
        content = line[:-1] if newline else line
        match = CHOICE_LINE_RE.match(content)
        if not match:
            continue
        label = match.group("label")
        if not _is_choice_label(label):
            continue
        parsed.append({
            "index": idx,
            "prefix": match.group("prefix"),
            "label": label,
            "pre_colon": match.group("pre_colon"),
            "post_colon": match.group("post_colon"),
            "text": match.group("text"),
            "suffix": match.group("suffix"),
            "newline": newline,
        })
    
    if not parsed:
        return question, answer_code, ""
    
    # Group consecutive options into segments
    segments = []
    current = [parsed[0]]
    for item in parsed[1:]:
        if item["index"] == current[-1]["index"] + 1:
            current.append(item)
        else:
            segments.append(current)
            current = [item]
    segments.append(current)
    
    # Find the segment containing the answer
    digits = re.sub(r"\D", "", answer_code)
    
    def _matches(label: str) -> bool:
        if label == answer_code:
            return True
        if digits:
            return re.sub(r"\D", "", label) == digits
        return False
    
    target_segment = None
    answer_entry = None
    for segment in segments:
        for entry in segment:
            if _matches(entry["label"]):
                target_segment = segment
                answer_entry = entry
                break
        if target_segment:
            break
    
    if not target_segment or not answer_entry:
        return question, answer_code, ""
    
    # Extract answer text
    raw_answer_text = f'{answer_entry["text"]}'
    
    # Shuffle remaining options
    remaining = [entry for entry in target_segment if entry is not answer_entry]
    if remaining:
        random.shuffle(remaining)
    
    # Random deletion: remove 0-3 distractor options
    remove_count = random.randint(0, min(3, len(remaining)))
    if remove_count:
        remove_indices = set(random.sample(range(len(remaining)), remove_count))
        remaining = [entry for i, entry in enumerate(remaining) if i not in remove_indices]
    
    # Insert answer at random position
    insert_pos = random.randint(0, len(remaining))
    remaining.insert(insert_pos, answer_entry)
    
    # Check if we restored original order - if so, force a change
    original_order = [e["text"] for e in target_segment]
    new_order = [e["text"] for e in remaining]
    if new_order == original_order and len(remaining) > 1:
        # Force move first element to end
        remaining = remaining[1:] + [remaining[0]]
        insert_pos = len(remaining) - 1 if remaining[-1] is answer_entry else 0
    
    # Renumber labels
    alpha_mode = all(entry["label"].isalpha() and len(entry["label"]) == 1 for entry in remaining)
    if alpha_mode:
        letters = string.ascii_uppercase
        if all(entry["label"].islower() for entry in remaining):
            letters = string.ascii_lowercase
        new_answer_code = letters[insert_pos]
        for idx, entry in enumerate(remaining):
            entry["label"] = letters[idx]
    else:
        label_prefix = _extract_label_prefix(answer_entry["label"])
        new_answer_code = f"{label_prefix}{insert_pos + 1}"
        for idx, entry in enumerate(remaining, start=1):
            entry["label"] = f"{label_prefix}{idx}"
    
    # Rebuild question
    start_idx = target_segment[0]["index"]
    end_idx = target_segment[-1]["index"]
    block_end_no_newline = end_idx == len(lines) - 1 and not lines[end_idx].endswith("\n")
    
    new_lines = []
    for idx, entry in enumerate(remaining):
        newline = "\n"
        if idx == len(remaining) - 1 and block_end_no_newline:
            newline = ""
        new_line = (
            f'{entry["prefix"]}{entry["label"]}{entry["pre_colon"]}:{entry["post_colon"]}'
            f'{entry["text"]}{entry["suffix"]}{newline}'
        )
        new_lines.append(new_line)
    
    new_question = "".join(lines[:start_idx] + new_lines + lines[end_idx + 1:])
    return new_question, new_answer_code, raw_answer_text


def _build_label_map(question: str, answer_code: str) -> dict:
    """Build label replacement mapping.
    
    Synced from step1_augment_raw_train_data.py
    """
    labels = set()
    for line in question.splitlines():
        content = line.rstrip("\n")
        match = CHOICE_LINE_RE.match(content)
        if not match:
            continue
        label = match.group("label")
        if _is_choice_label(label):
            labels.add(label)
    
    mapping = {}
    if not labels:
        if answer_code:
            mapping[answer_code] = answer_code
        return mapping
    
    # Skip replacement for pure single-letter labels (A, B, C, D)
    if all(label.isalpha() and len(label) == 1 for label in labels):
        for label in labels:
            mapping[label] = label
        if answer_code and answer_code not in mapping:
            mapping[answer_code] = answer_code
        return mapping
    
    # Step1-style prefix choices
    choices = ["", " ", "A", "B", "C", "M", "I", "d", "n", "t"]
    prefix = random.choice(choices)
    
    for label in labels:
        digits = re.sub(r"\D", "", label)
        if not digits:
            mapping[label] = label
        else:
            mapping[label] = f"{prefix}{digits}"
    
    if answer_code and answer_code not in mapping:
        mapping[answer_code] = answer_code
    
    return mapping


def _replace_question_labels(question: str, mapping: dict) -> str:
    """Replace labels in question text based on mapping."""
    lines = question.splitlines(keepends=True)
    new_lines = []
    
    for line in lines:
        newline = "\n" if line.endswith("\n") else ""
        content = line[:-1] if newline else line
        match = CHOICE_LINE_RE.match(content)
        
        if match and match.group("label") in mapping:
            new_label = mapping[match.group("label")]
            new_line = (
                f'{match.group("prefix")}{new_label}{match.group("pre_colon")}:'
                f'{match.group("post_colon")}{match.group("text")}{match.group("suffix")}{newline}'
            )
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    
    return "".join(new_lines)


def augment_record(record: dict, seed: int, suffix: str) -> dict:
    """Apply shuffle + replace_label augmentation to a single record."""
    random.seed(seed)
    
    # Get user message (find by role, not index - handles system prompt)
    user_msg_idx = None
    for i, msg in enumerate(record['input']):
        if msg.get('role') == 'user':
            user_msg_idx = i
            break
    if user_msg_idx is None:
        return record
    question = record['input'][user_msg_idx]['content']
    
    # Extract answer from output
    output_content = record['output'][0]['content']
    match = re.search(r'\\boxed\{([^}]+)\}', output_content)
    if not match:
        return record
    answer = match.group(1)
    
    # Step 1: Shuffle answer list
    new_question, new_answer, answer_text = _shuffle_answer_list(question, answer)
    
    # Step 2: Replace labels
    mapping = _build_label_map(new_question, new_answer)
    new_question = _replace_question_labels(new_question, mapping)
    new_answer = mapping.get(new_answer, new_answer)
    
    # Get answer description from new question
    if not answer_text:
        pattern = rf'(?:^|\n)\s*{re.escape(new_answer)}\s*:\s*([^\n]+)'
        desc_match = re.search(pattern, new_question)
        if desc_match:
            answer_text = desc_match.group(1).strip()
            if answer_text.endswith('.'):
                answer_text = answer_text[:-1].rstrip()
    
    # Format output: "description, \boxed{label}"
    if answer_text:
        output_text = f"{answer_text}, \\boxed{{{new_answer}}}"
    else:
        output_text = f"\\boxed{{{new_answer}}}"
    
    # Create new record with system prompt
    new_record = {
        'id': f"{record['id']}_{suffix}",
        'source': record.get('source', 'unknown'),
        'input': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': new_question}
        ],
        'output': [
            {'role': 'assistant', 'content': output_text}
        ]
    }
    
    return new_record


def main():
    parser = argparse.ArgumentParser(description='Training Data Enrichment')
    parser.add_argument('--tagged_jsonl', default='../data/tagged_question_answer.jsonl')
    parser.add_argument('--category_c_answers', default='../data/category_C_answers.csv')
    parser.add_argument('--category_c_questions', default='../data/category_C_non_5g.csv')
    parser.add_argument('--phase1_test', default='../challenge_data/phase_1_test.csv')
    parser.add_argument('--phase1_truth', default='../challenge_data/phase_1_test_truth.csv')
    parser.add_argument('--train_csv', default='../challenge_data/train.csv')
    parser.add_argument('--output', default='../data/training_dataset.jsonl')
    parser.add_argument('--multiply', type=int, default=5, help='Number of augmented copies')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true', help='Only show stats, do not save')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    random.seed(args.seed)
    
    print("=" * 70)
    print("Step 2: Training Data Enrichment (with shuffle + replace_label)")
    print("=" * 70)
    
    # 1. Load Tagged data (Category A + B from step2)
    print(f"\n[1] Loading data sources...")
    tagged_path = script_dir / args.tagged_jsonl
    tagged_records = load_jsonl(tagged_path)
    for r in tagged_records:
        r['source'] = 'tagged'
    print(f"    Tagged (A+B): {len(tagged_records)} samples")
    
    # 2. Load Category C (non-5G) questions and answers
    category_c_records = []
    
    # Load answers
    c_answers = {}
    c_answers_path = script_dir / args.category_c_answers
    if c_answers_path.exists():
        with open(c_answers_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['answer']:
                    c_answers[row['ID']] = row['answer']
    
    # Load questions and combine
    c_questions_path = script_dir / args.category_c_questions
    if c_questions_path.exists():
        with open(c_questions_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                qid = row['ID']
                if qid in c_answers:
                    category_c_records.append({
                        'id': qid,
                        'source': 'category_c',
                        'input': [
                            {'role': 'system', 'content': SYSTEM_PROMPT},
                            {'role': 'user', 'content': row['question']}
                        ],
                        'output': [
                            {'role': 'assistant', 'content': f"\\boxed{{{c_answers[qid]}}}"}
                        ]
                    })
    print(f"    Category C: {len(category_c_records)} samples")
    
    # 3. Load Phase 1 with truth
    phase1_records = load_phase1_with_truth(
        script_dir / args.phase1_test,
        script_dir / args.phase1_truth
    )
    print(f"    Phase 1: {len(phase1_records)} samples")
    
    # 4. Load train.csv
    train_records = load_train_csv(script_dir / args.train_csv)
    print(f"    train.csv: {len(train_records)} samples")
    
    # 5. Combine all
    all_records = tagged_records + category_c_records + phase1_records + train_records
    print(f"\n[2] Combined: {len(all_records)} samples")
    
    # 5. Apply shuffle + replace_label x multiply
    print(f"\n[3] Generating augmented data {args.multiply}x copies (shuffle + replace_label)...")
    # Time/Resource limit, I think use LLM model to generate synthetic data is good too.
    
    augmented = []
    for i in range(args.multiply):
        shuffled = all_records.copy()
        random.shuffle(shuffled)
        for j, rec in enumerate(shuffled):
            aug_rec = augment_record(rec, seed=args.seed + i * 100000 + j, suffix=f"s{i+1}")
            augmented.append(aug_rec)
        print(f"    Copy {i+1}: {len(shuffled)} samples")
    
    # 6. Final random shuffle
    print("\n[4] Final random shuffle...")
    random.shuffle(augmented)
    
    # 7. Save
    if args.dry_run:
        print("\n[DRY RUN] Would save to:", script_dir / args.output)
        print("Sample entry:")
        print(json.dumps(augmented[0], indent=2, ensure_ascii=False)[:500])
    else:
        output_path = script_dir / args.output
        with open(output_path, 'w', encoding='utf-8') as f:
            for rec in augmented:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"\n[5] Saved to: {output_path}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    sources = {}
    for r in augmented:
        src = r.get('source', 'unknown')
        sources[src] = sources.get(src, 0) + 1
    print(f"Total samples: {len(augmented)}")

    
    # 8. Data leakage check (EXACT content match)
    print(f"\n{'=' * 70}")
    print("DATA LEAKAGE CHECK (question match)")
    print(f"{'=' * 70}")
    
    import hashlib
    
    # Load phase_2_test questions (full content hash)
    phase2_test_path = script_dir / '../challenge_data/phase_2_test.csv'
    phase2_questions = {}
    if phase2_test_path.exists():
        with open(phase2_test_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_hash = hashlib.md5(row['question'].encode()).hexdigest()
                phase2_questions[q_hash] = row['ID']
    
    # Load phase_1_test questions
    phase1_test_path = script_dir / args.phase1_test
    phase1_questions = {}
    if phase1_test_path.exists():
        with open(phase1_test_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_hash = hashlib.md5(row['question'].encode()).hexdigest()
                phase1_questions[q_hash] = row['ID']
    
    # Check training data for EXACT matches
    phase2_exact = 0
    phase1_exact = 0
    for rec in augmented:
        # Find user message by role (not index)
        question = None
        for msg in rec.get('input', []):
            if msg.get('role') == 'user':
                question = msg.get('content', '')
                break
        if not question:
            continue
        q_hash = hashlib.md5(question.encode()).hexdigest()
        if q_hash in phase2_questions:
            phase2_exact += 1
        if q_hash in phase1_questions:
            phase1_exact += 1

    print("Checking Data leak...")
    
    if phase2_exact > 0:
        print(f"\n[WARNING] Phase 2 EXACT match: {phase2_exact} found!")
    else:
        print(f"\n✅ [OK] Phase 2 questions checked: Not Matched")
    
    if phase1_exact > 0:
        print(f"[INFO] Phase 1 EXACT match: {phase1_exact} (may be expected)")
    else:
        print(f"✅ [OK] Phase 1 questions checked: Not Matched")
    
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
