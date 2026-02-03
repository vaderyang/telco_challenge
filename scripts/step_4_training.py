#!/usr/bin/env python3
"""
Step 4: LoRA Training

Train a LoRA adapter using Unsloth on specified track.
Supports track1 (32B), track2 (7B), track3 (1.5B).

Usage:
    python3 step_4_training.py --track 3 --epochs 3
    python3 step_4_training.py --track 2 --epochs 5 --auto_eval
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Set offline mode before imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


# Track configurations with optimized hyperparameters
TRACK_CONFIG = {
    1: {
        'name': 'track1_32b',
        'model_path': '/root/Qwen3-32B',
        'description': 'Qwen3-32B (large)',
        'learning_rate': 5e-5,   # Conservative for large model
        'lora_r': 128,
        'epochs': 2,             # Fewer epochs for large model
    },
    2: {
        'name': 'track2_7b',
        'model_path': '/root/Qwen2.5-7B-Instruct',
        'description': 'Qwen2.5-7B-Instruct (medium)',
        'learning_rate': 1e-4,   # Standard
        'lora_r': 128,
        'epochs': 4,             # Standard
    },
    3: {
        'name': 'track3_1.5b',
        'model_path': '/root/Qwen2.5-1.5B-Instruct',
        'description': 'Qwen2.5-1.5B-Instruct (small)',
        'learning_rate': 1e-4,   # Standard
        'lora_r': 64,            # Lower rank for smaller model
        'epochs': 25,            # More epochs for small model
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='LoRA Training')
    parser.add_argument('--track', type=int, required=True, choices=[1, 2, 3],
                        help='Track to train: 1=32B, 2=7B, 3=1.5B')
    parser.add_argument('--data_path', default='data/training_dataset.jsonl',
                        help='Training data path')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Training split ratio (default: 0.9)')
    parser.add_argument('--epochs', type=float, default=3.0,
                        help='Number of training epochs')
    parser.add_argument('--max_seq_len', type=int, default=12000,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Per-device batch size')
    parser.add_argument('--gradient_accum', type=int, default=5,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lora_rank', type=int, default=128,
                        help='LoRA rank')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--auto_eval', action='store_true',
                        help='Run evaluation after training')
    parser.add_argument('--base_model_override', type=str, default=None,
                        help='Override base model path')
    return parser.parse_args()


def build_sft_features(example, tokenizer, max_length=None):
    """Build SFT features with proper masking."""
    if "input" in example and "output" in example:
        input_msgs = example["input"]
        output_msgs = example["output"]

        prompt_text = tokenizer.apply_chat_template(
            input_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_msgs = input_msgs + output_msgs
        full_text = tokenizer.apply_chat_template(
            full_msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
    elif "prompt" in example and "full" in example:
        prompt_text = example["prompt"]
        full_text = example["full"]
    else:
        raise ValueError("input/output or prompt/full must be provided")

    prompt_tok = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    full_tok = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    prompt_len = len(prompt_tok["input_ids"])

    labels = full_tok["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len

    if len(labels) < prompt_len:
        labels = [-100] * len(labels)

    full_tok["labels"] = labels
    return full_tok


def main():
    args = parse_args()
    
    # Setup paths
    exp_dir = Path(__file__).parent.parent
    data_path = exp_dir / args.data_path
    
    track_config = TRACK_CONFIG[args.track]
    track_name = track_config['name']
    model_path = args.base_model_override or track_config['model_path']
    
    # Add timestamp to adapter directory
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    adapter_dir = exp_dir / 'adapters' / f'{track_name}_{timestamp}'
    progress_dir = exp_dir / 'adapters' / f'{track_name}_progress'
    
    # Use track-specific hyperparams (CLI args can override)
    effective_lr = track_config.get('learning_rate', args.learning_rate)
    effective_lora_r = track_config.get('lora_r', args.lora_rank)
    effective_epochs = track_config.get('epochs', args.epochs)
    # Allow CLI override if explicitly set (different from default)
    if args.learning_rate != 1e-4:  # default value changed
        effective_lr = args.learning_rate
    if args.lora_rank != 128:  # default value changed
        effective_lora_r = args.lora_rank
    if args.epochs != 3.0:  # default value changed
        effective_epochs = args.epochs
    
    print("=" * 70)
    print(f"Step 4.1: Training {track_config['description']}")
    print("=" * 70)
    print(f"Track: {args.track} ({track_name})")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Epochs: {effective_epochs}")
    print(f"Learning Rate: {effective_lr}")
    print(f"LoRA Rank: {effective_lora_r}")
    print(f"Train/Eval split: {args.train_split}/{1-args.train_split:.2f}")
    print(f"Adapter output: {adapter_dir}")
    print("=" * 70)
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load and split dataset
    print(f"\n[1] Loading dataset from {data_path}...")
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    print(f"    Total samples: {len(dataset)}")
    
    # Split into train/eval
    split_dataset = dataset.train_test_split(
        train_size=args.train_split,
        seed=args.seed
    )
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    print(f"    Train samples: {len(train_dataset)}")
    print(f"    Eval samples: {len(eval_dataset)}")
    
    # Save eval IDs for later
    eval_ids = [ex['id'] for ex in eval_dataset]
    eval_ids_path = exp_dir / 'data' / f'eval_ids_{track_name}.json'
    with open(eval_ids_path, 'w') as f:
        json.dump(eval_ids, f)
    print(f"    Eval IDs saved to: {eval_ids_path}")
    
    # Load model
    print(f"\n[2] Loading model from {model_path}...")
    DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=args.max_seq_len,
        dtype=DTYPE,
        load_in_4bit=True,
        local_files_only=True,
    )
    print(f"    Model loaded with dtype={DTYPE}")
    
    # Apply LoRA
    print(f"\n[3] Applying LoRA (rank={effective_lora_r})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=effective_lora_r,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Tokenize dataset
    print("\n[4] Tokenizing dataset...")
    tokenized_train = train_dataset.map(
        build_sft_features,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_seq_len},
        remove_columns=train_dataset.column_names,
    )
    
    # Training arguments
    bf16 = (DTYPE == torch.bfloat16)
    fp16 = (DTYPE == torch.float16)
    
    training_args = TrainingArguments(
        output_dir=str(progress_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accum,
        num_train_epochs=effective_epochs,
        learning_rate=effective_lr,
        logging_steps=5,
        save_steps=8000,
        save_total_limit=10,
        bf16=bf16,
        fp16=fp16,
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        seed=args.seed,
        report_to="none",
    )
    
    # Train
    print("\n[5] Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        args=training_args,
    )
    
    trainer.train()
    
    # Save adapter
    print(f"\n[6] Saving adapter to {adapter_dir}...")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    
    # Save training info
    info = {
        'track': args.track,
        'track_name': track_name,
        'base_model': model_path,
        'epochs': args.epochs,
        'train_samples': len(train_dataset),
        'eval_samples': len(eval_dataset),
        'train_split': args.train_split,
        'lora_rank': args.lora_rank,
        'max_seq_len': args.max_seq_len,
    }
    with open(adapter_dir / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Training complete!")
    print(f"Adapter saved to: {adapter_dir}")
    print(f"{'=' * 70}")
    
    # Auto-eval using vLLM (skip for Track 1/32B due to memory)
    if args.auto_eval and args.track != 1:
        print("\n[7] Running auto-evaluation with vLLM...")
        import subprocess
        eval_script = exp_dir / 'scripts' / 'step_5_eval_vllm_and_save.sh'
        subprocess.run(['bash', str(eval_script), adapter_dir.name], cwd=str(exp_dir))
    elif args.auto_eval and args.track == 1:
        print("\n[7] Skipping auto-eval for Track 1 (32B) - run manually after training")


if __name__ == '__main__':
    main()
