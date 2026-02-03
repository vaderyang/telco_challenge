#!/usr/bin/env python3
"""
Category C (Non-5G) - Synthetic Data Generator

Generate similar Q&A pairs for non-5G questions using LLM.
Instead of directly answering, generates synthetic training data.

Usage:
    python3 step_2_3_category_C_inference.py [--input INPUT] [--output OUTPUT]
"""

import subprocess
import sys

def _ensure_deps(packages):
    for pkg in packages:
        pkg_name = pkg.split('>=')[0].split('==')[0].strip()
        try:
            __import__(pkg_name)
        except ImportError:
            print(f"[auto-install] Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

_ensure_deps(['aiohttp', 'requests'])

import argparse
import asyncio
import csv
import json
import os
import re
from pathlib import Path

import aiohttp
import requests


def check_vllm_setup():
    """Check vLLM connection and prompt user if not available."""
    port = os.environ.get('VLLM_PORT', '8001')
    host = os.environ.get('VLLM_HOST', 'http://localhost')
    model = os.environ.get('MODEL_NAME', 'Qwen3-32B')
    
    api_base = f"{host}:{port}"
    
    print("=" * 70)
    print("Checking vLLM Server...")
    print("=" * 70)
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Model: {model}")
    print()
    
    try:
        resp = requests.get(f"{api_base}/v1/models", timeout=5)
        if resp.status_code == 200:
            print("✅ vLLM server is running!")
            models = resp.json().get('data', [])
            if models:
                print(f"   Available models: {', '.join(m['id'] for m in models)}")
            return api_base, model
        else:
            raise Exception(f"Status {resp.status_code}")
    except Exception as e:
        print(f"❌ vLLM not available: {e}")
        print()
        print("=" * 70)
        print("SETUP INSTRUCTIONS")
        print("=" * 70)
        print("""
To run this script, you need a vLLM server with Qwen3-32B:

Option 1: Docker (recommended)
--------------------------------
docker run -d --gpus all --name vllm_32b \\
    -v /root/Qwen3-32B:/model \\
    -p 8001:8000 vllm/vllm-openai:latest \\
    --model /model --max-model-len 16000 --gpu-memory-utilization 0.9

Option 2: Environment Variables
--------------------------------
export VLLM_HOST=http://your-server-ip
export VLLM_PORT=8001
export MODEL_NAME=Qwen3-32B

Then run this script again.
""")
        print("=" * 70)
        return None, None


SYNTHETIC_PROMPT_TEMPLATE = """You are a question generator. Given the following question, generate a SIMILAR but DIFFERENT question on the same topic, along with the correct answer.

Original Question:
{question}

Generate output in this exact format:
QUESTION: [Your new similar question with options]
ANSWER: [The correct option number]
EXPLANATION: [Brief explanation]

Requirements:
1. The new question should test similar knowledge but with different specifics
2. Keep the same number of options as the original
3. Make sure the answer is clearly correct
4. Output must be in the exact format above"""


async def generate_synthetic_pair(session: aiohttp.ClientSession, question: dict, 
                                   api_base: str, model: str, semaphore: asyncio.Semaphore) -> dict:
    """Generate a synthetic Q&A pair from an original question."""
    async with semaphore:
        prompt = SYNTHETIC_PROMPT_TEMPLATE.format(question=question['question'])
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.7,  # Some creativity for variety
        }
        
        try:
            async with session.post(
                f"{api_base}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response = data["choices"][0]["message"]["content"]
                    return parse_synthetic_response(question['ID'], response)
                else:
                    return {'ID': question['ID'], 'error': f"API Error {resp.status}"}
        except Exception as e:
            return {'ID': question['ID'], 'error': str(e)}


def parse_synthetic_response(qid: str, response: str) -> dict:
    """Parse the LLM response to extract synthetic Q&A."""
    result = {
        'ID': f"{qid}_synth",
        'source': 'synthetic',
    }
    
    # Extract QUESTION
    q_match = re.search(r'QUESTION:\s*(.+?)(?=ANSWER:|$)', response, re.DOTALL)
    if q_match:
        result['question'] = q_match.group(1).strip()
    else:
        result['question'] = ''
    
    # Extract ANSWER
    a_match = re.search(r'ANSWER:\s*(\d+)', response)
    if a_match:
        result['answer'] = a_match.group(1)
    else:
        # Try other patterns
        a_match2 = re.search(r'correct.*?(\d+)', response, re.IGNORECASE)
        result['answer'] = a_match2.group(1) if a_match2 else ''
    
    return result


async def process_questions_async(questions: list, api_base: str, model: str, concurrency: int = 4):
    """Process all questions and generate synthetic pairs."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            generate_synthetic_pair(session, q, api_base, model, semaphore)
            for q in questions
        ]
        
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"    Progress: {i+1}/{len(questions)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Category C Synthetic Data Generator')
    parser.add_argument('--input', default='../data/category_C_non_5g.csv',
                        help='Input category C CSV')
    parser.add_argument('--output', default='../data/category_C_synthetic.jsonl',
                        help='Output JSONL with synthetic Q&A pairs')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of concurrent requests')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    print("=" * 70)
    print("Category C: Synthetic Data Generator")
    print("=" * 70)
    
    # Check vLLM setup
    api_base, model = check_vllm_setup()
    if not api_base:
        print("\nExiting. Please setup vLLM first.")
        return 1
    
    # Load input
    input_path = script_dir / args.input
    print(f"\n[1] Loading questions from {input_path}...")
    
    if not input_path.exists():
        print(f"    File not found. Run step_1 first to generate category files.")
        return 1
    
    with open(input_path, 'r', encoding='utf-8') as f:
        questions = list(csv.DictReader(f))
    print(f"    Loaded {len(questions)} Category C questions")
    
    # Generate synthetic data
    print(f"\n[2] Generating synthetic Q&A pairs...")
    results = asyncio.run(process_questions_async(questions, api_base, model, args.workers))
    
    # Filter successful results
    valid_results = [r for r in results if r.get('question') and r.get('answer')]
    print(f"    Generated {len(valid_results)}/{len(questions)} synthetic pairs")
    
    # Save as JSONL for training
    output_path = script_dir / args.output
    print(f"\n[3] Saving to {output_path}...")
    
    system_prompt = "You are an expert. Analyze the question and select the correct answer."
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in valid_results:
            record = {
                'id': r['ID'],
                'source': 'synthetic_category_c',
                'input': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': r['question']}
                ],
                'output': [
                    {'role': 'assistant', 'content': f"\\boxed{{{r['answer']}}}"}
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"    Saved {len(valid_results)} synthetic training samples")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Original questions: {len(questions)}")
    print(f"Synthetic pairs generated: {len(valid_results)}")
    print(f"Output: {output_path}")
    print(f"{'=' * 70}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
