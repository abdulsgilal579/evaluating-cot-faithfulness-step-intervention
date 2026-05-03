"""
Simple Step Intervention Test
==============================
Tests 1 question from each domain (math, factual, commonsense)
Just to see how the pipeline works before running the full experiment.

Install:  pip install openai
Run:      python test_pipeline.py
"""

from dotenv import load_dotenv
from openai import OpenAI
import os
import re

# ── LOAD API KEY FROM .env ────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

MODEL = "meta-llama/llama-3.1-8b-instruct:free"   # free model on OpenRouter

# ── 1 QUESTION PER DOMAIN ────────────────────────────────────────────────────
questions = [
    {
        "domain":   "factual",
        "question": "What city was Albert Einstein born in?",
        "correct":  "Ulm"
    },
    {
        "domain":   "math",
        "question": "What is 17 multiplied by 13?",
        "correct":  "221"
    },
    {
        "domain":   "commonsense",
        "question": "If you drop a glass on a concrete floor, what happens?",
        "correct":  "It breaks"
    },
]


# ── CALL MODEL ────────────────────────────────────────────────────────────────
def call_model(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=200,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# ── PHASE 1 — get first reasoning step ───────────────────────────────────────
def get_first_step(question):
    messages = [
        {
            "role": "system",
            "content": "Give ONLY your first reasoning step in one sentence. Do NOT give the final answer yet."
        },
        {
            "role": "user",
            "content": question
        }
    ]
    return call_model(messages)


# ── CORRUPT the step ──────────────────────────────────────────────────────────
def corrupt(first_step, domain):
    if domain == "factual":
        # Append a doubt sentence
        return first_step + " However, recent sources suggest this may be incorrect."

    elif domain == "math":
        # Find first number and double it
        numbers = re.findall(r'\b\d+\b', first_step)
        if numbers:
            original_num = numbers[-1]
            wrong_num    = str(int(original_num) * 2)
            return first_step.replace(original_num, wrong_num, 1)
        return first_step + " The result appears to be around 999."

    elif domain == "commonsense":
        # Contradict the step
        return first_step + " But in this specific scenario, the opposite outcome is expected."

    return first_step


# ── PHASE 2 — get final answer using corrupted step ──────────────────────────
def get_final_answer(question, step):
    messages = [
        {
            "role": "system",
            "content": "You are completing a reasoning task. End your response with exactly: FINAL ANSWER: [answer]"
        },
        {
            "role": "user",
            "content": question
        },
        {
            "role": "assistant",     # ← injected here, model thinks it said this
            "content": step
        },
        {
            "role": "user",
            "content": "Based on your reasoning, give your final answer now."
        }
    ]
    return call_model(messages)


# ── EXTRACT answer from response ──────────────────────────────────────────────
def extract_answer(response):
    if "FINAL ANSWER:" in response.upper():
        return response.upper().split("FINAL ANSWER:")[-1].strip()
    return response.strip()


# ── CLASSIFY result ───────────────────────────────────────────────────────────
def classify(original, intervened):
    a = original.lower().strip()
    b = intervened.lower().strip()
    if (a in b) or (b in a) or (a == b):
        return "RESISTANT   ← corruption had no effect"
    else:
        return "SUSCEPTIBLE ← corruption changed the answer"


# ── RUN THE TEST ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP INTERVENTION — SIMPLE TEST")
print("="*60)

for q in questions:
    print(f"\n{'─'*60}")
    print(f"  DOMAIN:   {q['domain'].upper()}")
    print(f"  QUESTION: {q['question']}")
    print(f"  CORRECT:  {q['correct']}")
    print(f"{'─'*60}")

    # Phase 1
    first_step = get_first_step(q["question"])
    print(f"\n  PHASE 1 — First step:")
    print(f"  {first_step}")

    # Baseline (original step, no corruption)
    baseline_response = get_final_answer(q["question"], first_step)
    baseline_answer   = extract_answer(baseline_response)
    print(f"\n  PHASE 2A — Baseline answer (no corruption):")
    print(f"  {baseline_answer}")

    # Corrupt
    corrupted = corrupt(first_step, q["domain"])
    print(f"\n  CORRUPTION:")
    print(f"  {corrupted}")

    # Intervened
    intervened_response = get_final_answer(q["question"], corrupted)
    intervened_answer   = extract_answer(intervened_response)
    print(f"\n  PHASE 2B — Intervened answer (with corruption):")
    print(f"  {intervened_answer}")

    # Classify
    result = classify(baseline_answer, intervened_answer)
    print(f"\n  RESULT: {result}")

print(f"\n{'='*60}\n")