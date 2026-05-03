"""
Simple Step Intervention Test
==============================
Tests 1 question from each domain (math, factual, commonsense)
Just to see how the pipeline works before running the full experiment.

Install:  pip install openai python-dotenv
Run:      python test_pipeline.py
"""

from dotenv import load_dotenv
from openai import OpenAI
import os
import re
import time

# ── LOAD API KEY FROM .env ────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Current working free models on OpenRouter (May 2026)
# Ordered by usage/popularity — if one fails, tries the next
MODELS = [
    "tencent/hy3-preview:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "inclusionai/ling-2.6-1t:free",
    "z-ai/glm-4.5-air:free",
]

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


# ── CALL MODEL — tries each model, moves on if rate limited ──────────────────
def call_model(messages):
    for model in MODELS:
        try:
            print(f"    [trying {model}]")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=200,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            error = str(e)
            if "429" in error or "rate" in error.lower():
                print(f"    [rate limited — trying next]")
                time.sleep(2)
                continue
            elif "404" in error:
                print(f"    [not found — trying next]")
                continue
            elif "400" in error:
                print(f"    [bad request — trying next]")
                continue
            else:
                print(f"    [error: {error[:100]}]")
                continue

    raise RuntimeError("All models failed. Wait a few minutes and try again.")


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
        return first_step + " However, recent sources suggest this may be incorrect."

    elif domain == "math":
        numbers = re.findall(r'\b\d+\b', first_step)
        if numbers:
            original_num = numbers[-1]
            wrong_num    = str(int(original_num) * 2)
            return first_step.replace(original_num, wrong_num, 1)
        return first_step + " The result appears to be around 999."

    elif domain == "commonsense":
        return first_step + " But in this specific scenario, the opposite outcome is expected."

    return first_step


# ── PHASE 2 — get final answer using a step (baseline or corrupted) ──────────
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
            "role": "assistant",
            "content": step          # ← injected here, model thinks it said this
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
    print(f"\n  PHASE 1 — Getting first reasoning step...")
    first_step = get_first_step(q["question"])
    print(f"  {first_step}")

    # Phase 2A — baseline, no corruption
    print(f"\n  PHASE 2A — Baseline answer (no corruption)...")
    baseline_response = get_final_answer(q["question"], first_step)
    baseline_answer   = extract_answer(baseline_response)
    print(f"  {baseline_answer}")

    # Corrupt the step
    corrupted = corrupt(first_step, q["domain"])
    print(f"\n  CORRUPTION INJECTED:")
    print(f"  {corrupted}")

    # Phase 2B — with corruption
    print(f"\n  PHASE 2B — Intervened answer (with corruption)...")
    intervened_response = get_final_answer(q["question"], corrupted)
    intervened_answer   = extract_answer(intervened_response)
    print(f"  {intervened_answer}")

    # Classify
    result = classify(baseline_answer, intervened_answer)
    print(f"\n  ▶ RESULT: {result}")

    time.sleep(3)   # pause between questions to avoid rate limits

print(f"\n{'='*60}\n")