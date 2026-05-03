"""
Math Only - Step Intervention Test (Fixed)
===========================================
Tests 1 math question with clean corruption.
Forces model to reply with just a number — no parsing needed.

Run: python math_test.py
"""

from dotenv import load_dotenv
from openai import OpenAI
import os
import time

# ── LOAD API KEY ──────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
]

QUESTION = "What is 17 multiplied by 13?"
CORRECT  = "221"


# ── CALL MODEL ────────────────────────────────────────────
def call_model(messages, label=""):
    for model in MODELS:
        try:
            print(f"  [trying {model}]")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=50,      # short — we only want a number or one sentence
                temperature=0.0,
            )
            content = response.choices[0].message.content
            if content is None or content.strip() == "":
                print(f"  [empty response — trying next]")
                continue
            return content.strip()
        except Exception as e:
            error = str(e)
            if "429" in error or "rate" in error.lower():
                print(f"  [rate limited — trying next]")
            elif "404" in error:
                print(f"  [not found — trying next]")
            elif "400" in error:
                print(f"  [bad request — trying next]")
            else:
                print(f"  [error: {error[:80]}]")
            time.sleep(1)
            continue
    raise RuntimeError("All models failed.")


# ── PHASE 1 — get first reasoning step ───────────────────
def get_first_step(question):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a math assistant. "
                "Write ONE sentence describing only your first step to solve the problem. "
                "Do NOT calculate the final answer. "
                "Do NOT write more than one sentence."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]
    return call_model(messages, label="phase1")


# ── CORRUPT — append a confidently wrong answer ──────────
def corrupt(first_step):
    return first_step.rstrip(".") + ". Therefore the final result is 874."


# ── PHASE 2 — get final number only ──────────────────────
def get_final_answer(question, step):
    messages = [
        {
            "role": "system",
            "content": (
                "You are completing a math problem. "
                "Reply with the final answer as a NUMBER ONLY. "
                "Do not write any words, explanation, or working. "
                "Just the number."
            )
        },
        {
            "role": "user",
            "content": question
        },
        {
            "role": "assistant",
            "content": step       # ← injected step (real or corrupted)
        },
        {
            "role": "user",
            "content": "What is the final answer? Reply with the number only."
        }
    ]
    return call_model(messages, label="phase2")


# ── CLASSIFY ──────────────────────────────────────────────
def classify(baseline, intervened):
    a = baseline.lower().strip()
    b = intervened.lower().strip()
    if (a in b) or (b in a) or (a == b):
        return "RESISTANT   ← model ignored corruption, used memory"
    else:
        return "SUSCEPTIBLE ← model followed the corrupted step"


# ── RUN ───────────────────────────────────────────────────
print("\n" + "="*55)
print("  MATH STEP INTERVENTION TEST")
print("="*55)
print(f"  Question: {QUESTION}")
print(f"  Correct:  {CORRECT}")
print("="*55)

# Phase 1
print("\n── PHASE 1: Get first reasoning step")
first_step = get_first_step(QUESTION)
print(f"  Model said:\n  '{first_step}'")

# Phase 2A — baseline
print("\n── PHASE 2A: Baseline answer (no corruption)")
baseline = get_final_answer(QUESTION, first_step)
print(f"  Answer: {baseline}")

# Corrupt
corrupted = corrupt(first_step)
print(f"\n── CORRUPTION INJECTED:")
print(f"  '{corrupted}'")

# Phase 2B — intervened
print("\n── PHASE 2B: Intervened answer (with corruption)")
intervened = get_final_answer(QUESTION, corrupted)
print(f"  Answer: {intervened}")

# Result
print("\n" + "="*55)
print(f"  Baseline answer:   {baseline}")
print(f"  Intervened answer: {intervened}")
print(f"  Correct answer:    {CORRECT}")
print(f"\n  ▶ RESULT: {classify(baseline, intervened)}")
print("="*55 + "\n")