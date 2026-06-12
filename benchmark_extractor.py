"""
job_tracker — Semantic Extractor Benchmark
===========================================

HOW TO RUN
----------
Place this file anywhere inside your project, e.g.:
    jobtracker-BE/evaluation/benchmark_extractor.py

Then run from the jobtracker-BE directory (with its venv active):
    python evaluation/benchmark_extractor.py

By default it tests whichever model is in OLLAMA_MODEL below.
To test a different model, pass it as an argument:
    python evaluation/benchmark_extractor.py qwen2.5:7b-instruct

WHAT IT DOES
------------
Sends each test phrase to your Ollama extractor and checks whether
the result matches the expected intent/fields.

It prints a full results table and a score summary at the end, and
writes a timestamped JSON results file so you can compare runs.

REQUIREMENTS
------------
- Ollama must be running locally on port 11434
- The model under test must already be pulled (ollama pull <model>)
- The benchmark calls Ollama directly using the same JSON-extraction
  prompt structure your backend uses — edit SYSTEM_PROMPT below to
  match your actual extractor prompt exactly.
"""

import json
import sys
import time
import datetime
import httpx
from dataclasses import dataclass, asdict, field
from typing import Optional

# ──────────────────────────────────────────────
# CONFIG — edit these to match your actual setup
# ──────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"          # default; override via CLI arg

# Paste your actual extractor system prompt here.
# This must match what jobtracker-BE sends to Ollama exactly.
# If you use a different prompt, results won't reflect real performance.
SYSTEM_PROMPT = """
You are a job application tracker assistant. Extract the user's intent from their message and return ONLY a JSON object.

Return this exact schema:
{
  "intent": <string>,
  "company": <string or null>,
  "role": <string or null>,
  "status": <string or null>,
  "priority": <string or null>,
  "location_mode": <string or null>,
  "note_text": <string or null>,
  "needs_clarification": <boolean>,
  "clarification_prompt": <string or null>
}

Valid intents:
  create_application, update_status, update_priority, update_location,
  add_note, archive_application, restore_application, save_draft,
  discard_draft, list_applications, query_informational, unsupported

Valid status values:    bookmarked, applied, interviewing, offered, rejected, withdrawn
Valid priority values:  low, medium, high
Valid location values:  remote, hybrid, onsite

Rules:
- If the user is clearly stating an intent but key info is missing, set needs_clarification=true and write a clarification_prompt.
- If the message is a question or informational request, use intent=query_informational and do NOT mutate.
- If you cannot determine intent, use intent=unsupported.
- Return ONLY the JSON object. No prose, no markdown fences.
""".strip()

# ──────────────────────────────────────────────
# TEST CASES
# ──────────────────────────────────────────────
#
# Each case has:
#   phrase          — what the user typed/said
#   expected_intent — the intent the extractor must return
#   expected_fields — optional dict of fields that must match (None = "don't care")
#   must_not_mutate — if True, the intent must be query_informational or unsupported
#   must_clarify    — if True, needs_clarification must be True
#   category        — used for grouping in the report
#
# HOW SCORING WORKS
#   PASS  — intent matches AND all expected_fields match
#   FAIL  — intent wrong OR a required field is wrong
#   SAFE  — for must_not_mutate cases: correctly returned non-mutating intent
#   UNSAFE— for must_not_mutate cases: returned a mutating intent (bad)

TEST_CASES = [

    # ── DIRECT COMMANDS (these should already work) ────────────────────────
    {
        "category": "direct",
        "phrase": "create application for data scientist role at Spotify",
        "expected_intent": "create_application",
        "expected_fields": {"company": "Spotify", "role": "data scientist"},
    },
    {
        "category": "direct",
        "phrase": "add Netflix software engineer",
        "expected_intent": "create_application",
        "expected_fields": {"company": "Netflix"},
    },
    {
        "category": "direct",
        "phrase": "mark Spotify as rejected",
        "expected_intent": "update_status",
        "expected_fields": {"company": "Spotify", "status": "rejected"},
    },
    {
        "category": "direct",
        "phrase": "set Google priority to high",
        "expected_intent": "update_priority",
        "expected_fields": {"company": "Google", "priority": "high"},
    },
    {
        "category": "direct",
        "phrase": "add note to Spotify: recruiter reached out",
        "expected_intent": "add_note",
        "expected_fields": {"company": "Spotify"},
    },
    {
        "category": "direct",
        "phrase": "archive the Netflix application",
        "expected_intent": "archive_application",
        "expected_fields": {"company": "Netflix"},
    },
    {
        "category": "direct",
        "phrase": "restore the Google application",
        "expected_intent": "restore_application",
        "expected_fields": {"company": "Google"},
    },
    {
        "category": "direct",
        "phrase": "save draft",
        "expected_intent": "save_draft",
        "expected_fields": {},
    },
    {
        "category": "direct",
        "phrase": "discard draft",
        "expected_intent": "discard_draft",
        "expected_fields": {},
    },

    # ── INDIRECT COMMANDS (the gap we're closing) ──────────────────────────
    {
        "category": "indirect",
        "phrase": "i should track at Spotify for data scientist role",
        "expected_intent": "create_application",
        "expected_fields": {"company": "Spotify"},
    },
    {
        "category": "indirect",
        "phrase": "been meaning to add the Netflix SWE role",
        "expected_intent": "create_application",
        "expected_fields": {"company": "Netflix"},
    },
    {
        "category": "indirect",
        "phrase": "i think i applied to Google last week",
        "expected_intent": "update_status",
        "expected_fields": {"company": "Google", "status": "applied"},
    },
    {
        "category": "indirect",
        "phrase": "the Spotify one should be higher priority",
        "expected_intent": "update_priority",
        "expected_fields": {"company": "Spotify", "priority": "high"},
    },
    {
        "category": "indirect",
        "phrase": "bump Spotify to high",
        "expected_intent": "update_priority",
        "expected_fields": {"company": "Spotify", "priority": "high"},
    },
    {
        "category": "indirect",
        "phrase": "Netflix just got back to me, moving to interviewing",
        "expected_intent": "update_status",
        "expected_fields": {"company": "Netflix", "status": "interviewing"},
    },
    {
        "category": "indirect",
        "phrase": "the Google role is remote by the way",
        "expected_intent": "update_location",
        "expected_fields": {"company": "Google", "location_mode": "remote"},
    },
    {
        "category": "indirect",
        "phrase": "Spotify turned me down",
        "expected_intent": "update_status",
        "expected_fields": {"company": "Spotify", "status": "rejected"},
    },
    {
        "category": "indirect",
        "phrase": "got an offer from Netflix!",
        "expected_intent": "update_status",
        "expected_fields": {"company": "Netflix", "status": "offered"},
    },
    {
        "category": "indirect",
        "phrase": "jot down that the Spotify recruiter said they'll follow up Friday",
        "expected_intent": "add_note",
        "expected_fields": {"company": "Spotify"},
    },
    {
        "category": "indirect",
        "phrase": "i don't think i'm pursuing Netflix anymore",
        "expected_intent": "archive_application",
        "expected_fields": {"company": "Netflix"},
    },

    # ── INCOMPLETE COMMANDS (should ask for clarification) ─────────────────
    {
        "category": "incomplete",
        "phrase": "add a new application",
        "expected_intent": "create_application",
        "expected_fields": {},
        "must_clarify": True,
    },
    {
        "category": "incomplete",
        "phrase": "mark it as applied",
        "expected_intent": "update_status",
        "expected_fields": {"status": "applied"},
        "must_clarify": True,
    },
    {
        "category": "incomplete",
        "phrase": "Spotify",
        "expected_intent": "create_application",
        "expected_fields": {},
        "must_clarify": True,
    },
    {
        "category": "incomplete",
        "phrase": "set priority high",
        "expected_intent": "update_priority",
        "expected_fields": {"priority": "high"},
        "must_clarify": True,
    },

    # ── INFORMATIONAL QUESTIONS (must NOT mutate state) ────────────────────
    {
        "category": "informational",
        "phrase": "how many applications do I have?",
        "expected_intent": "query_informational",
        "expected_fields": {},
        "must_not_mutate": True,
    },
    {
        "category": "informational",
        "phrase": "what's the status of my Spotify application?",
        "expected_intent": "query_informational",
        "expected_fields": {},
        "must_not_mutate": True,
    },
    {
        "category": "informational",
        "phrase": "show me all my applications",
        "expected_intent": "list_applications",
        "expected_fields": {},
        "must_not_mutate": True,
    },
    {
        "category": "informational",
        "phrase": "which ones are high priority?",
        "expected_intent": "query_informational",
        "expected_fields": {},
        "must_not_mutate": True,
    },
    {
        "category": "informational",
        "phrase": "have I heard back from Netflix?",
        "expected_intent": "query_informational",
        "expected_fields": {},
        "must_not_mutate": True,
    },

    # ── AMBIGUOUS / MIXED INTENT (safe failure expected) ───────────────────
    {
        "category": "ambiguous",
        "phrase": "do something with Spotify",
        "expected_intent": "unsupported",
        "expected_fields": {},
    },
    {
        "category": "ambiguous",
        "phrase": "update Spotify and also add a note that the interview went well",
        "expected_intent": "add_note",   # mixed intent — note part is clear
        "expected_fields": {"company": "Spotify"},
    },
    {
        "category": "ambiguous",
        "phrase": "hm",
        "expected_intent": "unsupported",
        "expected_fields": {},
    },
]

# ──────────────────────────────────────────────
# MUTATING INTENTS — for must_not_mutate check
# ──────────────────────────────────────────────

MUTATING_INTENTS = {
    "create_application",
    "update_status",
    "update_priority",
    "update_location",
    "add_note",
    "archive_application",
    "restore_application",
    "save_draft",
    "discard_draft",
}

# ──────────────────────────────────────────────
# OLLAMA CALL
# ──────────────────────────────────────────────

def call_ollama(phrase: str, model: str) -> tuple[dict | None, float]:
    """Call Ollama and return (parsed_json_or_None, latency_ms)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": phrase},
        ],
        "stream": False,
    }
    start = time.perf_counter()
    try:
        r = httpx.post(OLLAMA_URL, json=payload, timeout=30.0)
        latency_ms = (time.perf_counter() - start) * 1000
        r.raise_for_status()
        raw = r.json()["message"]["content"].strip()
        # Strip markdown fences if model wraps anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw), latency_ms
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return None, latency_ms


# ──────────────────────────────────────────────
# EVALUATION LOGIC
# ──────────────────────────────────────────────

def normalize(value: str | None) -> str | None:
    """Lowercase + strip for loose field comparison."""
    return value.strip().lower() if isinstance(value, str) else value


def evaluate_case(case: dict, result: dict | None) -> dict:
    """
    Returns a result dict with:
      outcome: PASS | FAIL | SAFE | UNSAFE | ERROR
      notes:   list of strings explaining what went wrong
    """
    if result is None:
        return {"outcome": "ERROR", "notes": ["Ollama call failed or returned unparseable JSON"]}

    notes = []
    actual_intent   = result.get("intent", "")
    expected_intent = case["expected_intent"]
    must_not_mutate = case.get("must_not_mutate", False)
    must_clarify    = case.get("must_clarify", False)

    # Must-not-mutate check takes priority
    if must_not_mutate:
        if actual_intent in MUTATING_INTENTS:
            return {
                "outcome": "UNSAFE",
                "notes": [f"Returned mutating intent '{actual_intent}' for informational query"],
            }
        return {"outcome": "SAFE", "notes": []}

    # Intent match
    if actual_intent != expected_intent:
        notes.append(f"Intent: expected '{expected_intent}', got '{actual_intent}'")

    # Field matches
    for field_name, expected_val in case.get("expected_fields", {}).items():
        if expected_val is None:
            continue
        actual_val = result.get(field_name)
        if normalize(actual_val) != normalize(expected_val):
            notes.append(
                f"Field '{field_name}': expected '{expected_val}', got '{actual_val}'"
            )

    # Clarification check
    if must_clarify and not result.get("needs_clarification", False):
        notes.append("Expected needs_clarification=true but got false")

    outcome = "PASS" if not notes else "FAIL"
    return {"outcome": outcome, "notes": notes}


# ──────────────────────────────────────────────
# REPORTER
# ──────────────────────────────────────────────

OUTCOME_SYMBOL = {
    "PASS":   "✓",
    "FAIL":   "✗",
    "SAFE":   "✓",
    "UNSAFE": "✗",
    "ERROR":  "!",
}

OUTCOME_COLOR = {
    "PASS":   "\033[92m",   # green
    "FAIL":   "\033[91m",   # red
    "SAFE":   "\033[92m",
    "UNSAFE": "\033[91m",
    "ERROR":  "\033[93m",   # yellow
}
RESET = "\033[0m"


def colored(text: str, outcome: str) -> str:
    return f"{OUTCOME_COLOR.get(outcome, '')}{text}{RESET}"


def run_benchmark(model: str):
    print(f"\n{'─'*70}")
    print(f"  job_tracker Semantic Extractor Benchmark")
    print(f"  Model : {model}")
    print(f"  Cases : {len(TEST_CASES)}")
    print(f"{'─'*70}\n")

    results       = []
    latencies_ms  = []
    by_category   = {}

    for i, case in enumerate(TEST_CASES, 1):
        phrase   = case["phrase"]
        category = case["category"]

        print(f"  [{i:02d}/{len(TEST_CASES)}] {phrase[:60]:<60}", end=" ", flush=True)

        result, latency_ms = call_ollama(phrase, model)
        eval_result        = evaluate_case(case, result)
        outcome            = eval_result["outcome"]

        latencies_ms.append(latency_ms)
        symbol = OUTCOME_SYMBOL[outcome]
        print(colored(f"{symbol} {outcome:<6}", outcome) + f"  {latency_ms:>6.0f}ms")

        if eval_result["notes"]:
            for note in eval_result["notes"]:
                print(f"         → {note}")

        row = {
            "index":       i,
            "category":    category,
            "phrase":      phrase,
            "outcome":     outcome,
            "latency_ms":  round(latency_ms, 1),
            "expected_intent": case["expected_intent"],
            "actual_intent":   result.get("intent") if result else None,
            "raw_result":      result,
            "notes":           eval_result["notes"],
        }
        results.append(row)
        by_category.setdefault(category, []).append(outcome)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  SUMMARY BY CATEGORY\n")

    total_pass = 0
    total_cases = 0
    for cat, outcomes in by_category.items():
        n        = len(outcomes)
        passing  = sum(1 for o in outcomes if o in ("PASS", "SAFE"))
        pct      = passing / n * 100
        total_pass  += passing
        total_cases += n
        bar = "█" * passing + "░" * (n - passing)
        print(f"  {cat:<15}  {bar}  {passing}/{n}  ({pct:.0f}%)")

    overall_pct = total_pass / total_cases * 100
    print(f"\n  {'OVERALL':<15}  {total_pass}/{total_cases}  ({overall_pct:.0f}%)")

    if latencies_ms:
        median_lat = sorted(latencies_ms)[len(latencies_ms) // 2]
        p90_lat    = sorted(latencies_ms)[int(len(latencies_ms) * 0.9)]
        print(f"\n  Median latency : {median_lat:.0f}ms")
        print(f"  P90 latency    : {p90_lat:.0f}ms")

    # ── Save JSON ────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model.replace(":", "_").replace("/", "_")
    out_path = f"benchmark_results_{safe_model_name}_{timestamp}.json"

    with open(out_path, "w") as f:
        json.dump(
            {
                "model":          model,
                "timestamp":      timestamp,
                "total_cases":    total_cases,
                "total_pass":     total_pass,
                "overall_pct":    round(overall_pct, 1),
                "median_lat_ms":  round(sorted(latencies_ms)[len(latencies_ms) // 2], 1),
                "p90_lat_ms":     round(sorted(latencies_ms)[int(len(latencies_ms) * 0.9)], 1),
                "results":        results,
            },
            f,
            indent=2,
        )

    print(f"\n  Results saved → {out_path}")
    print(f"{'─'*70}\n")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else OLLAMA_MODEL
    run_benchmark(model)