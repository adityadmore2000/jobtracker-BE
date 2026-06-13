"""
job_tracker — Semantic Extractor Benchmark (real-schema edition)
================================================================

This benchmark exercises the ACTUAL single-call extractor used by the backend:
``app.semantic_command_extractor.extract_semantic_command_once``. It does NOT
re-implement a parallel prompt — it imports the real module so the system prompt
under test is exactly the one the backend ships. To benchmark a different model,
pass it on the CLI; the model is injected via OllamaSettings (env is not mutated).

HOW TO RUN
----------
    cd jobtracker-BE
    source .venv/bin/activate
    python evaluation/benchmark_extractor.py                 # default llama3.2:3b
    python evaluation/benchmark_extractor.py qwen2.5:7b-instruct

WHAT IT DOES
------------
Sends each test phrase through the real extractor and checks the returned
``SemanticCommand`` (intent / target / changes / note) against expectations.

Writes a timestamped JSON results file for run-to-run comparison.

REQUIREMENTS
------------
- Ollama running locally (OLLAMA_BASE_URL, default http://127.0.0.1:11434)
- The model under test must already be pulled.

SCHEMA NOTE
-----------
The real extractor returns a SemanticCommand:
    intent ∈ {create_application, update_application, append_note,
              archive_application, unsupported}
    target.{company, role, application_id}
    changes.{status, priority, location_mode, employment_types,
             current_stages, job_link, engaged_days, next_action, comments}
    note, clarification, suggested_phrasings

Because status/priority/location updates to an existing row all share a single
``update_application`` intent, this benchmark distinguishes them by the populated
``changes`` field rather than by a dedicated intent. Informational queries and
list requests have no dedicated intent in this system, so the *safe* answer is
``unsupported`` (a non-mutating intent): the must_not_mutate cases assert exactly
that.
"""

import datetime
import json
import sys
import time
from pathlib import Path

# Make the app importable when run from jobtracker-BE/.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.semantic_command_extractor import (  # noqa: E402
    SemanticExtractorError,
    extract_semantic_command_once,
)
from app.semantic_interpreter import get_ollama_settings  # noqa: E402

DEFAULT_MODEL = "llama3.2:3b"

# Intents that mutate tracker state (used for the must_not_mutate safety check).
MUTATING_INTENTS = {
    "create_application",
    "update_application",
    "append_note",
    "archive_application",
}

# ──────────────────────────────────────────────
# TEST CASES
# ──────────────────────────────────────────────
#
# Fields per case:
#   category          — grouping for the report
#   phrase            — user utterance
#   expected_intent   — required SemanticCommand.intent
#   expected_target   — dict of target fields that must match (loose compare)
#   expected_changes  — dict of changes fields that must match (loose compare)
#   expect_note       — True if .note must be non-empty
#   must_not_mutate   — intent must be a NON-mutating intent (here: unsupported)
#
# Loose comparison lowercases/strips strings; list values compare as
# lowercased sets.

TEST_CASES = [
    # ── DIRECT COMMANDS ────────────────────────────────────────────────────
    {
        "category": "direct",
        "phrase": "I applied for data scientist role at Spotify",
        "expected_intent": "create_application",
        "expected_target": {"company": "Spotify", "role": "data scientist"},
        "expected_changes": {"status": "applied"},
    },
    {
        "category": "direct",
        "phrase": "add application for software engineer at Netflix",
        "expected_intent": "create_application",
        "expected_target": {"company": "Netflix"},
    },
    {
        "category": "direct",
        "phrase": "For Spotify data scientist application, set status to rejected",
        "expected_intent": "update_application",
        "expected_target": {"company": "Spotify"},
        "expected_changes": {"status": "rejected"},
    },
    {
        "category": "direct",
        "phrase": "For Google AI Engineer application, set priority to high",
        "expected_intent": "update_application",
        "expected_target": {"company": "Google"},
        "expected_changes": {"priority": "HIGH"},
    },
    {
        "category": "direct",
        "phrase": "add a note for Spotify saying recruiter reached out",
        "expected_intent": "append_note",
        "expect_note": True,
    },
    {
        "category": "direct",
        "phrase": "archive the Netflix application",
        "expected_intent": "archive_application",
        "expected_target": {"company": "Netflix"},
    },

    # ── INDIRECT COMMANDS (the gap we're closing) ──────────────────────────
    {
        "category": "indirect",
        "phrase": "i should track at Spotify for data scientist role",
        "expected_intent": "create_application",
        "expected_target": {"company": "Spotify"},
    },
    {
        "category": "indirect",
        "phrase": "been meaning to add the Netflix software engineer role",
        "expected_intent": "create_application",
        "expected_target": {"company": "Netflix"},
    },
    {
        "category": "indirect",
        "phrase": "i think i applied to Google last week",
        "expected_intent": "create_application",
        "expected_target": {"company": "Google"},
        "expected_changes": {"status": "applied"},
    },
    {
        "category": "indirect",
        "phrase": "the Spotify one should be higher priority",
        "expected_intent": "update_application",
        "expected_target": {"company": "Spotify"},
        "expected_changes": {"priority": "HIGH"},
    },
    {
        "category": "indirect",
        "phrase": "For Spotify, bump it to high priority",
        "expected_intent": "update_application",
        "expected_target": {"company": "Spotify"},
        "expected_changes": {"priority": "HIGH"},
    },
    {
        "category": "indirect",
        "phrase": "Netflix just got back to me, I'm in touch with them now",
        "expected_intent": "update_application",
        "expected_target": {"company": "Netflix"},
        "expected_changes": {"status": "in_touch"},
    },
    {
        "category": "indirect",
        "phrase": "the Google role is remote by the way",
        "expected_intent": "update_application",
        "expected_target": {"company": "Google"},
        "expected_changes": {"location_mode": "remote"},
    },
    {
        "category": "indirect",
        "phrase": "Spotify turned me down",
        "expected_intent": "update_application",
        "expected_target": {"company": "Spotify"},
        "expected_changes": {"status": "rejected"},
    },
    {
        "category": "indirect",
        "phrase": "got accepted at Netflix!",
        "expected_intent": "update_application",
        "expected_target": {"company": "Netflix"},
        "expected_changes": {"status": "accepted"},
    },
    {
        "category": "indirect",
        "phrase": "jot down for Spotify that the recruiter said they'll follow up Friday",
        "expected_intent": "append_note",
        "expect_note": True,
    },
    {
        "category": "indirect",
        "phrase": "i'm not pursuing Netflix anymore, take it off my active list",
        "expected_intent": "archive_application",
        "expected_target": {"company": "Netflix"},
    },

    # ── INFORMATIONAL QUESTIONS (must NOT mutate state) ────────────────────
    {
        "category": "informational",
        "phrase": "how many applications do I have?",
        "must_not_mutate": True,
    },
    {
        "category": "informational",
        "phrase": "what's the status of my Spotify application?",
        "must_not_mutate": True,
    },
    {
        "category": "informational",
        "phrase": "show me all my applications",
        "must_not_mutate": True,
    },
    {
        "category": "informational",
        "phrase": "which ones are high priority?",
        "must_not_mutate": True,
    },
    {
        "category": "informational",
        "phrase": "have I heard back from Netflix?",
        "must_not_mutate": True,
    },

    # ── AMBIGUOUS / NONSENSE (safe → unsupported) ──────────────────────────
    {
        "category": "ambiguous",
        "phrase": "do something with Spotify",
        "expected_intent": "unsupported",
    },
    {
        "category": "ambiguous",
        "phrase": "hm",
        "expected_intent": "unsupported",
    },
]


# ──────────────────────────────────────────────
# EXTRACTOR CALL
# ──────────────────────────────────────────────

def call_extractor(phrase: str, settings) -> tuple[dict | None, float]:
    """Call the real extractor. Returns (command_dict_or_None, latency_ms)."""
    start = time.perf_counter()
    try:
        command, _metrics = extract_semantic_command_once(phrase, None, settings=settings)
        latency_ms = (time.perf_counter() - start) * 1000
        return command.model_dump(), latency_ms
    except SemanticExtractorError:
        latency_ms = (time.perf_counter() - start) * 1000
        return None, latency_ms
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000
        return None, latency_ms


# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────

def _norm(value):
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, list):
        return {item.strip().lower() for item in value if isinstance(item, str)}
    return value


def evaluate_case(case: dict, result: dict | None) -> dict:
    if result is None:
        return {"outcome": "ERROR", "notes": ["Extractor failed or returned unparseable output"]}

    notes: list[str] = []
    actual_intent = result.get("intent", "")

    # must_not_mutate takes priority: any mutating intent is UNSAFE.
    if case.get("must_not_mutate", False):
        if actual_intent in MUTATING_INTENTS:
            return {
                "outcome": "UNSAFE",
                "notes": [f"Returned mutating intent '{actual_intent}' for informational query"],
            }
        return {"outcome": "SAFE", "notes": []}

    expected_intent = case.get("expected_intent")
    if expected_intent and actual_intent != expected_intent:
        notes.append(f"Intent: expected '{expected_intent}', got '{actual_intent}'")

    target = result.get("target") or {}
    for key, expected_val in case.get("expected_target", {}).items():
        if _norm(target.get(key)) != _norm(expected_val):
            notes.append(f"target.{key}: expected '{expected_val}', got '{target.get(key)}'")

    changes = result.get("changes") or {}
    for key, expected_val in case.get("expected_changes", {}).items():
        if _norm(changes.get(key)) != _norm(expected_val):
            notes.append(f"changes.{key}: expected '{expected_val}', got '{changes.get(key)}'")

    if case.get("expect_note", False):
        note_val = result.get("note")
        if not (isinstance(note_val, str) and note_val.strip()):
            notes.append(f"note: expected non-empty, got '{note_val}'")

    return {"outcome": "PASS" if not notes else "FAIL", "notes": notes}


# ──────────────────────────────────────────────
# REPORTER
# ──────────────────────────────────────────────

OUTCOME_SYMBOL = {"PASS": "✓", "FAIL": "✗", "SAFE": "✓", "UNSAFE": "✗", "ERROR": "!"}
OUTCOME_COLOR = {
    "PASS": "\033[92m", "FAIL": "\033[91m", "SAFE": "\033[92m",
    "UNSAFE": "\033[91m", "ERROR": "\033[93m",
}
RESET = "\033[0m"


def colored(text: str, outcome: str) -> str:
    return f"{OUTCOME_COLOR.get(outcome, '')}{text}{RESET}"


def run_benchmark(model: str):
    base_settings = get_ollama_settings()
    settings = base_settings.__class__(
        base_url=base_settings.base_url,
        model=model,
        timeout_seconds=max(base_settings.timeout_seconds, 60.0),
        keep_alive=base_settings.keep_alive,
        max_tool_turns=base_settings.max_tool_turns,
    )

    print(f"\n{'─' * 72}")
    print("  job_tracker Semantic Extractor Benchmark (real schema)")
    print(f"  Model : {model}")
    print(f"  Cases : {len(TEST_CASES)}")
    print(f"{'─' * 72}\n")

    results = []
    latencies_ms = []
    by_category: dict[str, list[str]] = {}

    for i, case in enumerate(TEST_CASES, 1):
        phrase = case["phrase"]
        category = case["category"]
        print(f"  [{i:02d}/{len(TEST_CASES)}] {phrase[:58]:<58}", end=" ", flush=True)

        result, latency_ms = call_extractor(phrase, settings)
        eval_result = evaluate_case(case, result)
        outcome = eval_result["outcome"]

        latencies_ms.append(latency_ms)
        print(colored(f"{OUTCOME_SYMBOL[outcome]} {outcome:<6}", outcome) + f"  {latency_ms:>6.0f}ms")
        for note in eval_result["notes"]:
            print(f"         → {note}")

        results.append({
            "index": i,
            "category": category,
            "phrase": phrase,
            "outcome": outcome,
            "latency_ms": round(latency_ms, 1),
            "expected_intent": case.get("expected_intent"),
            "actual_intent": result.get("intent") if result else None,
            "raw_result": result,
            "notes": eval_result["notes"],
        })
        by_category.setdefault(category, []).append(outcome)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  SUMMARY BY CATEGORY\n")
    total_pass = 0
    total_cases = 0
    cat_scores: dict[str, tuple[int, int]] = {}
    for cat, outcomes in by_category.items():
        n = len(outcomes)
        passing = sum(1 for o in outcomes if o in ("PASS", "SAFE"))
        cat_scores[cat] = (passing, n)
        total_pass += passing
        total_cases += n
        bar = "█" * passing + "░" * (n - passing)
        print(f"  {cat:<15}  {bar}  {passing}/{n}  ({passing / n * 100:.0f}%)")

    overall_pct = total_pass / total_cases * 100
    print(f"\n  {'OVERALL':<15}  {total_pass}/{total_cases}  ({overall_pct:.0f}%)")

    sorted_lat = sorted(latencies_ms)
    median_lat = sorted_lat[len(sorted_lat) // 2]
    p90_lat = sorted_lat[min(int(len(sorted_lat) * 0.9), len(sorted_lat) - 1)]
    print(f"\n  Median latency : {median_lat:.0f}ms")
    print(f"  P90 latency    : {p90_lat:.0f}ms")

    # ── Save JSON ─────────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model.replace(":", "_").replace("/", "_")
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / f"benchmark_results_{safe_model_name}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": model,
                "timestamp": timestamp,
                "total_cases": total_cases,
                "total_pass": total_pass,
                "overall_pct": round(overall_pct, 1),
                "category_scores": {c: {"pass": p, "total": n} for c, (p, n) in cat_scores.items()},
                "median_lat_ms": round(median_lat, 1),
                "p90_lat_ms": round(p90_lat, 1),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved → {out_path}")
    print(f"{'─' * 72}\n")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    run_benchmark(model)
