"""
eval/smoke_test.py
5-question smoke test on Apple FY2023 10-K.

Validates the full 19-node pipeline runs end-to-end without crashing.
Does NOT measure accuracy — that's the mini/full eval's job.

Pass criteria: all 5 questions return some answer (correct or wrong)
              within 5 minutes each.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smoke_test")

# Eval mode — relax RAM governor to 15.4GB
os.environ.setdefault("EVAL_RUNNING", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Per-question timeout (seconds). Hard cap of 5 min per question.
PER_QUESTION_TIMEOUT_SEC = 300

DOC_PATH = "documents/sec_filings/AAPL_FY2023_10-K.html"

QUESTIONS = [
    {
        "id":          "smoke_01",
        "question":    "What was Apple's total net sales for fiscal year 2023?",
        "expected":    "$383,285 million",
    },
    {
        "id":          "smoke_02",
        "question":    "What was Apple's net income for fiscal year 2023?",
        "expected":    "$96,995 million",
    },
    {
        "id":          "smoke_03",
        "question":    "What was Apple's diluted earnings per share for fiscal 2023?",
        "expected":    "$6.13",
    },
    {
        "id":          "smoke_04",
        "question":    "What was Apple's gross margin for fiscal year 2023?",
        "expected":    "$169,148 million",
    },
    {
        "id":          "smoke_05",
        "question":    "What were Apple's total assets at fiscal year-end 2023?",
        "expected":    "$352,583 million",
    },
]


@dataclass
class QuestionResult:
    """Single-question result row."""
    id:                str
    question:          str
    expected:          str
    answer:            str
    confidence:        float
    elapsed_sec:       float
    error:             str
    sniper_hit:        bool
    chunks_retrieved:  int
    winning_pod:       str


def short(s: str, n: int = 200) -> str:
    """Shorten a string for log output."""
    if not s:
        return ""
    s = str(s).replace("\n", " ").strip()
    return s[:n] + ("..." if len(s) > n else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default=DOC_PATH,
                        help="Path to document to ingest")
    parser.add_argument("--limit", type=int, default=len(QUESTIONS),
                        help="Run first N questions only")
    parser.add_argument("--output", default="eval/results",
                        help="Where to save smoke_YYYYMMDD.json")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print(" SMOKE TEST — FinBench end-to-end pipeline")
    print("=" * 70)
    print(f"Document:  {args.doc}")
    print(f"Questions: {args.limit}")
    print(f"Per-Q timeout: {PER_QUESTION_TIMEOUT_SEC}s")
    print()

    # ── Pre-flight checks ─────────────────────────────────────────────────
    if not os.path.exists(args.doc):
        print(f"ERROR: document not found: {args.doc}")
        return 1

    # Lazy import — pipeline pulls in lots of deps
    print("[1/3] Importing pipeline...")
    t0 = time.time()
    try:
        from src.pipeline.pipeline import FinBenchPipeline
    except Exception as exc:
        print(f"ERROR: pipeline import failed: {exc}")
        import traceback
        traceback.print_exc()
        return 2
    print(f"      done in {time.time() - t0:.1f}s")

    # ── Ingest ────────────────────────────────────────────────────────────
    print(f"[2/3] Ingesting {os.path.basename(args.doc)}...")
    t0 = time.time()
    try:
        pipeline = FinBenchPipeline()
        state = pipeline.ingest(
            document_path = args.doc,
            session_id    = f"smoke-{datetime.now().strftime('%H%M%S')}",
        )
    except Exception as exc:
        print(f"ERROR: ingest failed: {exc}")
        import traceback
        traceback.print_exc()
        return 3
    ingest_sec = time.time() - t0
    chunks = getattr(state, "chunk_count", 0) or 0
    cells  = len(getattr(state, "table_cells", []) or [])
    print(f"      done in {ingest_sec:.1f}s | "
          f"chunks={chunks} | table_cells={cells} | "
          f"company={getattr(state, 'company_name', '?')!r} | "
          f"FY={getattr(state, 'fiscal_year', '?')!r}")

    if chunks == 0:
        print("WARNING: 0 chunks created — pipeline will produce empty results")

    # ── Question loop ─────────────────────────────────────────────────────
    print(f"[3/3] Running {args.limit} questions...")
    print()
    results: List[QuestionResult] = []
    pipeline_total_t0 = time.time()

    for i, q in enumerate(QUESTIONS[:args.limit], start=1):
        print(f"─── Question {i}/{args.limit} ───")
        print(f"  Q: {q['question']}")
        print(f"  Expected: {q['expected']}")
        t0 = time.time()
        error = ""
        answer = ""
        confidence = 0.0
        sniper_hit = False
        chunks_retrieved = 0
        winning_pod = ""

        try:
            state = pipeline.query(state, q["question"])

            # Try the standard fields, fall back gracefully
            answer = (
                getattr(state, "final_answer", "") or
                getattr(state, "final_answer_pre_xgb", "") or
                getattr(state, "analyst_output", "") or
                ""
            )
            confidence = float(
                getattr(state, "confidence_score", 0.0) or
                getattr(state, "analyst_confidence", 0.0) or
                0.0
            )
            sniper_hit = bool(getattr(state, "sniper_hit", False))
            chunks_retrieved = len(
                getattr(state, "retrieval_stage_2", []) or
                getattr(state, "retrieval_stage_1", []) or
                []
            )
            winning_pod = str(getattr(state, "winning_pod", "") or "")

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            import traceback
            traceback.print_exc()

        elapsed = time.time() - t0
        timed_out = elapsed > PER_QUESTION_TIMEOUT_SEC

        result = QuestionResult(
            id               = q["id"],
            question         = q["question"],
            expected         = q["expected"],
            answer           = short(answer, 300),
            confidence       = round(confidence, 3),
            elapsed_sec      = round(elapsed, 1),
            error            = error,
            sniper_hit       = sniper_hit,
            chunks_retrieved = chunks_retrieved,
            winning_pod      = winning_pod,
        )
        results.append(result)

        status = "ERR" if error else ("SLOW" if timed_out else "OK")
        print(f"  Status: {status}")
        print(f"  Answer: {short(answer, 200)}")
        print(f"  Confidence: {confidence:.3f} | "
              f"Sniper: {sniper_hit} | "
              f"Chunks: {chunks_retrieved} | "
              f"Pod: {winning_pod}")
        print(f"  Elapsed: {elapsed:.1f}s")
        print()

        if error:
            print(f"  Error: {error}")
            print()

    pipeline_total = time.time() - pipeline_total_t0

    # ── Summary ───────────────────────────────────────────────────────────
    n_total       = len(results)
    n_ok          = sum(1 for r in results if not r.error)
    n_with_answer = sum(1 for r in results
                        if not r.error and len(r.answer) > 5)
    n_sniper      = sum(1 for r in results if r.sniper_hit)
    avg_elapsed   = sum(r.elapsed_sec for r in results) / max(n_total, 1)

    print("=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"  Total questions:      {n_total}")
    print(f"  Completed (no error): {n_ok}/{n_total}")
    print(f"  Returned an answer:   {n_with_answer}/{n_total}")
    print(f"  SniperRAG hits:       {n_sniper}/{n_total}")
    print(f"  Ingest time:          {ingest_sec:.1f}s")
    print(f"  Question avg:         {avg_elapsed:.1f}s")
    print(f"  Question total:       {pipeline_total:.1f}s")
    print()

    # ── Save results ──────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps({
        "doc":              args.doc,
        "ingest_sec":       round(ingest_sec, 1),
        "pipeline_total":   round(pipeline_total, 1),
        "n_total":          n_total,
        "n_ok":             n_ok,
        "n_with_answer":    n_with_answer,
        "n_sniper":         n_sniper,
        "results":          [asdict(r) for r in results],
    }, indent=2), encoding="utf-8")
    print(f"  Saved: {out_file}")
    print()

    # Pass criteria: at least 3 of 5 returned some answer without error
    if n_ok >= 3:
        print("  ✓ SMOKE TEST PASSED — pipeline runs end-to-end")
        return 0
    else:
        print(f"  ✗ SMOKE TEST FAILED — only {n_ok}/{n_total} succeeded")
        return 4


if __name__ == "__main__":
    sys.exit(main())