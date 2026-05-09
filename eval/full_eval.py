"""
Full FinanceBench-style Evaluation — 200 questions across 7 SEC filings
PDR-BAAAI-001 · Rev 1.0

Runs the FinBench pipeline on 200 financial questions covering:
  - 7 companies: Apple, Amazon, Google, Meta, Microsoft, Nvidia, Tesla
  - 28 question types per company (numerical financial metrics)
  - 4 cross-document comparisons

Usage:
    python eval/full_eval.py
    python eval/full_eval.py --limit 50           # First 50 only
    python eval/full_eval.py --start 100 --limit 50  # Q100-149
    python eval/full_eval.py --skip-cross-doc     # Skip 4 cross-doc Qs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Path setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────────────────────────────────
# Company configuration
# ──────────────────────────────────────────────────────────────────────────

COMPANIES: List[Dict[str, str]] = [
    {"ticker": "AAPL",  "name": "Apple Inc.",       "fy": "FY2023", "doc": "AAPL_FY2023_10-K.html"},
    {"ticker": "AMZN",  "name": "Amazon",           "fy": "FY2023", "doc": "AMZN_FY2023_10-K.html"},
    {"ticker": "GOOGL", "name": "Alphabet (Google)","fy": "FY2023", "doc": "GOOGL_FY2023_10-K.html"},
    {"ticker": "META",  "name": "Meta Platforms",   "fy": "FY2023", "doc": "META_FY2023_10-K.html"},
    {"ticker": "MSFT",  "name": "Microsoft",        "fy": "FY2023", "doc": "MSFT_FY2023_10-K.html"},
    {"ticker": "NVDA",  "name": "Nvidia",           "fy": "FY2024", "doc": "NVDA_FY2024_10-K.html"},
    {"ticker": "TSLA",  "name": "Tesla",            "fy": "FY2023", "doc": "TSLA_FY2023_10-K.html"},
]


# ──────────────────────────────────────────────────────────────────────────
# Question templates (28 per company × 7 = 196, +4 cross-doc = 200)
# ──────────────────────────────────────────────────────────────────────────

QUESTION_TEMPLATES: List[Dict[str, str]] = [
    # ── Income statement ────────────────────────────────────────────────
    {"key": "revenue",        "q": "What was {company}'s total revenue (or net sales) in {fy}?"},
    {"key": "cogs",           "q": "What was {company}'s cost of revenue (or cost of sales) in {fy}?"},
    {"key": "gross_profit",   "q": "What was {company}'s gross profit in {fy}?"},
    {"key": "operating_income","q": "What was {company}'s operating income in {fy}?"},
    {"key": "net_income",     "q": "What was {company}'s net income in {fy}?"},
    {"key": "eps_diluted",    "q": "What was {company}'s diluted earnings per share in {fy}?"},
    {"key": "eps_basic",      "q": "What was {company}'s basic earnings per share in {fy}?"},
    {"key": "r_and_d",        "q": "What was {company}'s research and development expense in {fy}?"},
    {"key": "sga",            "q": "What was {company}'s selling, general and administrative expense in {fy}?"},
    {"key": "interest_expense","q": "What was {company}'s interest expense in {fy}?"},
    {"key": "income_tax",     "q": "What was {company}'s provision for income taxes in {fy}?"},
    {"key": "ebit",           "q": "What was {company}'s EBIT in {fy}?"},

    # ── Balance sheet ────────────────────────────────────────────────────
    {"key": "total_assets",   "q": "What were {company}'s total assets at the end of {fy}?"},
    {"key": "total_liab",     "q": "What were {company}'s total liabilities at the end of {fy}?"},
    {"key": "equity",         "q": "What was {company}'s total stockholders equity at the end of {fy}?"},
    {"key": "cash",           "q": "How much cash and cash equivalents did {company} have at the end of {fy}?"},
    {"key": "long_term_debt", "q": "What was {company}'s long-term debt at the end of {fy}?"},
    {"key": "current_assets", "q": "What were {company}'s total current assets at the end of {fy}?"},
    {"key": "current_liab",   "q": "What were {company}'s total current liabilities at the end of {fy}?"},
    {"key": "accounts_recv",  "q": "What was {company}'s accounts receivable at the end of {fy}?"},
    {"key": "inventory",      "q": "What was {company}'s total inventory at the end of {fy}?"},
    {"key": "goodwill",       "q": "What was {company}'s goodwill at the end of {fy}?"},

    # ── Cash flow ────────────────────────────────────────────────────────
    {"key": "operating_cf",   "q": "What was {company}'s net cash provided by operating activities in {fy}?"},
    {"key": "capex",          "q": "What was {company}'s capital expenditures in {fy}?"},
    {"key": "free_cash_flow", "q": "What was {company}'s free cash flow in {fy}?"},
    {"key": "dividends_paid", "q": "How much did {company} pay in dividends in {fy}?"},
    {"key": "share_repurchase","q": "How much did {company} spend on share repurchases in {fy}?"},
    {"key": "deferred_revenue","q": "What was {company}'s deferred revenue at the end of {fy}?"},
]


# ──────────────────────────────────────────────────────────────────────────
# Cross-document questions (currently weak — testing baseline only)
# ──────────────────────────────────────────────────────────────────────────

CROSS_DOC_QUESTIONS: List[Dict[str, str]] = [
    {
        "id":       "cross_001",
        "company":  "Apple vs Microsoft",
        "question": "Compare Apple's and Microsoft's total revenue for fiscal year 2023.",
        "doc":      "AAPL_FY2023_10-K.html",
        "fy":       "FY2023",
        "key":      "compare_revenue",
    },
    {
        "id":       "cross_002",
        "company":  "Google vs Meta",
        "question": "Which had higher net income in 2023, Google or Meta, and by how much?",
        "doc":      "GOOGL_FY2023_10-K.html",
        "fy":       "FY2023",
        "key":      "compare_net_income",
    },
    {
        "id":       "cross_003",
        "company":  "Tesla vs Nvidia",
        "question": "Compare the gross margins of Tesla (FY2023) and Nvidia (FY2024).",
        "doc":      "TSLA_FY2023_10-K.html",
        "fy":       "FY2023",
        "key":      "compare_gross_margin",
    },
    {
        "id":       "cross_004",
        "company":  "Among 7 companies",
        "question": "Among Apple, Amazon, Google, Meta, Microsoft, Nvidia, and Tesla, which had the largest total assets in their most recent 10-K?",
        "doc":      "AAPL_FY2023_10-K.html",
        "fy":       "FY2023",
        "key":      "largest_assets",
    },
]


# ──────────────────────────────────────────────────────────────────────────
# Build full question list
# ──────────────────────────────────────────────────────────────────────────

def build_questions(skip_cross_doc: bool = False) -> List[Dict[str, str]]:
    """Build the full 200-question list."""
    questions: List[Dict[str, str]] = []
    qid = 1

    # Per-company questions
    for company in COMPANIES:
        for tpl in QUESTION_TEMPLATES:
            q_text = tpl["q"].format(company=company["name"], fy=company["fy"])
            questions.append({
                "id":       f"q{qid:03d}",
                "company":  company["name"],
                "ticker":   company["ticker"],
                "fy":       company["fy"],
                "doc":      company["doc"],
                "key":      tpl["key"],
                "question": q_text,
            })
            qid += 1

    # Cross-document
    if not skip_cross_doc:
        for cd in CROSS_DOC_QUESTIONS:
            questions.append({
                "id":       cd["id"],
                "company":  cd["company"],
                "ticker":   "MULTI",
                "fy":       cd["fy"],
                "doc":      cd["doc"],
                "key":      cd["key"],
                "question": cd["question"],
            })

    return questions


# ──────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ──────────────────────────────────────────────────────────────────────────

def import_pipeline():
    """Import FinBenchPipeline lazily."""
    from src.pipeline.pipeline import FinBenchPipeline
    return FinBenchPipeline


def ingest_document(pipeline_cls, doc_path: str) -> Optional[object]:
    """Ingest one document and return the pipeline instance ready for queries."""
    print(f"  [ingest] {doc_path} ...", flush=True)
    t0 = time.time()
    try:
        pipeline = pipeline_cls()
        pipeline.ingest(doc_path)
        elapsed = time.time() - t0
        print(f"  [ingest] done in {elapsed:.1f}s", flush=True)
        return pipeline
    except Exception as exc:
        print(f"  [ingest] FAILED: {exc}", flush=True)
        traceback.print_exc()
        return None


def run_one_question(pipeline, question_dict: Dict, timeout_sec: int = 120) -> Dict:
    """Run a single question, return result dict."""
    qid = question_dict["id"]
    qtext = question_dict["question"]

    result: Dict[str, Any] = {
        "id":           qid,
        "company":      question_dict["company"],
        "ticker":       question_dict["ticker"],
        "fy":           question_dict["fy"],
        "doc":          question_dict["doc"],
        "key":          question_dict["key"],
        "question":     qtext,
        "answer":       "",
        "confidence":   0.0,
        "winning_pod":  "",
        "sniper_hit":   False,
        "chunks_used":  0,
        "elapsed_sec":  0.0,
        "status":       "OK",
        "error":        "",
    }

    t0 = time.time()
    try:
        out = pipeline.query(qtext)
        elapsed = time.time() - t0

        result["answer"]      = str(out.get("final_answer", ""))[:500]
        result["confidence"]  = float(out.get("confidence_score", 0.0))
        result["winning_pod"] = str(out.get("winning_pod", ""))[:50]
        result["sniper_hit"]  = bool(out.get("sniper_hit", False))
        result["chunks_used"] = int(out.get("chunks_used", 0))
        result["elapsed_sec"] = round(elapsed, 2)

    except Exception as exc:
        result["status"]      = "ERROR"
        result["error"]       = f"{type(exc).__name__}: {str(exc)[:200]}"
        result["elapsed_sec"] = round(time.time() - t0, 2)

    return result


# ──────────────────────────────────────────────────────────────────────────
# Main eval loop
# ──────────────────────────────────────────────────────────────────────────

def run_eval(args) -> Dict:
    """Run the full eval, return summary dict."""
    questions = build_questions(skip_cross_doc=args.skip_cross_doc)

    # Apply --start and --limit
    if args.start > 0:
        questions = questions[args.start:]
    if args.limit > 0:
        questions = questions[:args.limit]

    print(f"\n{'='*70}")
    print(f" FULL FINANCEBENCH EVAL — {len(questions)} questions")
    print(f"{'='*70}")
    print(f"Questions: {len(questions)}")
    print(f"Companies: {len(COMPANIES)}")
    print(f"Per-Q timeout: {args.timeout}s")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*70}\n", flush=True)

    pipeline_cls = import_pipeline()

    # Group questions by document for efficient ingestion
    by_doc: Dict[str, List[Dict]] = {}
    for q in questions:
        by_doc.setdefault(q["doc"], []).append(q)

    print(f"Documents to ingest: {len(by_doc)}\n", flush=True)

    # Output paths
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(args.output_dir, f"full_eval_{timestamp}.json")
    out_csv  = os.path.join(args.output_dir, f"full_eval_{timestamp}.csv")

    # Run questions per-document
    all_results: List[Dict] = []
    grand_t0 = time.time()

    for doc_idx, (doc, qs) in enumerate(by_doc.items(), start=1):
        print(f"\n{'─'*70}")
        print(f"Document {doc_idx}/{len(by_doc)}: {doc} ({len(qs)} questions)")
        print(f"{'─'*70}", flush=True)

        doc_path = os.path.join("documents", "sec_filings", doc)
        if not os.path.exists(doc_path):
            print(f"  ✗ Document not found: {doc_path}", flush=True)
            for q in qs:
                all_results.append({
                    **q,
                    "answer": "", "confidence": 0.0, "winning_pod": "",
                    "sniper_hit": False, "chunks_used": 0,
                    "elapsed_sec": 0.0,
                    "status": "ERROR", "error": "document_not_found",
                })
            continue

        pipeline = ingest_document(pipeline_cls, doc_path)
        if pipeline is None:
            for q in qs:
                all_results.append({
                    **q,
                    "answer": "", "confidence": 0.0, "winning_pod": "",
                    "sniper_hit": False, "chunks_used": 0,
                    "elapsed_sec": 0.0,
                    "status": "ERROR", "error": "ingestion_failed",
                })
            continue

        for q_idx, q in enumerate(qs, start=1):
            print(f"  Q{q_idx}/{len(qs)} [{q['id']}] {q['key']:18s} ", end="", flush=True)
            res = run_one_question(pipeline, q, timeout_sec=args.timeout)
            all_results.append(res)

            status = "✓" if res["status"] == "OK" else "✗"
            preview = res["answer"][:80].replace("\n", " ")
            print(f"{status} {res['elapsed_sec']:5.1f}s | {preview}", flush=True)

            # Save incrementally (in case Colab disconnects)
            with open(out_json, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    grand_elapsed = time.time() - grand_t0

    # ── Summary ──────────────────────────────────────────────────────────
    total = len(all_results)
    ok    = sum(1 for r in all_results if r["status"] == "OK")
    err   = sum(1 for r in all_results if r["status"] == "ERROR")
    sniper_hits = sum(1 for r in all_results if r.get("sniper_hit"))
    answered = sum(1 for r in all_results if r["status"] == "OK" and r["answer"].strip())

    avg_conf = (
        sum(r["confidence"] for r in all_results if r["status"] == "OK") / max(1, ok)
    )
    avg_elapsed = (
        sum(r["elapsed_sec"] for r in all_results) / max(1, total)
    )

    print(f"\n{'='*70}")
    print(f" SUMMARY")
    print(f"{'='*70}")
    print(f"Total questions:     {total}")
    print(f"  Completed (OK):    {ok}/{total} ({100*ok/max(1,total):.1f}%)")
    print(f"  Errors:            {err}/{total} ({100*err/max(1,total):.1f}%)")
    print(f"  Returned answer:   {answered}/{total} ({100*answered/max(1,total):.1f}%)")
    print(f"  SniperRAG hits:    {sniper_hits}/{total} ({100*sniper_hits/max(1,total):.1f}%)")
    print()
    print(f"Avg confidence:      {avg_conf:.3f}")
    print(f"Avg elapsed/Q:       {avg_elapsed:.2f}s")
    print(f"Total elapsed:       {grand_elapsed/60:.1f} min")
    print()
    print(f"Saved JSON: {out_json}")
    print(f"{'='*70}\n", flush=True)

    # Per-company breakdown
    print(f"\nPer-company breakdown:")
    print(f"{'Company':<20}  {'Total':>6}  {'OK':>6}  {'Sniper':>6}  {'Avg conf':>8}")
    print(f"{'-'*55}")
    by_company: Dict[str, List[Dict]] = {}
    for r in all_results:
        by_company.setdefault(r["company"], []).append(r)
    for company, rs in by_company.items():
        c_total  = len(rs)
        c_ok     = sum(1 for r in rs if r["status"] == "OK")
        c_sniper = sum(1 for r in rs if r.get("sniper_hit"))
        c_conf   = sum(r["confidence"] for r in rs if r["status"] == "OK") / max(1, c_ok)
        print(f"{company:<20}  {c_total:>6}  {c_ok:>6}  {c_sniper:>6}  {c_conf:>8.3f}")

    return {
        "total":         total,
        "ok":            ok,
        "errors":        err,
        "answered":      answered,
        "sniper_hits":   sniper_hits,
        "avg_confidence":avg_conf,
        "avg_elapsed":   avg_elapsed,
        "total_elapsed": grand_elapsed,
        "output_json":   out_json,
    }


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Full FinanceBench-style eval")
    ap.add_argument("--limit",          type=int, default=0,  help="Max questions (0 = all)")
    ap.add_argument("--start",          type=int, default=0,  help="Start index")
    ap.add_argument("--timeout",        type=int, default=120,help="Per-Q timeout (sec)")
    ap.add_argument("--output-dir",     type=str, default="eval/results", help="Output dir")
    ap.add_argument("--skip-cross-doc", action="store_true",  help="Skip cross-doc Qs")
    args = ap.parse_args()

    summary = run_eval(args)

    # Exit with non-zero if too many errors
    if summary["errors"] > summary["total"] * 0.3:
        print(f"⚠ {summary['errors']}/{summary['total']} errors — exit 1")
        sys.exit(1)

    print("✓ Eval complete")


if __name__ == "__main__":
    main()