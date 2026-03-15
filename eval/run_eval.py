"""
eval/run_eval.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

THE ONLY SCORE THAT MATTERS:
    python eval/run_eval.py --dataset financebench --seed 42

ABSOLUTE RULE:
    Never post projections as scores.
    Only post CONFIRMED scores from this script.

Usage:
    python eval/run_eval.py --dataset financebench --seed 42
    python eval/run_eval.py --dataset bizbench --seed 42
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

# ── C5: Set seed before anything else ───────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Add project root to path ─────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.seed_manager import SeedManager
from src.utils.resource_governor import ResourceGovernor


def load_financebench() -> List[Dict]:
    """
    Load FinanceBench dataset from HuggingFace.
    Source: PatronusAI/financebench
    150 expert questions on real SEC filings.
    No API key required. Free public dataset.
    """
    try:
        from datasets import load_dataset
        print("[eval] Loading FinanceBench from HuggingFace...")
        print("[eval] Source: PatronusAI/financebench")
        ds = load_dataset("PatronusAI/financebench", split="train")
        questions = []
        for row in ds:
            questions.append({
                "question_id":   row.get("question_id", ""),
                "question":      row.get("question", ""),
                "answer":        row.get("answer", ""),
                "company":       row.get("company", ""),
                "doc_type":      row.get("doc_type", ""),
                "fiscal_year":   row.get("fiscal_year_end", ""),
                "question_type": row.get("question_type", ""),
            })
        print(f"[eval] Loaded {len(questions)} questions.")
        return questions

    except Exception as e:
        print(f"[eval] WARNING: Could not load FinanceBench: {e}")
        print("[eval] Using 3-question stub for structure testing.")
        return [
            {
                "question_id":   "FB-001",
                "question":      "What was Apple's net income for FY2022?",
                "answer":        "99.803 billion",
                "company":       "Apple Inc",
                "doc_type":      "10-K",
                "fiscal_year":   "FY2022",
                "question_type": "numerical",
            },
            {
                "question_id":   "FB-002",
                "question":      "What was Microsoft's revenue for FY2023?",
                "answer":        "211.915 billion",
                "company":       "Microsoft",
                "doc_type":      "10-K",
                "fiscal_year":   "FY2023",
                "question_type": "numerical",
            },
            {
                "question_id":   "FB-003",
                "question":      "What was Tesla's gross profit margin for FY2022?",
                "answer":        "25.6%",
                "company":       "Tesla",
                "doc_type":      "10-K",
                "fiscal_year":   "FY2022",
                "question_type": "ratio",
            },
        ]


def stub_pipeline(
    question:    str,
    company:     str,
    doc_type:    str,
    fiscal_year: str,
) -> str:
    """
    STUB: Returns empty string until full pipeline is built.
    Week 9 Gate M4: every question must return non-empty.
    Replace this with real pipeline call at Week 9:

    from src.pipeline.pipeline import FinBenchPipeline
    pipeline = FinBenchPipeline()
    return pipeline.run(question, company, doc_type, fiscal_year)
    """
    return ""


def score_answer(predicted: str, gold: str) -> bool:
    """
    Exact-match scoring with normalisation.
    Strips whitespace, lowercases, removes $ , % symbols.
    """
    if not predicted or not gold:
        return False

    def normalise(s: str) -> str:
        return (
            s.lower()
             .strip()
             .replace(",", "")
             .replace("$", "")
             .replace("%", "")
             .replace(" ", "")
        )

    return normalise(predicted) == normalise(gold)


def run_chi_square(correct: int, total: int, baseline_pct: float = 0.0):
    """
    Statistical significance test vs baseline.
    PDR spec: p < 0.05 required before ANY public announcement.
    """
    from scipy.stats import chi2_contingency

    observed_correct   = correct
    observed_incorrect = total - correct
    expected_correct   = max(1, int(total * baseline_pct))
    expected_incorrect = max(1, total - expected_correct)

    table = [
        [observed_correct,   expected_correct],
        [observed_incorrect, expected_incorrect],
    ]

    try:
        chi2, p_value, dof, _ = chi2_contingency(table)
        return float(chi2), float(p_value)
    except Exception:
        return 0.0, 1.0


def run_eval(dataset: str = "financebench", seed: int = 42) -> Dict:
    """
    Main evaluation function.
    Writes results to results.json in project root.
    Returns results dict.
    """
    # C5: enforce seed
    SeedManager.set_all(seed)

    # C4: check RAM before starting
    ResourceGovernor.check("eval start")

    print(f"\n{'='*60}")
    print(f"  FinBench Multi-Agent Business Analyst AI")
    print(f"  Evaluation Script")
    print(f"  Dataset  : {dataset}")
    print(f"  Seed     : {seed}  (C5 enforced)")
    print(f"  Time     : {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    # Load questions
    if dataset == "financebench":
        questions = load_financebench()
    else:
        print(f"[eval] '{dataset}' not supported yet. Using FinanceBench.")
        questions = load_financebench()

    total   = len(questions)
    correct = 0
    results = []

    for i, q in enumerate(questions):

        # C4: check RAM every 10 questions
        if i % 10 == 0:
            ResourceGovernor.check(f"eval question {i+1}/{total}")

        predicted = stub_pipeline(
            question    = q["question"],
            company     = q["company"],
            doc_type    = q["doc_type"],
            fiscal_year = q["fiscal_year"],
        )

        is_correct = score_answer(predicted, q["answer"])
        if is_correct:
            correct += 1

        results.append({
            "question_id":   q["question_id"],
            "question":      q["question"],
            "gold_answer":   q["answer"],
            "predicted":     predicted,
            "correct":       is_correct,
            "company":       q["company"],
            "doc_type":      q["doc_type"],
            "fiscal_year":   q["fiscal_year"],
            "question_type": q.get("question_type", ""),
        })

        # Progress every 10 questions
        if (i + 1) % 10 == 0 or (i + 1) == total:
            pct = (correct / (i + 1)) * 100
            print(f"  [{i+1:3d}/{total}]  Running accuracy: {pct:.1f}%")

    accuracy = (correct / total * 100) if total > 0 else 0.0

    # Statistical significance vs 0% baseline
    chi2, p_value = run_chi_square(correct, total, baseline_pct=0.0)

    # C9: verify no _rlef_ keys in output
    c9_passed = not any("_rlef_" in str(r) for r in results)

    output = {
        "dataset":          dataset,
        "seed":             seed,
        "timestamp":        datetime.now().isoformat(),
        "total_questions":  total,
        "correct":          correct,
        "accuracy_pct":     round(accuracy, 2),
        "chi2_vs_baseline": round(chi2, 4),
        "p_value":          round(p_value, 4),
        "p_significant":    p_value < 0.05,
        "c9_rlef_check":    c9_passed,
        "gate_m8_ready":    accuracy >= 82.0 and p_value < 0.05,
        "results":          results,
    }

    # Write results.json to project root
    output_path = ROOT / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  RESULT   : {accuracy:.1f}%  ({correct}/{total} correct)")
    print(f"  Chi2     : {chi2:.4f}  |  p-value : {p_value:.4f}")
    print(f"  Sig.     : {p_value < 0.05}")
    print(f"  C9 check : {'PASS' if c9_passed else 'FAIL'}")
    print(f"  M8 ready : {accuracy >= 82.0 and p_value < 0.05}")
    print(f"  Output   : {output_path}")
    print(f"{'='*60}\n")

    return output


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FinBench Evaluation Script — the only score that matters"
    )
    parser.add_argument(
        "--dataset",
        default="financebench",
        choices=["financebench", "bizbench"],
        help="Which benchmark to run (default: financebench)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed — must be 42 (C5)"
    )
    args = parser.parse_args()

    if args.seed != 42:
        print(f"[C5 VIOLATION] seed must be 42, got {args.seed}. Overriding.")
        args.seed = 42

    run_eval(dataset=args.dataset, seed=args.seed)