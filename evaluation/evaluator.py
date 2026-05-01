"""
evaluation/evaluator.py — RAGAS-based evaluation pipeline with a CI gate.

SOLID — Single Responsibility Principle (SRP):
  Three clearly separated concerns:
    - load_eval_dataset: I/O only, no evaluation logic
    - run_ragas: RAGAS scoring only, no CI logic
    - run_ci_eval: orchestrates the above and enforces thresholds

KISS Principle:
  The CI gate is just: run eval → check thresholds → exit(1) if failing.
  No retry logic, no complex state — simple and transparent.

12-Factor Factor XI (Logs):
  Results are printed to stdout as structured output. The CI runner
  captures stdout; no file writing needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import structlog
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from config import settings
from models import EvalMetrics, EvalSample

log = structlog.get_logger()


# ── Dataset I/O ────────────────────────────────────────────────────────────────

def load_eval_dataset(path: str | Path) -> list[EvalSample]:
    """Load evaluation samples from a JSONL file (one JSON object per line)."""
    samples: list[EvalSample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            samples.append(EvalSample(**json.loads(line)))
    log.info("eval_dataset_loaded", path=str(path), n=len(samples))
    return samples


def generate_missing_answers(samples: list[EvalSample]) -> list[EvalSample]:
    """
    For samples without an answer, run the live RAG pipeline to generate one.
    This lets you keep a stable question+ground_truth set and evaluate
    whatever the current system produces.
    """
    from retrieval.pipeline import hybrid_retrieve
    from generation.generator import generate

    updated: list[EvalSample] = []
    for sample in samples:
        if sample.answer:
            updated.append(sample)
            continue

        retrieval = hybrid_retrieve(sample.question)
        result = generate(sample.question, retrieval)
        updated.append(
            EvalSample(
                question=sample.question,
                ground_truth=sample.ground_truth,
                contexts=[c.chunk.text for c in retrieval.chunks],
                answer=result.answer,
            )
        )
        log.debug("answer_generated_for_eval", question=sample.question[:60])

    return updated


# ── RAGAS scoring ──────────────────────────────────────────────────────────────

def run_ragas(samples: list[EvalSample]) -> EvalMetrics:
    """Run RAGAS over the samples and return structured EvalMetrics."""
    if not samples:
        raise ValueError("Cannot evaluate an empty sample set.")

    data: dict[str, list[Any]] = {
        "question": [s.question for s in samples],
        "answer": [s.answer for s in samples],
        "contexts": [s.contexts for s in samples],
        "ground_truth": [s.ground_truth for s in samples],
    }
    dataset = Dataset.from_dict(data)

    log.info("ragas_evaluation_starting", n_samples=len(samples))
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    log.info("ragas_evaluation_done", scores=dict(result))

    return EvalMetrics(
        faithfulness=float(result["faithfulness"]),
        answer_relevancy=float(result["answer_relevancy"]),
        context_precision=float(result["context_precision"]),
        context_recall=float(result["context_recall"]),
        n_samples=len(samples),
    )


# ── CI gate ────────────────────────────────────────────────────────────────────

def run_ci_eval(dataset_path: str | Path) -> EvalMetrics:
    """
    Full CI evaluation:
      1. Load eval dataset
      2. Generate any missing answers via the live pipeline
      3. Score with RAGAS
      4. Print results to stdout
      5. sys.exit(1) if any metric is below its configured threshold
    """
    samples = load_eval_dataset(dataset_path)
    samples = generate_missing_answers(samples)
    metrics = run_ragas(samples)

    print("\n" + "=" * 52)
    print("  RAG Evaluation Results")
    print("=" * 52)
    print(f"  faithfulness       {metrics.faithfulness:.3f}  (min {settings.min_faithfulness})")
    print(f"  answer_relevancy   {metrics.answer_relevancy:.3f}  (min {settings.min_answer_relevancy})")
    print(f"  context_precision  {metrics.context_precision:.3f}  (min {settings.min_context_precision})")
    print(f"  context_recall     {metrics.context_recall:.3f}  (min {settings.min_context_recall})")
    print(f"  samples            {metrics.n_samples}")
    print("=" * 52)

    passed, failures = metrics.passed_ci(
        min_faithfulness=settings.min_faithfulness,
        min_answer_relevancy=settings.min_answer_relevancy,
        min_context_precision=settings.min_context_precision,
        min_context_recall=settings.min_context_recall,
    )

    if passed:
        print("  CI GATE: PASSED ✓")
    else:
        print("  CI GATE: FAILED ✗")
        for msg in failures:
            print(f"    - {msg}")
        print("=" * 52 + "\n")
        sys.exit(1)

    print("=" * 52 + "\n")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG CI evaluation gate")
    parser.add_argument("--dataset", required=True, help="Path to JSONL eval dataset")
    args = parser.parse_args()
    run_ci_eval(args.dataset)