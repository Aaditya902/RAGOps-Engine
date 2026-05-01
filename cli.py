#!/usr/bin/env python3
"""
cli.py — Command-line interface for Ask My Docs.

12-Factor Factor XI: CLI writes to stdout; no file logging.

Usage:
  python cli.py ingest ./docs
  python cli.py ingest ./docs --glob "**/*.pdf"
  python cli.py ask "What is the refund policy?"
  python cli.py ask "Summarise findings" --hyde
  python cli.py eval --dataset evaluation/eval_dataset.jsonl
"""
from __future__ import annotations

import argparse
import sys

from core.logging import configure_logging

configure_logging()


def cmd_ingest(args) -> None:
    from pathlib import Path
    from ingestion.chunker import ingest_file, ingest_directory
    from ingestion.embedder import embed_chunks, get_embedder
    from ingestion.indexer import index_chunks

    print("Verifying embedding model dimensions...")
    get_embedder().verify_dim(expected_dim=__import__("config").settings.embed_dim)

    path = Path(args.path)
    print(f"Ingesting: {path}")

    chunks = ingest_file(path) if path.is_file() else ingest_directory(path, args.glob)
    print(f"  {len(chunks)} chunks extracted")

    print("Embedding...")
    chunks = embed_chunks(chunks)

    print("Writing to dual index...")
    stats = index_chunks(chunks)

    print(f"\n✓ Indexed {stats.total_chunks} chunks from {stats.total_documents} document(s)")
    print(f"  Collection: {stats.collection_name}")


def cmd_ask(args) -> None:
    from retrieval.pipeline import hybrid_retrieve
    from generation.generator import generate

    print(f"\nQuery: {args.query}")
    print("─" * 60)

    retrieval = hybrid_retrieve(args.query, use_hyde=args.hyde)
    result = generate(args.query, retrieval)

    print(f"\n{result.answer}\n")
    print("─" * 60)

    if result.sources:
        print(f"Sources ({len(result.sources)}):")
        for i, src in enumerate(result.sources, 1):
            page = f" p.{src.page}" if src.page else ""
            print(f"  [{i}] {src.source}{page}")

    print(
        f"\nLatency — retrieval: {retrieval.retrieval_ms:.0f}ms  "
        f"rerank: {retrieval.rerank_ms:.0f}ms  "
        f"generation: {result.generation_ms:.0f}ms"
    )

    if result.has_uncited_claims:
        print("⚠  Warning: response may contain uncited claims")


def cmd_eval(args) -> None:
    from evaluation.evaluator import run_ci_eval
    run_ci_eval(args.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask My Docs — production RAG CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest documents into the dual index")
    p_ingest.add_argument("path", help="File or directory to ingest")
    p_ingest.add_argument("--glob", default="**/*", help="Glob pattern for directories")

    p_ask = sub.add_parser("ask", help="Query the RAG pipeline")
    p_ask.add_argument("query", help="Question to answer")
    p_ask.add_argument("--hyde", action="store_true", help="Enable HyDE query expansion")

    p_eval = sub.add_parser("eval", help="Run RAGAS evaluation CI gate")
    p_eval.add_argument("--dataset", required=True, help="Path to JSONL evaluation dataset")

    args = parser.parse_args()
    {"ingest": cmd_ingest, "ask": cmd_ask, "eval": cmd_eval}[args.command](args)


if __name__ == "__main__":
    main()