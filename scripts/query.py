#!/usr/bin/env python3
"""CLI: ask a question against the persisted knowledge base."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from rag_kb.chains.rag_chain import KnowledgeBasePipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG knowledge base")
    parser.add_argument("question", help="Natural-language question")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-recs", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print raw JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    pipeline = KnowledgeBasePipeline()
    pipeline.load()
    result = pipeline.query(
        args.question,
        use_cache=not args.no_cache,
        include_recommendations=not args.no_recs,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "answer": result.answer,
                    "sources": result.sources,
                    "recommendations": result.recommendations,
                    "cached": result.cached,
                },
                indent=2,
            )
        )
        return

    print("\n=== Answer ===\n")
    print(result.answer)
    print("\n=== Sources ===\n")
    for s in result.sources:
        print(f"- {s.get('source')} (score={s.get('score'):.4f})")
    if result.recommendations:
        print("\n=== Related ===\n")
        for r in result.recommendations:
            print(f"- {r['title']}: {r['reason']}")


if __name__ == "__main__":
    main()
