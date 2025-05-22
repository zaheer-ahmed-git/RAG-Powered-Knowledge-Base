#!/usr/bin/env python3
"""CLI: ingest documents into FAISS + BM25 + knowledge graph stores."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from rag_kb.chains.rag_chain import KnowledgeBasePipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a document corpus into the knowledge base")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory of PDF/MD/TXT files (default: DATA_DIR from settings)",
    )
    parser.add_argument("--no-persist", action="store_true", help="Skip writing indexes to disk")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    pipeline = KnowledgeBasePipeline()
    count = pipeline.ingest(args.data_dir, persist=not args.no_persist)
    print(f"Ingested {count} chunks.")


if __name__ == "__main__":
    main()
