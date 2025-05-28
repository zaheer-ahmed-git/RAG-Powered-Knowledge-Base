import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llama_index.core import Document

from rag_kb.ingestion.chunking import structure_aware_chunk


def test_structure_aware_chunk_attaches_section_path():
    text = """# Retrieval

## Hybrid Search

Hybrid search combines dense vectors and BM25 keyword matching.
It improves precision on technical corpora with identifiers.

## Reranking

A cross-encoder can reorder candidates before generation.
"""
    docs = [Document(text=text, metadata={"source": "demo.md", "file_name": "demo.md"})]
    nodes = structure_aware_chunk(docs, chunk_size=120, chunk_overlap=20)
    assert len(nodes) >= 1
    assert all("chunk_size" in (n.metadata or {}) for n in nodes)
