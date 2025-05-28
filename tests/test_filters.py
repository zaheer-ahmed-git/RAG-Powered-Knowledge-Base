import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llama_index.core.schema import TextNode

from rag_kb.retrieval.hybrid import HybridRetriever, ScoredNode


def test_metadata_filter_by_source_substring():
    nodes = [
        TextNode(
            text="Hybrid search combines FAISS and BM25.",
            id_="1",
            metadata={"source": "data/sample/rag_best_practices.md", "doc_type": "md"},
        ),
        TextNode(
            text="Cooking pasta needs boiling water.",
            id_="2",
            metadata={"source": "data/sample/cooking.md", "doc_type": "md"},
        ),
    ]
    retriever = HybridRetriever(nodes, faiss_manager=None, kg_builder=None)
    hits = retriever.retrieve(
        "hybrid FAISS BM25",
        top_k=5,
        top_n=5,
        use_graph=False,
        use_mmr=False,
        source_contains="rag_best",
    )
    assert hits
    assert all("rag_best" in str((h.node.metadata or {}).get("source")) for h in hits)
