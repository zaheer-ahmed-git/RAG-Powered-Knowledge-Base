import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llama_index.core.schema import TextNode

from rag_kb.retrieval.hybrid import HybridRetriever, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("FAISS + BM25 Hybrid!") == ["faiss", "bm25", "hybrid"]


def test_hybrid_bm25_retrieves_keyword_match():
    nodes = [
        TextNode(
            text="FAISS stores dense embeddings for fast similarity search.",
            id_="n1",
            metadata={"source": "faiss.md"},
        ),
        TextNode(
            text="Cooking pasta requires boiling water and salt.",
            id_="n2",
            metadata={"source": "cooking.md"},
        ),
        TextNode(
            text="BM25 ranks documents by term frequency and document length.",
            id_="n3",
            metadata={"source": "bm25.md"},
        ),
    ]
    retriever = HybridRetriever(nodes, faiss_manager=None, kg_builder=None)
    hits = retriever.retrieve("BM25 term frequency ranking", top_k=3, top_n=2, use_graph=False)
    assert hits
    assert "bm25" in str(hits[0].node.metadata.get("source")).lower()


def test_rrf_fuse_prefers_shared_ranks():
    fused = HybridRetriever._rrf_fuse(
        [{"a": 0, "b": 1}, {"b": 0, "c": 1}],
        weights=[0.5, 0.5],
    )
    assert fused[0][0] == "b"
