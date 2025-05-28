import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llama_index.core.schema import TextNode

from rag_kb.retrieval.hybrid import ScoredNode
from rag_kb.retrieval.mmr import mmr_rerank


def test_mmr_prefers_diverse_content():
    nodes = [
        ScoredNode(
            node=TextNode(text="FAISS stores dense embeddings for similarity search.", id_="a"),
            score=1.0,
        ),
        ScoredNode(
            node=TextNode(text="FAISS stores dense embeddings for similarity search again.", id_="b"),
            score=0.95,
        ),
        ScoredNode(
            node=TextNode(text="BM25 ranks documents using sparse keyword overlap.", id_="c"),
            score=0.8,
        ),
    ]
    ranked = mmr_rerank(nodes, top_n=2, lambda_mult=0.5)
    assert len(ranked) == 2
    assert ranked[0].node.node_id == "a"
    # Near-duplicate of a should lose to the different BM25 chunk
    assert ranked[1].node.node_id == "c"
