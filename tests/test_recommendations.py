import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llama_index.core.schema import TextNode

from rag_kb.indexing.knowledge_graph import KnowledgeGraphBuilder
from rag_kb.retrieval.hybrid import HybridRetriever
from rag_kb.retrieval.recommendations import ContextualRecommender


def test_contextual_recommendations_exclude_primary():
    nodes = [
        TextNode(
            text="LlamaIndex builds indexes over documents for RAG pipelines.",
            id_="a",
            metadata={"source": "li.md", "section_path": "Indexing > LlamaIndex"},
        ),
        TextNode(
            text="LangChain orchestrates prompts, tools, and chains around LLMs.",
            id_="b",
            metadata={"source": "lc.md", "section_path": "Orchestration > LangChain"},
        ),
        TextNode(
            text="FAISS provides efficient vector similarity search for embeddings.",
            id_="c",
            metadata={"source": "faiss.md", "section_path": "Indexing > FAISS"},
        ),
    ]
    kg = KnowledgeGraphBuilder()
    kg.build_from_nodes(nodes)
    retriever = HybridRetriever(nodes, faiss_manager=None, kg_builder=kg)
    recommender = ContextualRecommender(retriever, nodes)
    primary = retriever.retrieve("LlamaIndex indexes", top_n=1, use_graph=True)
    recs = recommender.recommend("LlamaIndex indexes", primary=primary, limit=3)
    primary_ids = {p.node.node_id for p in primary}
    assert all(r.node_id not in primary_ids for r in recs)


def test_knowledge_graph_extracts_triples():
    nodes = [
        TextNode(
            text="LangChain uses OpenAI models. Hybrid search combines FAISS and BM25.",
            id_="g1",
            metadata={"source": "kg.md"},
        )
    ]
    kg = KnowledgeGraphBuilder()
    graph = kg.build_from_nodes(nodes)
    assert len(graph.triples) >= 1
    assert len(graph.nodes) >= 2
