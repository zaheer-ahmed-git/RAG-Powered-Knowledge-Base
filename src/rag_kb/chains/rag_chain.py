"""LangChain + LlamaIndex orchestration for grounded RAG answers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from llama_index.core.schema import BaseNode

from rag_kb.cache import query_cache
from rag_kb.config import Settings, get_settings
from rag_kb.indexing.faiss_store import FaissIndexManager
from rag_kb.indexing.knowledge_graph import KnowledgeGraphBuilder
from rag_kb.ingestion.chunking import structure_aware_chunk
from rag_kb.ingestion.loaders import load_documents
from rag_kb.retrieval.hybrid import HybridRetriever, ScoredNode
from rag_kb.retrieval.recommendations import ContextualRecommender, Recommendation

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a knowledge-base assistant. Answer ONLY using the provided context.
If the context is insufficient, say you do not know. Cite sources by file name when possible.
Keep answers concise and factual."""

USER_PROMPT = """Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGAnswer:
    answer: str
    sources: List[dict] = field(default_factory=list)
    recommendations: List[dict] = field(default_factory=list)
    cached: bool = False


class KnowledgeBasePipeline:
    """End-to-end ingest → hybrid retrieve → LangChain generate → recommend."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.nodes: List[BaseNode] = []
        self.faiss = FaissIndexManager(self.settings)
        self.kg = KnowledgeGraphBuilder(self.settings)
        self.retriever: Optional[HybridRetriever] = None
        self.recommender: Optional[ContextualRecommender] = None
        self._llm = ChatOpenAI(
            model=self.settings.openai_chat_model,
            api_key=self.settings.openai_api_key or None,
            temperature=0.1,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
        )
        self._chain = self._prompt | self._llm | StrOutputParser()

    def ingest(self, data_dir: Optional[str] = None, persist: bool = True) -> int:
        root = data_dir or str(self.settings.data_dir)
        documents = load_documents(root)
        self.nodes = structure_aware_chunk(documents)
        self.faiss.build(self.nodes)
        self.kg.build_from_nodes(self.nodes)
        self.retriever = HybridRetriever(self.nodes, self.faiss, self.kg, self.settings)
        self.recommender = ContextualRecommender(self.retriever, self.nodes)

        if persist:
            self.settings.storage_dir.mkdir(parents=True, exist_ok=True)
            self.faiss.persist()
            self.kg.persist()
            self.retriever.persist_bm25()
            # Persist a lightweight node dump for reload without re-embedding text
            import json

            dump = [
                {
                    "id": n.node_id,
                    "text": n.get_content(),
                    "metadata": n.metadata,
                }
                for n in self.nodes
            ]
            path = self.settings.docstore_dir
            path.mkdir(parents=True, exist_ok=True)
            (path / "nodes.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")

        logger.info("Ingested %d chunks from %s", len(self.nodes), root)
        return len(self.nodes)

    def load(self) -> int:
        """Reload FAISS, BM25 corpus, and knowledge graph from disk."""
        import json

        from llama_index.core.schema import TextNode

        nodes_path = self.settings.docstore_dir / "nodes.json"
        raw = json.loads(nodes_path.read_text(encoding="utf-8"))
        self.nodes = [
            TextNode(text=item["text"], id_=item["id"], metadata=item.get("metadata") or {})
            for item in raw
        ]
        self.faiss.load()
        self.kg.load()
        self.retriever = HybridRetriever(self.nodes, self.faiss, self.kg, self.settings)
        self.recommender = ContextualRecommender(self.retriever, self.nodes)
        return len(self.nodes)

    def corpus_stats(self) -> dict:
        """Return lightweight index / graph / cache statistics."""
        sources = set()
        doc_types: dict[str, int] = {}
        for node in self.nodes:
            meta = node.metadata or {}
            src = meta.get("source") or meta.get("file_name")
            if src:
                sources.add(str(src))
            dtype = str(meta.get("doc_type") or "unknown")
            doc_types[dtype] = doc_types.get(dtype, 0) + 1
        kg_triples = len(self.kg.graph.triples) if self.kg and self.kg.graph else 0
        kg_entities = len(self.kg.graph.nodes) if self.kg and self.kg.graph else 0
        return {
            "ready": self.retriever is not None,
            "chunks": len(self.nodes),
            "unique_sources": len(sources),
            "sources": sorted(sources),
            "doc_types": doc_types,
            "knowledge_graph": {"entities": kg_entities, "triples": kg_triples},
            "cache": query_cache.stats,
            "mmr_enabled": self.settings.use_mmr,
        }

    def query(
        self,
        question: str,
        use_cache: bool = True,
        include_recommendations: bool = True,
        use_mmr: Optional[bool] = None,
        source_contains: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> RAGAnswer:
        if self.retriever is None:
            raise RuntimeError("Pipeline not ready. Call ingest() or load() first.")

        cache_key = query_cache.make_key(
            question,
            recs=include_recommendations,
            mmr=use_mmr if use_mmr is not None else self.settings.use_mmr,
            source=source_contains or "",
            doc_type=doc_type or "",
        )
        if use_cache:
            cached = query_cache.get(cache_key)
            if cached is not None:
                cached.cached = True
                return cached

        hits: List[ScoredNode] = self.retriever.retrieve(
            question,
            use_mmr=use_mmr,
            source_contains=source_contains,
            doc_type=doc_type,
        )
        context = self._format_context(hits)
        answer_text = self._chain.invoke({"context": context, "question": question})

        sources = [
            {
                "node_id": h.node.node_id,
                "score": h.score,
                "source": (h.node.metadata or {}).get("source"),
                "section": (h.node.metadata or {}).get("section_path"),
                "preview": h.node.get_content()[:240],
                "channels": h.sources,
            }
            for h in hits
        ]

        recs: List[Recommendation] = []
        if include_recommendations and self.recommender:
            recs = self.recommender.recommend(question, primary=hits)

        result = RAGAnswer(
            answer=answer_text,
            sources=sources,
            recommendations=[
                {
                    "title": r.title,
                    "source": r.source,
                    "score": r.score,
                    "reason": r.reason,
                }
                for r in recs
            ],
            cached=False,
        )
        if use_cache:
            query_cache.set(cache_key, result)
        return result

    @staticmethod
    def _format_context(hits: List[ScoredNode]) -> str:
        blocks = []
        for i, h in enumerate(hits, start=1):
            meta = h.node.metadata or {}
            header = meta.get("file_name") or meta.get("source") or f"chunk-{i}"
            section = meta.get("section_path")
            label = f"[{i}] {header}" + (f" :: {section}" if section else "")
            blocks.append(f"{label}\n{h.node.get_content()}")
        return "\n\n---\n\n".join(blocks)
