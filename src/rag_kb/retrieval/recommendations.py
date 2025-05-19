"""Contextual document recommendations based on query + retrieved neighborhood."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

from llama_index.core.schema import BaseNode

from rag_kb.retrieval.hybrid import HybridRetriever, ScoredNode

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    node_id: str
    title: str
    source: str
    score: float
    reason: str


class ContextualRecommender:
    """
    Suggest related passages the user did not explicitly ask for.

    Uses overlapping section paths, shared entities in metadata, and
    complementary hybrid scores from a broadened query neighborhood.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        nodes: Sequence[BaseNode],
    ) -> None:
        self.retriever = retriever
        self.nodes = list(nodes)
        self.node_by_id = {n.node_id: n for n in self.nodes}

    def recommend(
        self,
        query: str,
        primary: Optional[List[ScoredNode]] = None,
        limit: int = 5,
    ) -> List[Recommendation]:
        primary = primary or self.retriever.retrieve(query, top_n=5)
        primary_ids: Set[str] = {s.node.node_id for s in primary}

        # Broaden the candidate pool
        neighborhood = self.retriever.retrieve(query, top_k=30, top_n=15, use_graph=True)
        section_boost = self._section_affinity(primary)

        recs: List[Recommendation] = []
        for scored in neighborhood:
            nid = scored.node.node_id
            if nid in primary_ids:
                continue
            meta = scored.node.metadata or {}
            section = meta.get("section_path", "")
            boost = section_boost.get(section, 0.0)
            final = scored.score + 0.15 * boost
            reason = self._reason(scored, section, boost > 0)
            recs.append(
                Recommendation(
                    node_id=nid,
                    title=self._title(scored.node),
                    source=str(meta.get("source", "unknown")),
                    score=final,
                    reason=reason,
                )
            )

        recs.sort(key=lambda r: r.score, reverse=True)
        logger.debug("Recommendations for %r: %d", query[:40], len(recs[:limit]))
        return recs[:limit]

    def _section_affinity(self, primary: List[ScoredNode]) -> Dict[str, float]:
        counter: Counter[str] = Counter()
        for s in primary:
            section = (s.node.metadata or {}).get("section_path") or ""
            if section:
                counter[section] += 1
        total = sum(counter.values()) or 1
        return {sec: count / total for sec, count in counter.items()}

    @staticmethod
    def _title(node: BaseNode) -> str:
        meta = node.metadata or {}
        section = meta.get("section_path")
        if section:
            return section.split(" > ")[-1]
        source = str(meta.get("file_name") or meta.get("source") or "Passage")
        return source

    @staticmethod
    def _reason(scored: ScoredNode, section: str, same_section: bool) -> str:
        parts = []
        if same_section and section:
            parts.append(f"same section ({section})")
        src = scored.sources or {}
        if src.get("graph", 0) > 0:
            parts.append("linked via knowledge graph")
        if src.get("bm25", 0) > src.get("dense", 0):
            parts.append("strong keyword overlap")
        else:
            parts.append("semantic similarity")
        return "; ".join(parts) if parts else "related context"
