"""Automated knowledge-graph construction for multi-hop reasoning."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
from llama_index.core.schema import BaseNode

from rag_kb.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Lightweight entity patterns for offline / demo construction without LLM calls.
ENTITY_PATTERNS = [
    re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"),
    re.compile(r"\b((?:RAG|FAISS|BM25|LangChain|LlamaIndex|OpenAI|GraphRAG)[A-Za-z0-9-]*)\b"),
]

RELATION_CUES = [
    ("uses", re.compile(r"\b(?:uses|using|built on|powered by)\b", re.I)),
    ("improves", re.compile(r"\b(?:improves?|boosts?|increases?)\b", re.I)),
    ("combines", re.compile(r"\b(?:combines?|hybrid|fuses?|merges?)\b", re.I)),
    ("retrieves", re.compile(r"\b(?:retriev\w+|searches?|indexes?)\b", re.I)),
    ("related_to", re.compile(r"\b(?:related to|associated with|linked to)\b", re.I)),
]


@dataclass
class GraphTriple:
    subject: str
    predicate: str
    object: str
    source_node_id: str
    confidence: float = 0.5


@dataclass
class KnowledgeGraph:
    triples: List[GraphTriple] = field(default_factory=list)
    nodes: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_triple(self, triple: GraphTriple) -> None:
        self.triples.append(triple)
        self.nodes[triple.subject].add(triple.source_node_id)
        self.nodes[triple.object].add(triple.source_node_id)

    def to_networkx(self) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        for t in self.triples:
            g.add_edge(
                t.subject,
                t.object,
                predicate=t.predicate,
                source_node_id=t.source_node_id,
                confidence=t.confidence,
            )
        return g


class KnowledgeGraphBuilder:
    """
    Extract entity-relation triples from chunks and persist a property-style graph.

    Designed for multi-hop queries: seed entities from the question, walk neighbors,
    then map graph hits back to source chunks.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.graph = KnowledgeGraph()

    def build_from_nodes(self, nodes: Sequence[BaseNode]) -> KnowledgeGraph:
        self.graph = KnowledgeGraph()
        for node in nodes:
            text = node.get_content()
            entities = self._extract_entities(text)
            if len(entities) < 2:
                continue
            predicate = self._infer_predicate(text)
            # Connect consecutive distinct entities as local co-occurrence edges
            for i in range(len(entities) - 1):
                subj, obj = entities[i], entities[i + 1]
                if subj == obj:
                    continue
                self.graph.add_triple(
                    GraphTriple(
                        subject=subj,
                        predicate=predicate,
                        object=obj,
                        source_node_id=node.node_id,
                        confidence=0.55 if predicate != "co_occurs_with" else 0.4,
                    )
                )
        logger.info(
            "Knowledge graph: %d triples, %d entities",
            len(self.graph.triples),
            len(self.graph.nodes),
        )
        return self.graph

    def persist(self, directory: Optional[Path] = None) -> Path:
        out = directory or self.settings.graph_dir
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "triples": [asdict(t) for t in self.graph.triples],
            "nodes": {k: sorted(v) for k, v in self.graph.nodes.items()},
        }
        path = out / "knowledge_graph.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Also dump GraphML for visualization tools
        nx.write_graphml(self.graph.to_networkx(), str(out / "knowledge_graph.graphml"))
        logger.info("Persisted knowledge graph to %s", out)
        return path

    def load(self, directory: Optional[Path] = None) -> KnowledgeGraph:
        root = directory or self.settings.graph_dir
        path = root / "knowledge_graph.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.graph = KnowledgeGraph()
        for t in data.get("triples", []):
            self.graph.add_triple(GraphTriple(**t))
        return self.graph

    def multi_hop_retrieve(
        self,
        query: str,
        max_hops: int = 2,
        max_entities: int = 8,
    ) -> List[Tuple[str, List[str]]]:
        """
        Seed entities from the query, walk up to `max_hops`, return (entity, node_ids).
        """
        seeds = self._extract_entities(query)[:max_entities]
        if not seeds:
            return []

        g = self.graph.to_networkx()
        visited: Set[str] = set()
        frontier = list(seeds)
        collected: Dict[str, Set[str]] = defaultdict(set)

        for _ in range(max_hops):
            nxt: List[str] = []
            for ent in frontier:
                if ent in visited or ent not in g:
                    continue
                visited.add(ent)
                collected[ent].update(self.graph.nodes.get(ent, set()))
                for neighbor in list(g.successors(ent)) + list(g.predecessors(ent)):
                    if neighbor not in visited:
                        nxt.append(neighbor)
                        collected[neighbor].update(self.graph.nodes.get(neighbor, set()))
            frontier = nxt

        return [(e, sorted(ids)) for e, ids in collected.items() if ids]

    def _extract_entities(self, text: str) -> List[str]:
        found: List[str] = []
        seen: Set[str] = set()
        for pattern in ENTITY_PATTERNS:
            for match in pattern.findall(text):
                name = match.strip()
                if len(name) < 3 or name.lower() in {"the", "and", "for", "with"}:
                    continue
                key = name.lower()
                if key not in seen:
                    seen.add(key)
                    found.append(name)
        return found[:12]

    def _infer_predicate(self, text: str) -> str:
        for name, pattern in RELATION_CUES:
            if pattern.search(text):
                return name
        return "co_occurs_with"
