"""Structure-aware chunking that preserves headings and section context."""

from __future__ import annotations

import re
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode

from rag_kb.config import get_settings

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _section_path(text_before: str) -> str:
    """Build a breadcrumb of the nearest Markdown headings."""
    headings = HEADING_RE.findall(text_before)
    if not headings:
        return ""
    # Keep last heading per level for a compact path
    levels: dict[int, str] = {}
    for hashes, title in headings:
        level = len(hashes)
        levels = {k: v for k, v in levels.items() if k < level}
        levels[level] = title.strip()
    return " > ".join(levels[k] for k in sorted(levels))


def structure_aware_chunk(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[BaseNode]:
    """
    Split documents into overlapping sentence chunks and attach section metadata.

    Chunk size/overlap defaults are tuned for contextual relevance on technical docs.
    """
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    splitter = SentenceSplitter(chunk_size=size, chunk_overlap=overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    enriched: List[BaseNode] = []
    for node in nodes:
        if not isinstance(node, TextNode):
            enriched.append(node)
            continue

        source_text = ""
        for doc in documents:
            if doc.doc_id == node.ref_doc_id or doc.metadata.get("source") == node.metadata.get(
                "source"
            ):
                source_text = doc.text or ""
                break

        # Approximate position for section path
        snippet = node.get_content()[:80]
        idx = source_text.find(snippet) if snippet and source_text else -1
        section = _section_path(source_text[:idx]) if idx > 0 else ""

        node.metadata["section_path"] = section
        node.metadata["chunk_size"] = size
        node.metadata["chunk_overlap"] = overlap
        enriched.append(node)

    return enriched
