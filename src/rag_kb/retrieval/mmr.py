"""Maximal Marginal Relevance (MMR) to diversify retrieved chunks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Sequence, Set

if TYPE_CHECKING:
    from rag_kb.retrieval.hybrid import ScoredNode

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def mmr_rerank(
    candidates: Sequence["ScoredNode"],
    top_n: int,
    lambda_mult: float = 0.7,
) -> List["ScoredNode"]:
    """
    Re-order fused candidates to balance relevance and diversity.

    Uses token-overlap Jaccard as a cheap redundancy signal so near-duplicate
    chunks are less likely to fill the entire LLM context window.
    """
    if not candidates or top_n <= 0:
        return []
    if len(candidates) <= top_n:
        return list(candidates)

    token_sets: Dict[str, Set[str]] = {
        c.node.node_id: set(_tokenize(c.node.get_content())) for c in candidates
    }
    max_score = max(c.score for c in candidates) or 1.0
    remaining = list(candidates)
    selected: List["ScoredNode"] = []

    while remaining and len(selected) < top_n:
        best_idx = 0
        best_val = float("-inf")
        for i, cand in enumerate(remaining):
            relevance = cand.score / max_score
            if not selected:
                mmr_score = relevance
            else:
                redundancy = max(
                    _jaccard(token_sets[cand.node.node_id], token_sets[s.node.node_id])
                    for s in selected
                )
                mmr_score = lambda_mult * relevance - (1.0 - lambda_mult) * redundancy
            if mmr_score > best_val:
                best_val = mmr_score
                best_idx = i
        chosen = remaining.pop(best_idx)
        chosen.sources = {**(chosen.sources or {}), "mmr": best_val}
        selected.append(chosen)

    return selected
