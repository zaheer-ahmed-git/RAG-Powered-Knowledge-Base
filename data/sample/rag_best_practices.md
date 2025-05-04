# RAG Best Practices

## What is RAG?

Retrieval-Augmented Generation (RAG) grounds LLM answers in private documents.
A retriever selects relevant passages; a generator writes the final answer from that evidence.

## Hybrid Search

Hybrid search in RAG combines dense vector retrieval with sparse keyword retrieval such as BM25.
Dense models catch paraphrases; BM25 catches exact identifiers, API names, and rare tokens.
Fusing ranked lists with Reciprocal Rank Fusion (RRF) avoids mixing incompatible score scales.
On technical question sets, hybrid search often improves precision by about 22% versus dense-only baselines.

## Knowledge Graphs

Why build a knowledge graph for multi-hop questions?
Vector search alone struggles when the answer requires connecting facts across documents.
Automated knowledge-graph construction extracts entities and relations from chunks, enabling multi-hop reasoning over linked concepts.
Graph hits should still be validated against source text before generation.

## Reranking and Context Assembly

Retrieve broadly (top 10–20), then rerank down to 3–5 passages for the prompt.
Drop near-duplicate chunks, preserve section metadata, and cite sources in the answer.
Evaluate Recall@k and MRR for retrieval; judge groundedness separately for answers.
