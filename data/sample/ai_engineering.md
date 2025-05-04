# AI Engineering Foundations

## Embeddings

Embeddings map text into a dense vector space where semantic similarity becomes geometric proximity.
Popular models such as OpenAI `text-embedding-3-small` produce 1536-dimensional vectors suitable for FAISS.

## FAISS Vector Store

FAISS stores dense embeddings for fast similarity search.
`IndexFlatIP` performs exact inner-product search and works well when vectors are L2-normalized (equivalent to cosine similarity).
For larger corpora, IVF or HNSW indexes trade a small recall loss for much lower latency.

## Chunking Strategy

Chunk size and overlap strongly affect retrieval quality.
Structure-aware splitting that keeps headings with their paragraphs improves contextual relevance.
In internal experiments, tuned chunking lifted contextual relevance by roughly 38% versus naive fixed windows.

## Production Concerns

Caching repeated queries, bounding concurrency, and persisting indexes to disk keep interactive RAG systems responsive under load.
Always evaluate retrieval separately from generation so you can tell whether misses come from indexing or from the LLM.
