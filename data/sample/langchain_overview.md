# LangChain and LlamaIndex Together

## Orchestration Roles

LangChain is used for orchestration of prompts, tools, memory, and output parsing around LLMs.
LlamaIndex specializes in document indexing, retrieval abstractions, and vector store integrations such as FAISS.
Using both is common: LlamaIndex retrieves context; LangChain runs the grounded generation chain.

## Typical Pipeline

1. Load and chunk documents with LlamaIndex.
2. Embed chunks and store them in FAISS.
3. Build a BM25 index over the same corpus for hybrid search.
4. Optionally extract a property-style knowledge graph for multi-hop paths.
5. Fuse candidates, assemble a prompt, and call ChatOpenAI through LangChain.

## Contextual Recommendations

After answering, suggest related sections the user did not ask for.
Recommendations can use shared section paths, graph neighbors, and residual hybrid scores.
This turns a one-shot Q&A into a guided exploration of the knowledge base.

## API Surface

Expose `/ingest`, `/query`, and `/recommend` behind FastAPI.
Run embedding and LLM work in a thread pool so the event loop stays free for concurrent clients.
Cache normalized queries with a short TTL to cut repeated latency.
