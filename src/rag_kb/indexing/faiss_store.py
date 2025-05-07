"""FAISS vector index lifecycle: build, persist, and query."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import faiss
import numpy as np
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from rag_kb.config import Settings, get_settings

logger = logging.getLogger(__name__)


class FaissIndexManager:
    """Manages a FAISS IndexFlatIP store wrapped by LlamaIndex."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._embed_model = OpenAIEmbedding(
            model=self.settings.openai_embed_model,
            api_key=self.settings.openai_api_key or None,
        )
        self._index: Optional[VectorStoreIndex] = None
        self._faiss_index: Optional[faiss.Index] = None
        self._node_ids: List[str] = []

    @property
    def index(self) -> Optional[VectorStoreIndex]:
        return self._index

    def build(self, nodes: Sequence[BaseNode]) -> VectorStoreIndex:
        """Embed nodes and build an inner-product FAISS index."""
        dim = self.settings.embed_dim
        self._faiss_index = faiss.IndexFlatIP(dim)
        vector_store = FaissVectorStore(faiss_index=self._faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self._index = VectorStoreIndex(
            nodes=list(nodes),
            storage_context=storage_context,
            embed_model=self._embed_model,
            show_progress=True,
        )
        self._node_ids = [n.node_id for n in nodes]
        logger.info("Built FAISS index with %d vectors (dim=%d)", len(nodes), dim)
        return self._index

    def persist(self, directory: Optional[Path] = None) -> None:
        """Persist FAISS index bytes and node id map to disk."""
        if self._index is None or self._faiss_index is None:
            raise RuntimeError("No index to persist. Call build() first.")

        out = directory or self.settings.faiss_dir
        out.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._faiss_index, str(out / "index.faiss"))
        self._index.storage_context.persist(persist_dir=str(out / "llama"))
        (out / "node_ids.json").write_text(
            json.dumps(self._node_ids, indent=2),
            encoding="utf-8",
        )
        logger.info("Persisted FAISS index to %s", out)

    def load(self, directory: Optional[Path] = None) -> VectorStoreIndex:
        """Reload a previously persisted FAISS index."""
        from llama_index.core import load_index_from_storage

        root = directory or self.settings.faiss_dir
        self._faiss_index = faiss.read_index(str(root / "index.faiss"))
        vector_store = FaissVectorStore(faiss_index=self._faiss_index)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(root / "llama"),
        )
        self._index = load_index_from_storage(
            storage_context,
            embed_model=self._embed_model,
        )
        ids_path = root / "node_ids.json"
        if ids_path.exists():
            self._node_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        logger.info("Loaded FAISS index from %s", root)
        return self._index

    def similarity_search(self, query: str, top_k: int = 10) -> List[BaseNode]:
        """Dense retrieval against the FAISS index."""
        if self._index is None:
            raise RuntimeError("Index not ready. Call build() or load() first.")
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)
        return [r.node for r in results]

    def embed_query(self, query: str) -> np.ndarray:
        """Return a normalized query embedding for manual FAISS search."""
        vec = np.array(self._embed_model.get_query_embedding(query), dtype="float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
