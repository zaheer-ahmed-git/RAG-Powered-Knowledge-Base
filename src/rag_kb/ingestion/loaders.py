"""Document loaders for PDF, Markdown, and plain text corpora."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from llama_index.core import Document, SimpleDirectoryReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".markdown"}


def load_documents(
    data_dir: Path | str,
    recursive: bool = True,
) -> List[Document]:
    """Load documents from a directory with source metadata attached."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    reader = SimpleDirectoryReader(
        input_dir=str(root),
        recursive=recursive,
        required_exts=list(SUPPORTED_EXTENSIONS),
        filename_as_id=True,
    )
    documents = reader.load_data()
    for doc in documents:
        source = doc.metadata.get("file_path") or doc.metadata.get("file_name", "unknown")
        doc.metadata.setdefault("source", source)
        doc.metadata.setdefault("doc_type", Path(str(source)).suffix.lstrip(".") or "txt")
    logger.info("Loaded %d documents from %s", len(documents), root)
    return documents


def load_document_paths(paths: Iterable[Path | str]) -> List[Document]:
    """Load an explicit list of file paths as LlamaIndex documents."""
    docs: List[Document] = []
    for path in paths:
        p = Path(path)
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning("Skipping unsupported file: %s", p)
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        docs.append(
            Document(
                text=text,
                metadata={
                    "source": str(p),
                    "file_name": p.name,
                    "doc_type": p.suffix.lstrip("."),
                },
            )
        )
    return docs
