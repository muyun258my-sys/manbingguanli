from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence


DEFAULT_PDF_DIR = Path("raw_pdfs")
DEFAULT_VECTOR_DB_DIR = Path("vector_db")
DEFAULT_COLLECTION_NAME = "pdf_knowledge"
DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_LOCAL_EMBEDDING_CACHE_DIR = Path("models/embedding")


@dataclass(frozen=True)
class PageText:
    source_path: str
    pdf_name: str
    category: str
    page: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    id: str
    text: str
    source_path: str
    pdf_name: str
    category: str
    page: int
    chunk_index: int

    def metadata(self) -> dict[str, str | int]:
        return {
            "source_path": self.source_path,
            "pdf_name": self.pdf_name,
            "category": self.category,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }


def discover_pdfs(pdf_dir: Path) -> List[Path]:
    return sorted(path for path in pdf_dir.rglob("*.pdf") if path.is_file())


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, *, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = clean_text(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    sentence_breaks = ".!?\n;:\u3002\uff01\uff1f\uff1b\uff1a"
    min_break = max(80, chunk_size // 2)

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            break_at = max(window.rfind(ch) for ch in sentence_breaks)
            if break_at >= min_break:
                end = start + break_at + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(0, end - overlap)

    return chunks


def infer_category(pdf_path: Path, pdf_dir: Path) -> str:
    try:
        relative = pdf_path.relative_to(pdf_dir)
    except ValueError:
        return "uncategorized"
    if len(relative.parts) <= 1:
        return "uncategorized"
    return relative.parts[0]


def stable_chunk_id(source_path: str, page: int, chunk_index: int) -> str:
    key = f"{source_path}:{page}:{chunk_index}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()


def extract_pdf_pages(pdf_path: Path, pdf_dir: Path) -> List[PageText]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install pymupdf to parse PDFs.") from exc

    category = infer_category(pdf_path, pdf_dir)
    source_path = str(pdf_path.relative_to(pdf_dir.parent)).replace("\\", "/")
    pages: List[PageText] = []

    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append(
                    PageText(
                        source_path=source_path,
                        pdf_name=pdf_path.name,
                        category=category,
                        page=index,
                        text=text,
                    )
                )
    return pages


def chunk_pages(
    pages: Sequence[PageText],
    *,
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    for page in pages:
        for index, text in enumerate(split_text(page.text, chunk_size=chunk_size, overlap=overlap)):
            chunks.append(
                TextChunk(
                    id=stable_chunk_id(page.source_path, page.page, index),
                    text=text,
                    source_path=page.source_path,
                    pdf_name=page.pdf_name,
                    category=page.category,
                    page=page.page,
                    chunk_index=index,
                )
            )
    return chunks


def load_pdf_chunks(
    pdf_dir: Path,
    *,
    chunk_size: int = 900,
    overlap: int = 150,
) -> tuple[List[TextChunk], dict[str, int]]:
    pdfs = discover_pdfs(pdf_dir)
    all_chunks: List[TextChunk] = []
    pdf_page_counts: dict[str, int] = {}

    for pdf_path in pdfs:
        pages = extract_pdf_pages(pdf_path, pdf_dir)
        pdf_page_counts[str(pdf_path.relative_to(pdf_dir)).replace("\\", "/")] = len(pages)
        all_chunks.extend(chunk_pages(pages, chunk_size=chunk_size, overlap=overlap))

    return all_chunks, pdf_page_counts


class LocalEmbeddingClient:
    def __init__(self, model_name: str, cache_dir: Path | str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install sentence-transformers for local embeddings.") from exc

        self.model_name = model_name
        self.cache_dir = str(cache_dir)
        self.model = SentenceTransformer(model_name, cache_folder=self.cache_dir)

    def embed_documents(self, texts: Sequence[str], *, batch_size: int = 32) -> List[List[float]]:
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        prompt = f"\u4e3a\u8fd9\u4e2a\u53e5\u5b50\u751f\u6210\u8868\u793a\u4ee5\u7528\u4e8e\u68c0\u7d22\u76f8\u5173\u6587\u7ae0\uff1a{query}"
        embedding = self.model.encode([prompt], normalize_embeddings=True)
        return embedding[0].tolist()


def batched(items: Sequence[TextChunk], batch_size: int) -> Iterable[Sequence[TextChunk]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_vector_store(
    *,
    pdf_dir: Path = DEFAULT_PDF_DIR,
    vector_db_dir: Path = DEFAULT_VECTOR_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str | None = None,
    cache_dir: Path | str | None = None,
    chunk_size: int = 900,
    overlap: int = 150,
    batch_size: int = 32,
    reset: bool = False,
) -> dict[str, object]:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install chromadb for local vector storage.") from exc

    pdf_dir = Path(pdf_dir)
    vector_db_dir = Path(vector_db_dir)
    model_name = model_name or os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL)
    cache_dir = cache_dir or os.getenv("LOCAL_EMBEDDING_CACHE_DIR", str(DEFAULT_LOCAL_EMBEDDING_CACHE_DIR))

    chunks, pdf_page_counts = load_pdf_chunks(pdf_dir, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise RuntimeError(f"No text chunks were extracted from {pdf_dir}.")

    client = chromadb.PersistentClient(path=str(vector_db_dir))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": model_name},
    )

    embedder = LocalEmbeddingClient(model_name=model_name, cache_dir=cache_dir)
    for batch in batched(chunks, batch_size):
        texts = [chunk.text for chunk in batch]
        embeddings = embedder.embed_documents(texts, batch_size=batch_size)
        collection.upsert(
            ids=[chunk.id for chunk in batch],
            documents=texts,
            metadatas=[chunk.metadata() for chunk in batch],
            embeddings=embeddings,
        )

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "pdf_dir": str(pdf_dir),
        "vector_db_dir": str(vector_db_dir),
        "collection_name": collection_name,
        "embedding_model": model_name,
        "embedding_cache_dir": str(cache_dir),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "pdf_count": len(pdf_page_counts),
        "page_count": sum(pdf_page_counts.values()),
        "chunk_count": len(chunks),
        "pdf_pages": pdf_page_counts,
    }
    vector_db_dir.mkdir(parents=True, exist_ok=True)
    (vector_db_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def retrieve(
    query: str,
    *,
    vector_db_dir: Path = DEFAULT_VECTOR_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str | None = None,
    cache_dir: Path | str | None = None,
    top_k: int = 5,
) -> list[dict[str, object]]:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install chromadb for retrieval.") from exc

    model_name = model_name or os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL)
    cache_dir = cache_dir or os.getenv("LOCAL_EMBEDDING_CACHE_DIR", str(DEFAULT_LOCAL_EMBEDDING_CACHE_DIR))
    embedder = LocalEmbeddingClient(model_name=model_name, cache_dir=cache_dir)
    query_embedding = embedder.embed_query(query)

    client = chromadb.PersistentClient(path=str(vector_db_dir))
    collection = client.get_collection(collection_name)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    payloads: list[dict[str, object]] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        payloads.append({"text": document, "metadata": metadata, "distance": distance})
    return payloads
