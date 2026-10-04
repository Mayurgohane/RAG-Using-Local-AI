"""Extract local PDFs, chunk them, embed them, and store them in Qdrant."""

import sys
import unicodedata
from pathlib import Path

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag.config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    INDEX_DIR,
)
from rag.embeddings.local import LocalEmbedder
from rag.vectorstore.store import VectorStore


class IngestionError(Exception):
    """The document folder cannot be indexed."""


def list_pdfs(documents_dir: Path) -> list[Path]:
    if not documents_dir.is_dir():
        raise IngestionError(f"Documents folder not found: {documents_dir}")
    return sorted(path for path in documents_dir.glob("*.pdf") if path.is_file())


def extract_pdf(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    pages: list[dict] = []
    for number, page in enumerate(reader.pages, start=1):
        text = unicodedata.normalize("NFKC", page.extract_text() or "").strip()
        if not text:
            continue
        pages.append({"source": path.name, "page": number, "text": text})
    return pages


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    collapsed = " ".join(text.split())
    if not collapsed:
        return []
    if len(collapsed) <= size:
        return [collapsed]

    chunks: list[str] = []
    start = 0
    while start < len(collapsed):
        end = min(start + size, len(collapsed))
        if end < len(collapsed):
            split_at = collapsed.rfind(" ", start, end)
            if split_at > start + size // 2:
                end = split_at
        piece = collapsed[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(collapsed):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(pages: list[dict], size: int, overlap: int) -> list[dict]:
    chunks: list[dict] = []
    for page in pages:
        for text in chunk_text(page["text"], size, overlap):
            chunks.append(
                {
                    "source": page["source"],
                    "page": page["page"],
                    "text": text,
                }
            )
    return chunks


def run() -> dict:
    pdfs = list_pdfs(DOCUMENTS_DIR)
    if not pdfs:
        raise IngestionError(f"No PDF files found in {DOCUMENTS_DIR}")

    print(f"[1/4] Extracting text from {len(pdfs)} PDF(s)")
    pages: list[dict] = []
    for path in pdfs:
        extracted = extract_pdf(path)
        print(f"  {path.name}: {len(extracted)} page(s) with text")
        pages.extend(extracted)

    if not pages:
        raise IngestionError("No extractable text found. These PDFs may be scanned images.")

    print("[2/4] Chunking extracted text")
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  {len(chunks)} chunk(s)")

    print(f"[3/4] Generating embeddings with local model {EMBEDDING_MODEL}")
    embedder = LocalEmbedder(EMBEDDING_MODEL)
    vectors = embedder.embed([chunk["text"] for chunk in chunks])

    print(f"[4/4] Saving vectors to Qdrant at {INDEX_DIR}")
    ids = [f"chunk-{index}" for index in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {"source": chunk["source"], "page": chunk["page"]} for chunk in chunks
    ]
    store = VectorStore(INDEX_DIR, COLLECTION_NAME)
    try:
        store.replace(ids, documents, vectors, metadatas)
        chunks_indexed = store.count()
        print(f"Done. Indexed {chunks_indexed} chunk(s) in Qdrant. Next: python main.py")
    finally:
        store.close()
    return {"files": len(pdfs), "pages": len(pages), "chunks": chunks_indexed}


if __name__ == "__main__":
    try:
        run()
    except IngestionError as exc:
        raise SystemExit(str(exc)) from exc
