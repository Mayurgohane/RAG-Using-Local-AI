"""Search the local index with the same embedding model used at ingest time."""

from rag.config.settings import COLLECTION_NAME, EMBEDDING_MODEL, INDEX_DIR, TOP_K
from rag.embeddings.local import LocalEmbedder
from rag.vectorstore.store import VectorStore


class LocalSearch:
    def __init__(self) -> None:
        self.store = VectorStore(INDEX_DIR, COLLECTION_NAME)
        self.embedder = LocalEmbedder(EMBEDDING_MODEL)

    def ready(self) -> bool:
        return self.store.count() > 0

    def close(self) -> None:
        self.store.close()

    def search(self, question: str, k: int = TOP_K) -> list[dict]:
        embedding = self.embedder.embed([question])[0]
        result = self.store.query(embedding, k)
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]

        hits: list[dict] = []
        for text, metadata, distance in zip(documents[0], metadatas[0], distances[0]):
            hits.append(
                {
                    "text": text,
                    "source": (metadata or {}).get("source", "unknown"),
                    "page": (metadata or {}).get("page", "?"),
                    "similarity": round(1 - float(distance), 4),
                }
            )
        return hits
