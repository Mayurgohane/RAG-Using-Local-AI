"""Persist and query embeddings in a local Qdrant database."""

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class VectorStore:
    def __init__(self, index_dir: Path, collection_name: str) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = QdrantClient(path=str(index_dir))

    def replace(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        batch_size: int = 64,
    ) -> None:
        if not embeddings:
            raise ValueError("No embeddings to store.")
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=len(embeddings[0]),
                distance=Distance.COSINE,
            ),
        )
        points = [
            PointStruct(
                id=index,
                vector=vector,
                payload={
                    "id": chunk_id,
                    "text": text,
                    "source": metadata["source"],
                    "page": metadata["page"],
                },
            )
            for index, (chunk_id, text, vector, metadata) in enumerate(
                zip(ids, documents, embeddings, metadatas)
            )
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
            )

    def query(self, embedding: list[float], k: int) -> dict:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=k,
        )
        documents: list[str] = []
        metadatas: list[dict] = []
        distances: list[float] = []
        for point in response.points:
            payload = point.payload or {}
            documents.append(str(payload.get("text", "")))
            metadatas.append(
                {
                    "source": payload.get("source", "unknown"),
                    "page": payload.get("page", "?"),
                }
            )
            distances.append(float(1.0 - point.score))
        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def count(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        return int(self.client.count(self.collection_name).count)

    def close(self) -> None:
        self.client.close()
