"""Local embedding model. Weights run on this machine through ONNX Runtime."""

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2


class LocalEmbedder:
    def __init__(self, model_name: str) -> None:
        if model_name != ONNXMiniLM_L6_V2.MODEL_NAME:
            raise ValueError(
                f"Unsupported local model {model_name!r}. "
                f"Use {ONNXMiniLM_L6_V2.MODEL_NAME}."
            )
        self.model_name = model_name
        self._model = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])

    def embed(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        vectors: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(vector.tolist() for vector in self._model(batch))
            if total > 1:
                print(f"  embedded {min(start + batch_size, total)}/{total}", flush=True)
        return vectors
