"""Paths and local-model settings shared by ingestion and the demo."""

from pathlib import Path


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Project root not found. Expected pyproject.toml.")


ROOT = project_root()
DOCUMENTS_DIR = ROOT / "data" / "documents"
INDEX_DIR = ROOT / "data" / "qdrant"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
MIN_SIMILARITY = 0.50
MODELS_DIR = ROOT / "models"
LLM_REPO = "onnx-community/Qwen2.5-1.5B-Instruct"
LLM_FILE = "onnx/model_q4.onnx"
LLM_MAX_TOKENS = 120
API_HOST = "127.0.0.1"
API_PORT = 8000
