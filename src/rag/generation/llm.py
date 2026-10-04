"""Local Qwen2.5-1.5B-Instruct generator running on ONNX Runtime."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from rag.config.settings import LLM_FILE, LLM_MAX_TOKENS, LLM_REPO, MODELS_DIR
from rag.generation.prompt import SYSTEM_PROMPT

_FILES = (
    "config.json",
    "tokenizer.json",
    LLM_FILE,
)


def _model_dir() -> Path:
    destination = MODELS_DIR / "qwen2.5-1.5b-instruct"
    missing = [name for name in _FILES if not (destination / name).is_file()]
    if not missing:
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    for name in missing:
        hf_hub_download(repo_id=LLM_REPO, filename=name, local_dir=destination)
    return destination


def _numpy_dtype(type_name: str):
    if "float16" in type_name:
        return np.float16
    return np.float32


class LocalLLM:
    def __init__(self) -> None:
        root = _model_dir()
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        self.model_name = "Qwen2.5-1.5B-Instruct"
        eos = config.get("eos_token_id", 151645)
        self._eos = {int(item) for item in eos} if isinstance(eos, list) else {int(eos)}
        self._eos.add(151643)
        self._tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 4
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(root / LLM_FILE),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._output_names = [item.name for item in self._session.get_outputs()]
        past_inputs = [
            item
            for item in self._session.get_inputs()
            if item.name.startswith("past_key_values")
        ]
        self._past_dtype = (
            _numpy_dtype(past_inputs[0].type) if past_inputs else np.float32
        )

    def _chat(self, user: str) -> str:
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _empty_past(self) -> dict[str, np.ndarray]:
        past: dict[str, np.ndarray] = {}
        for item in self._session.get_inputs():
            if not item.name.startswith("past_key_values"):
                continue
            shape: list[int] = []
            symbolic = 0
            for dim in item.shape:
                if isinstance(dim, int) and dim > 0:
                    shape.append(dim)
                else:
                    symbolic += 1
                    shape.append(1 if symbolic == 1 else 0)
            past[item.name] = np.zeros(shape, dtype=self._past_dtype)
        return past

    def _encode(self, prompt: str) -> np.ndarray:
        ids = self._tokenizer.encode(self._chat(prompt)).ids[-480:]
        return np.array([ids], dtype=np.int64)

    def stream(self, prompt: str):
        input_ids = self._encode(prompt)
        past = self._empty_past()
        attention = np.ones_like(input_ids)
        position = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
        produced: list[int] = []
        previous = ""
        for _ in range(LLM_MAX_TOKENS):
            feeds = {
                "input_ids": input_ids,
                "attention_mask": attention,
                "position_ids": position,
            }
            feeds.update(past)
            named = dict(zip(self._output_names, self._session.run(None, feeds)))
            next_id = int(np.argmax(named["logits"][0, -1]))
            if next_id in self._eos:
                break
            produced.append(next_id)
            if len(produced) >= 8 and produced[-4:] == produced[-8:-4]:
                break
            past = {
                name.replace("present", "past_key_values"): value
                for name, value in named.items()
                if name != "logits"
            }
            past_len = next(iter(past.values())).shape[2]
            input_ids = np.array([[next_id]], dtype=np.int64)
            attention = np.ones((1, past_len + 1), dtype=np.int64)
            position = np.array([[past_len]], dtype=np.int64)
            text = self._tokenizer.decode(produced)
            for marker in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
                text = text.replace(marker, "")
            delta = text[len(previous) :] if text.startswith(previous) else text
            previous = text
            if delta:
                yield delta

    def generate(self, prompt: str) -> str:
        return "".join(self.stream(prompt)).strip()
