"""Retrieve passages and generate a grounded answer."""

import sys
import threading

from rag.config.settings import MIN_SIMILARITY, ROOT

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion import IngestionError, run as run_ingestion
from rag.generation.llm import LocalLLM
from rag.generation.prompt import (
    NOT_ENOUGH,
    build_user_prompt,
    choose_answer,
    smalltalk_reply,
)
from rag.retrieval.search import LocalSearch


class RagService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.search = LocalSearch()
        self.llm = LocalLLM()

    def ready(self) -> bool:
        return self.search.ready()

    def chunk_count(self) -> int:
        return self.search.store.count()

    def _compose(self, question: str, hits: list[dict]) -> tuple[str, list[dict]]:
        greeting = smalltalk_reply(question)
        if greeting:
            return greeting, []
        if not hits or float(hits[0].get("similarity") or 0) < MIN_SIMILARITY:
            return NOT_ENOUGH, []
        prompt, context = build_user_prompt(question, hits)
        if not context:
            return NOT_ENOUGH, []
        return choose_answer(self.llm.generate(prompt), context), hits

    def answer(self, question: str, top_k: int) -> dict:
        with self._lock:
            hits = self.search.search(question, k=top_k)
            text, sources = self._compose(question, hits)
            return {"question": question, "answer": text, "sources": sources}

    def stream(self, question: str, top_k: int):
        self._lock.acquire()
        try:
            hits = self.search.search(question, k=top_k)
            answer, sources = self._compose(question, hits)
            if sources:
                yield "sources", {"sources": sources}
            words = answer.split(" ")
            for index, word in enumerate(words):
                yield "token", {"text": word if index == 0 else f" {word}"}
            yield "done", {
                "question": question,
                "answer": answer,
                "sources": sources,
            }
        finally:
            self._lock.release()

    def ingest(self) -> dict:
        with self._lock:
            self.search.close()
            try:
                return run_ingestion()
            except IngestionError:
                raise
            finally:
                self.search = LocalSearch()

    def close(self) -> None:
        self.search.close()
