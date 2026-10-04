"""Generic prompts for answering a user question from retrieved passages."""

import re
import unicodedata

NOT_ENOUGH = "The documents do not contain enough information."

SYSTEM_PROMPT = (
    "You are a question-answering assistant for a private document collection. "
    "Answer only from the context in the user message. "
    "Answer the exact question the user asked, whether it asks what something is, "
    "how it works, why it is used, where it applies, or how two things differ. "
    "Write a clear and complete answer in your own words, in one or two sentences. "
    "Do not copy broken text, page numbers, or figure captions. "
    "Do not add facts that are not in the context. "
    f"If the context does not answer the question, reply exactly: {NOT_ENOUGH}"
)

_SMALLTALK = re.compile(
    r"^(?:hi+|hey+|hello+|yo+|hola|thanks|thank you|bye+|goodbye|"
    r"good\s+(?:morning|afternoon|evening)|how are you|who are you)"
    r"[!.?,\s]*$",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    cleaned = re.sub(r"\s+", " ", normalized)
    return cleaned.strip()


def smalltalk_reply(question: str) -> str:
    if not _SMALLTALK.match(question.strip()):
        return ""
    return "Hello. Ask a question about the documents that have been indexed."


def relevant_context(hits: list[dict], limit: int = 1400) -> str:
    parts: list[str] = []
    total = 0
    for hit in hits:
        text = _clean(str(hit.get("text") or ""))
        if not text:
            continue
        if parts and total + len(text) > limit:
            room = limit - total
            if room > 80:
                parts.append(text[:room].rsplit(" ", 1)[0])
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)


def build_user_prompt(question: str, hits: list[dict]) -> tuple[str, str]:
    context = relevant_context(hits)
    prompt = f"Context:\n{context}\n\nQuestion: {question.strip()}"
    return prompt, context


def _content_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{4,}", text.lower())


def choose_answer(generated: str, context: str) -> str:
    polished = polish_answer(generated)
    if not polished or polished.lower().startswith(NOT_ENOUGH.lower()):
        return NOT_ENOUGH
    words = _content_words(polished)
    context_text = context.lower()
    if words and not any(word in context_text for word in words):
        return NOT_ENOUGH
    return polished


def polish_answer(text: str) -> str:
    cleaned = " ".join(text.split()).strip(" -:")
    if not cleaned:
        return ""
    cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned
