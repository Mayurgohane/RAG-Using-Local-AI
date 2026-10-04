"""Local RAG application: API and chat UI.

Start the API with `python main.py`.
Start the chat with `chainlit run main.py --host 127.0.0.1 --port 8080`.
"""

import json
import sys
from pathlib import Path

import chainlit as cl
import httpx
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag.config.settings import API_HOST, API_PORT, TOP_K

API_URL = f"http://{API_HOST}:{API_PORT}/query/stream"


async def _events(question: str):
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            API_URL,
            json={"question": question, "top_k": TOP_K},
        ) as response:
            response.raise_for_status()
            event_name = "message"
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    yield event_name, json.loads(line.split(":", 1)[1].strip())


def _source_elements(sources: list[dict]) -> list[cl.Text]:
    elements: list[cl.Text] = []
    for index, hit in enumerate(sources, start=1):
        elements.append(
            cl.Text(
                name=f"{index}. {hit['source']} p.{hit['page']} ({hit['similarity']})",
                content=hit["text"],
                display="side",
            )
        )
    return elements


@cl.on_chat_start
async def start() -> None:
    await cl.Message(
        content=(
            "Ask a question about the ingested PDFs. "
            "The answer streams in as the local model writes it, "
            "and the source passages open beside the reply."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    answer = cl.Message(content="")
    sources: list[dict] = []
    try:
        async for event_name, data in _events(message.content):
            if event_name == "sources":
                sources = data.get("sources") or []
            elif event_name == "token":
                await answer.stream_token(data.get("text") or "")
    except httpx.ConnectError:
        await cl.Message(
            content="The RAG API is not running. Start it with python main.py, then ask again."
        ).send()
        return
    except httpx.HTTPStatusError as exc:
        await cl.Message(content=f"The API returned {exc.response.status_code}.").send()
        return

    answer.elements = _source_elements(sources)
    if answer.streaming:
        await answer.update()
    else:
        answer.content = "The documents do not contain enough information."
        await answer.send()


def main() -> None:
    uvicorn.run(
        "rag.api.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
