# Local RAG — Knowledge Transfer

Handover notes for the local retrieval-augmented generation app in this folder. The app answers questions from PDFs on this machine. It does not call a paid API.

## 1. What it does

1. Read PDFs from `data/documents`.
2. Split each PDF into overlapping text chunks.
3. Turn each chunk into a vector with a local embedding model.
4. Save the vectors in a local Qdrant database.
5. When a user asks a question, find the closest chunks and ask a local language model to answer from those chunks only.

The indexed set today is two papers: `attention_mechanism.pdf` and `lora.pdf`. The last ingestion stored **241 chunks**.

## 2. How to run a demo

Use the project virtual environment. Open PowerShell in `C:\Users\LEGION\Desktop\RAG`.

Stop `main.py` before ingestion. Qdrant local mode allows only one process to hold `data/qdrant`.

```powershell
.\.venv\Scripts\python.exe ingestion.py
.\.venv\Scripts\python.exe main.py
```

In a second terminal, after the API has started:

```powershell
.\.venv\Scripts\chainlit.exe run main.py --host 127.0.0.1 --port 8080
```

| What | Where |
| --- | --- |
| Chat | http://127.0.0.1:8080 |
| API | http://127.0.0.1:8000 |
| API docs | http://127.0.0.1:8000/docs |
| Health | http://127.0.0.1:8000/health |

`python ingestion.py` prints four stages: extract, chunk, embed, save. `python main.py` starts the API. The chat calls `POST /query/stream` on that API.

To index new PDFs, copy them into `data/documents`, stop the API, and run `ingestion.py` again. Ingestion replaces the whole collection.

## 3. Entry files

| File | Role |
| --- | --- |
| `ingestion.py` | The only ingestion script. Extract, chunk, embed, and save. |
| `main.py` | The only application entry. `python main.py` starts FastAPI. `chainlit run main.py` starts the chat. |

Library code lives under `src/rag`.

| Path | Role |
| --- | --- |
| `src/rag/config/settings.py` | Paths, chunk size, model names, ports. |
| `src/rag/embeddings/local.py` | Local MiniLM embeddings. |
| `src/rag/vectorstore/store.py` | Qdrant create, replace, and search. |
| `src/rag/retrieval/search.py` | Embed the question and return the top chunks. |
| `src/rag/generation/prompt.py` | System prompt and the user message built from retrieved text. |
| `src/rag/generation/llm.py` | Qwen2.5-1.5B-Instruct on ONNX Runtime. |
| `src/rag/pipeline/rag.py` | Search, then generate. Also re-runs ingestion from `POST /ingest`. |
| `src/rag/api/app.py` | FastAPI app and lifespan. |
| `src/rag/api/routes/query.py` | `/health`, `/query`, `/query/stream`. |
| `src/rag/api/routes/ingest.py` | `POST /ingest`. |

## 4. Request path

1. Chainlit sends the question to `POST /query/stream`.
2. The question is embedded with the same MiniLM model used at ingestion.
3. Qdrant returns the top 4 chunks by cosine similarity.
4. A greeting such as "hi" gets a short hello and does not call the model.
5. If the best chunk scores below `MIN_SIMILARITY` (0.50), the reply is: "The documents do not contain enough information."
6. Otherwise the retrieved text and the question are sent to Qwen.
7. The model is instructed to answer only from that text.
8. The chat streams the finished answer word by word and can open the source passages beside it.

`POST /query` returns the same answer as one JSON object. The stream uses server-sent events named `sources`, `token`, and `done`.

## 5. Settings that matter

From `src/rag/config/settings.py`:

| Setting | Value | Meaning |
| --- | --- | --- |
| `CHUNK_SIZE` | 800 | Characters per chunk. |
| `CHUNK_OVERLAP` | 150 | Overlap so a sentence is less likely to be cut in half. |
| `TOP_K` | 4 | Passages retrieved per question. |
| `MIN_SIMILARITY` | 0.50 | Weak matches are refused. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Must stay the same for ingestion and search. |
| `LLM_REPO` | `onnx-community/Qwen2.5-1.5B-Instruct` | Generator. |
| `LLM_FILE` | `onnx/model_q4.onnx` | Quantized ONNX weights. |
| `LLM_MAX_TOKENS` | 120 | Maximum new tokens per answer. |
| `API_PORT` | 8000 | API. Do not put Chainlit on this port. |

## 6. Models and where they live

| Piece | What | Where |
| --- | --- | --- |
| Embeddings | `all-MiniLM-L6-v2` through the ONNX helper in Chroma, CPU only | `C:\Users\LEGION\.cache\chroma\onnx_models\all-MiniLM-L6-v2` |
| Generator | Qwen2.5-1.5B-Instruct, 4-bit ONNX | `models/qwen2.5-1.5b-instruct` |
| Index | Qdrant, local files, collection `documents`, cosine distance | `data/qdrant` |

Chroma is a dependency because it supplies the MiniLM ONNX runtime. The vector store is Qdrant, not Chroma. `onnxruntime` is pinned to **1.19.2**. A newer 1.30 build crashes on import on this PC. Do not upgrade it casually.

PyTorch and `llama-cpp` were tried and do not run reliably here (DLL load failure and an access violation in `llama_backend_init`). Stay on ONNX Runtime 1.19.2.

The first question after startup is slow because the 1.5B model loads on CPU. Later questions are faster, but still take several seconds.

## 7. Answer rules

The system prompt is in `src/rag/generation/prompt.py`. It tells the model to:

- answer the question the user actually asked
- use only the retrieved context
- write one or two clear sentences
- say "The documents do not contain enough information." when the context does not contain the answer

There is no list of allowed topics and no stop-word list. A new PDF is handled by ingestion and retrieval, not by editing question text in code.

If the model's words do not overlap the retrieved text at all, the app replaces the answer with the "not enough information" sentence so a fully invented reply is not shown.

## 8. Machine limits to tell the next person

- About 8 GB RAM. Qwen2.5-1.5B quantized fits. A 7B model does not.
- One writer for `data/qdrant`. Never run ingestion while `main.py` is up.
- PDF text is extracted with `pypdf`. Scanned pages with no text layer are skipped.
- Some PDF lines are still noisy (glued words, captions). The model is asked to write clean sentences from that text.
- `data/qdrant` and `models/` are local artifacts. They are not required in git if the next person can re-run ingestion and let the model files download.

## 9. Handover checklist

- [ ] `.\.venv\Scripts\python.exe ingestion.py` finishes and prints the chunk count.
- [ ] `.\.venv\Scripts\python.exe main.py` reaches "Application startup complete".
- [ ] http://127.0.0.1:8000/health shows `status: ok` and a non-zero chunk count.
- [ ] Chat at http://127.0.0.1:8080 answers a question that is in the PDFs.
- [ ] A question outside the PDFs returns "The documents do not contain enough information."
- [ ] A greeting returns a short hello and does not invent a fact.
