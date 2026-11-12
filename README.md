# Hybrid RAG with reranking

Evaluation update: `evaluate.py` accepts `relevant_sources: []` for unanswerable
questions and reports negative-query false-positive/abstention rates separately.
Use `--output report.json` to save metrics. Empty retrieval output counts as
abstention; dense retrieval is not automatically calibrated. These metrics do not
score answer correctness. Regression tests cover PDF page references, actual PDF
text extraction and removal/update of indexed documents after reindexing.

A local learning project: real BM25 + dense semantic search, reciprocal rank fusion,
cross-encoder reranking, source excerpts, optional Ollama answers, CLI, and Streamlit UI.
No paid API key is required. First use downloads embedding/reranking model weights.

## 1. Set up (Windows PowerShell)

Use Python 3.10 or newer (3.11/3.12 are conservative choices for ML compatibility).
Run from this project folder. A local `.venv` may already exist; install requirements
even if it does. Commands avoid PowerShell activation-policy changes.

```powershell
git clone https://github.com/KhawajaSaad22/hybrid-rag.git
cd hybrid-rag
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For macOS/Linux, use `python3 -m venv .venv` and `.venv/bin/python` instead.

## 2. Index the supplied example documents

```powershell
.\.venv\Scripts\python.exe rag.py ingest sample_docs
```

This creates `index.json`, containing passage text, metadata, and normalized vectors.
To use my own documents, replace `sample_docs` with a folder path in quotes.
TXT and Markdown must be UTF-8. PDFs must contain selectable text; OCR is not included.
Each PDF page is chunked separately so citations preserve its actual page number.
Rebuild the index after editing, adding, or removing documents. Ingestion replaces it.

## 3. Ask questions

```powershell
.\.venv\Scripts\python.exe rag.py ask "How do I fix E104?"
.\.venv\Scripts\python.exe rag.py ask "How much vacation do employees get?" --mode hybrid
.\.venv\Scripts\python.exe rag.py ask "E104" --mode bm25
```

Default mode is `hybrid-rerank`. Without `--model`, the result is explicitly labeled
as evidence extracts, not a synthesized answer. BM25-only search needs no downloaded
models once an index exists. Retrieval uses up to 30 candidates from each retriever,
RRF (k=60), then reranks up to 30 fused candidates and returns five passages.

## 4. Open the interface

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address printed by Streamlit. Build/rebuild my document collection
from the sidebar, choose a retrieval mode, ask a question, and expand the source cards.
The interface accepts a local folder, rather than uploading files.

## 5. Optional: generate written answers using Ollama

Install Ollama from https://ollama.com and start it. Pull a model that fits my machine,
for example `ollama pull llama3.2:3b`. Then:

```powershell
.\.venv\Scripts\python.exe rag.py ask "How do I fix E104?" --model llama3.2:3b
```

Or enter the installed model name in the UI. The API defaults to localhost:11434.
Evidence and questions are sent to that local Ollama endpoint. This project does not
call a hosted generation API. Third-party model downloads still require internet access.

## 6. Validate and compare

```powershell
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe evaluate.py
```

Evaluation compares BM25, vector, hybrid, and hybrid + reranking. Edit `questions.json`
with my own questions and relevant relative source filenames. It reports source-level
hit@5, recall@5, and MRR@5; it does not measure claim correctness or passage-level recall.
The tiny example corpus is a smoke test, not evidence of retrieval superiority.
Separate held-out questions from questions used to tune settings.

## Files

- `rag.py`: extraction, chunking, indexing, BM25, fusion, reranking, answering, CLI.
- `app.py`: local Streamlit interface.
- `evaluate.py`: retrieval comparison.
- `test_rag.py`: offline logic and citation-validation tests.
- `sample_docs/`: fictional sample material.

## Practical limitations and next steps

- Dense search always retrieves neighbors, including for unanswerable questions. No
  calibrated relevance threshold is included; reranker scores are not probabilities.
- Prompts request abstention and citations, but cannot guarantee grounding. Citation ID
  validation rejects unknown IDs; it does not prove that cited text supports each claim.
- Retrieved text is treated as untrusted data in the generation prompt. Prompt injection
  is not fully solved by prompting; this app gives the model no tools to execute actions.
- Uses in-memory exhaustive vector search and a JSON index for simplicity. Intended for
  small local collections; move to a vector database for large collections.
- English starter models, word-window chunking, no OCR, no table-aware extraction, and
  no authentication or multi-user document permissions. Run as a trusted local app.
- Avoid concurrent index rebuilds. Private document text is stored in `index.json`.
- Model initialization/download time can dominate the first query. Later queries in the
  UI reuse cached models. CLI processes load them again each time.

Next improve extraction, add passage-level relevance labels, measure held-out retrieval
and citation support, then tune chunking and candidate counts. Add permissions before
deploying a shared service.

## Implementation references

- https://sbert.net/docs/sentence_transformer/usage/usage.html
- https://sbert.net/docs/package_reference/cross_encoder/model.html
- https://qdrant.tech/documentation/search/hybrid-queries/ (fusion background; Qdrant is not required)
- https://docs.ollama.com/api

## Docker image

This project has its own CPU image: `ghcr.io/khawajasaad22/hybrid-rag:latest`.
Images are stored in GitHub Container Registry; the repository contains the Dockerfile,
Compose configuration, and build/publish workflow. The workflow tests the image and
checks the running app before publishing `latest` and a full commit-SHA tag. A failed
build does not publish. Use the Actions tab to confirm the first successful publication.

With Docker Engine/Desktop and Compose installed, from this repository:

```sh
docker compose up --build -d
```

Open http://localhost:8501. To use the registry image after it is published:

```sh
docker compose pull app
docker compose up -d --no-build app
```

If the registry package is private, authenticate with `docker login ghcr.io` using an
account/token allowed to read the package. Repository visibility alone does not ensure
anonymous image access. No custom registry secret is needed by the publishing workflow:
it uses the repository's built-in GitHub token with package-write permission.

The container runs as UID 10001. Named volumes preserve inputs, generated artifacts,
and downloaded model weights. Weights and private datasets are not baked into the image.
First model use requires internet access. These are Linux/amd64 CPU images; GPU/CUDA
acceleration is not included. Initial builds can take several minutes.

Use `docker compose exec app python ...` for the documented CLI commands. Inputs are
container paths, not Windows host paths. Copy real files with `docker compose cp`:

```sh
docker compose cp ./my-input app:/app/data/my-input
docker compose logs -f app
docker compose down
```

`down` keeps named volumes; `down -v` deletes them, including my generated work.
The app port is published on localhost only. Container health means Streamlit is running;
you still need to prepare data and fit/index models before using the app workflow.

```sh
docker compose exec app python rag.py ingest sample_docs
docker compose exec app python rag.py ask "How do I fix E104?"
```

The index persists at `/app/artifacts/index.json` via `RAG_INDEX_PATH`.

### Optional Ollama service

```sh
docker compose --profile generation up -d
docker compose exec ollama ollama pull YOUR_MODEL_NAME
```

Choose a text model for RAG or a vision-capable model for video verification, then enter
its installed name in the UI. Ollama is accessible only inside the Compose network;
no host port is published. `OLLAMA_BASE_URL` selects the service endpoint. This optional
service also runs on CPU. Its model quality and memory needs depend on my selection.
