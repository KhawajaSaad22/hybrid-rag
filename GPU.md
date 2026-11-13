# GPU execution

For my RTX 4060, use the optional Linux GPU configuration:

```sh
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
docker compose -f compose.yaml -f compose.gpu.yaml exec app python gpu_check.py
```

This builds a distinct local CUDA-capable image instead of pulling the published
CPU image. It requests NVIDIA GPU passthrough and fails startup if CUDA computation
is unavailable. The base Dockerfile remains CPU-compatible for ordinary CI/hosting.
Windows requires the Docker/WSL2 NVIDIA GPU setup and a compatible host driver.

The override installs Linux PyTorch from PyPI rather than the CPU-only wheel index.
That can download several GB of CUDA dependencies. The selected wheel/runtime must
match the host driver; supplying a GPU alone does not change a CPU-only PyTorch build.
Versions remain compatibility ranges, so record/lock the validated deployment stack.

## GPU workload

Embedding and cross-encoder reranking accept --device auto|cpu|cuda on ingest/ask. INSPECTION_DEVICE also controls application selection. Explicit CUDA fails when unavailable. Parsing, BM25, stored-vector search and interface work remain on CPU.

## Validation status — 13 November 2026

Real all-MiniLM-L6-v2 embeddings and ms-marco-MiniLM-L6-v2 cross-encoder reranking both ran on cuda:0 using two generated text documents.

Verified on my RTX 4060 in a local Linux container with PyTorch 2.15.0+cu132 / CUDA 13.2 using cached weights. These are real model execution checks on synthetic fixtures, not accuracy or speed benchmarks. The complete application GPU image was not independently rebuilt in this patch. CPU CI does not validate CUDA.

To repeat in a CUDA-capable environment containing dependencies and cached weights:

```sh
python gpu_validate.py --device cuda --output artifacts/gpu-validation.json
```

The output path must be new. Validation defaults to offline model loading; --allow-download explicitly permits downloads. No real inspection dataset is needed.
