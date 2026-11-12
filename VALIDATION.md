# Local verification

Reliability update (12 November 2026): regression tests now cover negative-query
evaluation, document removal/reindexing, PDF page routing and real PDF text
extraction. Model retrieval is simulated for deterministic reindex tests. Actual
local answer-model correctness and broad private-document accuracy remain unverified.

Verified on this workspace with Python 3.14:

- Dependency installation completed; `pip check` reported no broken requirements.
- Six unit tests passed, including a controlled hybrid/reranking pipeline test.
- Real embedding model indexed the two supplied sample documents successfully.
- Real BM25, vector, hybrid, and cross-encoder reranking evaluation completed on all
  four supplied questions. This is a smoke test, not a meaningful quality benchmark.
- Command-line evidence search returned the expected E104 support passage and source ID.
- Streamlit AppTest startup passed with no application exceptions.

Written-answer generation was not tested against a real model: local Ollama had no
models installed. Unknown citation-ID rejection was tested using a simulated response.
PDF extraction and large/private document collections were not exercised end to end.

This workspace already has `.venv` dependencies, the sample `index.json`, and cached
retrieval models. The ZIP contains source and instructions, excluding those installed
dependencies, model weights, and generated index; follow README setup after extracting.
