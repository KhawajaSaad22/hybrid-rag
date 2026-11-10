"""Launch with: python -m streamlit run app.py"""
from pathlib import Path
import streamlit as st
from rag import ROOT, DEFAULT_INDEX, build_index, load_index, search, answer

st.set_page_config(page_title="Hybrid document search", layout="wide")
st.title("Search your documents")
st.caption("Keyword + semantic retrieval, cross-encoder reranking, and source evidence.")
with st.sidebar:
    st.header("Documents")
    st.caption("This app is for trusted local use. Indexing replaces the current collection.")
    folder = st.text_input("Local document folder", str(ROOT / "sample_docs"))
    if st.button("Build / rebuild index"):
        try:
            with st.spinner("Reading documents and creating embeddings…"):
                count = build_index(Path(folder))
            st.success(f"Indexed {count} passages.")
        except Exception as exc:
            st.error(str(exc))
    mode = st.selectbox("Retrieval", ["hybrid-rerank", "hybrid", "vector", "bm25"])
    model = st.text_input("Ollama model (optional)", placeholder="Installed model name")
    st.caption("Leave blank to see evidence extracts without an answer-generation model.")
    top_k = st.slider("Evidence passages", 1, 10, 5)

with st.form("question"):
    query = st.text_input("Your question")
    submit = st.form_submit_button("Search")
if submit:
    try:
        if not DEFAULT_INDEX.exists():
            st.info("Build the document index using the sidebar first.")
        else:
            with st.spinner("Retrieving evidence…"):
                results = search(query, load_index(), mode, top_k=top_k)
            with st.spinner("Preparing response…"):
                response = answer(query, results, model)
            st.markdown(response)
            st.subheader("Sources")
            for r in results:
                label = f"[{r['citation']}] {r['source']}"
                if r["page"]:
                    label += f" — page {r['page']}"
                with st.expander(label):
                    st.text(r["text"])
                    st.caption(f"Passage: {r['id']} | rerank score: {r['rerank_score']}")
    except Exception as exc:
        st.error(f"Could not complete the request: {exc}")
