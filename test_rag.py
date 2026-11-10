import unittest
from unittest.mock import patch
import rag


class RetrievalTests(unittest.TestCase):
    def test_chunk_coverage_and_validation(self):
        text = " ".join(str(i) for i in range(23))
        chunks = list(rag.chunk_text(text, 10, 2))
        self.assertEqual(set(text.split()), set(" ".join(chunks).split()))
        self.assertEqual(chunks[-1].split()[-1], "22")
        with self.assertRaises(ValueError):
            list(rag.chunk_text(text, 10, 10))

    def test_exact_identifier_and_empty_query(self):
        scores = rag.bm25("E104", ["Vacation policy", "E104 means expired token"])
        self.assertGreater(scores[1], scores[0])
        self.assertEqual(rag.bm25("anything", []), [])
        with self.assertRaises(ValueError):
            rag.search(" ", {"chunks": []}, mode="bm25")

    def test_fusion_consensus(self):
        self.assertEqual(rag.fuse([[0, 1], [2, 1]])[0], 1)

    def test_sample_search(self):
        chunks = list(rag.read_documents(rag.ROOT / "sample_docs"))
        results = rag.search("E104 expired token", {"chunks": chunks}, mode="bm25")
        self.assertEqual(results[0]["source"], "support.md")
        self.assertIn("[S1]", rag.answer("E104", results))
        self.assertEqual(rag.search("zzzznonexistent", {"chunks": chunks}, mode="bm25"), [])

    def test_model_citation_validation(self):
        import io
        results = [{"citation": "S1", "source": "x.md", "page": None, "text": "Evidence"}]
        response = io.BytesIO(b'{"message":{"content":"Invented [S99]"}}')
        with patch("urllib.request.urlopen", return_value=response):
            with self.assertRaises(ValueError):
                rag.answer("question", results, model="test")

    def test_hybrid_and_reranking_pipeline(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy not installed")
        from types import SimpleNamespace
        chunks = [
            {"id": "a", "source": "a.md", "page": None, "text": "E104 expired token"},
            {"id": "b", "source": "b.md", "page": None, "text": "Replace credentials"}]
        index = {"chunks": chunks, "vectors": [[1., 0.], [0., 1.]],
                 "embedding_model": "test"}
        fake_embedder = SimpleNamespace(encode_query=lambda *a, **k: np.array([1., 0.]))
        fake_reranker = SimpleNamespace(predict=lambda pairs: [0., 10.])
        with patch("rag.embedder", return_value=fake_embedder):
            self.assertEqual(rag.search("E104", index, mode="hybrid")[0]["id"], "a")
            with patch("rag.reranker", return_value=fake_reranker):
                results = rag.search("E104", index)
                self.assertEqual(results[0]["id"], "b")
                self.assertEqual(results[0]["citation"], "S1")


if __name__ == "__main__":
    unittest.main()
