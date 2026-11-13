import unittest
from unittest.mock import patch
import rag


class RetrievalTests(unittest.TestCase):
    def test_explicit_device_validation(self):
        with patch('torch.cuda.is_available',return_value=False):
            self.assertEqual(rag.resolve_device('auto'),'cpu')
            self.assertEqual(rag.resolve_device('cpu'),'cpu')
            with self.assertRaisesRegex(ValueError,'CUDA'):rag.resolve_device('cuda')
            with self.assertRaises(ValueError):rag.resolve_device('invalid')

    def test_negative_query_evaluation(self):
        from evaluate import evaluate
        def retrieve(question,index,mode):
            return [] if question=='unknown' else [{'source':'a.md'}]
        report=evaluate([dict(question='known',relevant_sources=['a.md']),
                         dict(question='unknown',relevant_sources=[]),
                         dict(question='misleading',relevant_sources=[])],{},search_fn=retrieve)
        self.assertEqual(report['source_hit_at_5'],1)
        self.assertEqual(report['negative_false_positive_rate'],.5)
        self.assertEqual(report['negative_abstention_rate'],.5)
        self.assertIsNone(evaluate([dict(question='unknown',relevant_sources=[])],{},search_fn=retrieve)['mrr_at_5'])

    def test_ingestion_refresh_removes_deleted_documents(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); docs=root/'docs'; docs.mkdir(); index=root/'index.json'
            (docs/'first.md').write_text('First document')
            (docs/'removed.md').write_text('Second document')
            fake=SimpleNamespace(encode_document=lambda texts,**kw:np.ones((len(texts),2)))
            with patch('rag.embedder',return_value=fake):
                rag.build_index(docs,index)
                self.assertEqual(len(rag.load_index(index)['chunks']),2)
                (docs/'removed.md').unlink()
                (docs/'first.md').write_text('Updated first document')
                rag.build_index(docs,index)
            data=rag.load_index(index)
            self.assertEqual(len(data['chunks']),1)
            self.assertEqual(data['chunks'][0]['text'],'Updated first document')

    def test_pdf_ingestion_preserves_page_reference(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        pages=[SimpleNamespace(extract_text=lambda:'First page evidence'),
               SimpleNamespace(extract_text=lambda:'Second page evidence')]
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'paper.pdf').write_bytes(b'fixture')
            with patch('pypdf.PdfReader',return_value=SimpleNamespace(pages=pages)):
                chunks=list(rag.read_documents(folder))
            self.assertEqual([c['page'] for c in chunks],[1,2])
            self.assertEqual(chunks[1]['source'],'paper.pdf')

    def test_real_pdf_text_extraction(self):
        import tempfile
        from pathlib import Path
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject,NameObject,DecodedStreamObject
        with tempfile.TemporaryDirectory() as folder:
            writer=PdfWriter(); page=writer.add_blank_page(300,300)
            font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),
                NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
            page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):font})})
            stream=DecodedStreamObject(); stream.set_data(b'BT /F1 12 Tf 20 200 Td (Inspection evidence E104) Tj ET')
            page[NameObject('/Contents')]=writer._add_object(stream)
            with (Path(folder)/'text.pdf').open('wb') as output: writer.write(output)
            chunks=list(rag.read_documents(folder))
            self.assertIn('Inspection evidence E104',chunks[0]['text'])
            self.assertEqual(chunks[0]['page'],1)

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
