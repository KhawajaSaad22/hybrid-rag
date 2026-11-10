"""Compare retrieval modes against question/source labels (not answer quality)."""
import argparse
import json
import time
from rag import ROOT, load_index, search

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(ROOT / "questions.json"))
    parser.add_argument("--mode", choices=["all", "bm25", "vector", "hybrid", "hybrid-rerank"], default="all")
    args = parser.parse_args()
    questions = json.loads(open(args.questions, encoding="utf-8").read())
    if not questions or any(not q.get("relevant_sources") for q in questions):
        parser.error("Provide nonempty questions with relevant_sources labels.")
    index = load_index()
    modes = ["bm25", "vector", "hybrid", "hybrid-rerank"] if args.mode == "all" else [args.mode]
    for mode in modes:
        hits, reciprocal, total_recall = 0, 0., 0.
        started = time.perf_counter()
        for q in questions:
            results = search(q["question"], index, mode=mode)
            expected = set(q["relevant_sources"])
            retrieved = [r["source"] for r in results]
            ranks = [i for i, source in enumerate(retrieved, 1) if source in expected]
            hits += bool(ranks)
            reciprocal += 1 / ranks[0] if ranks else 0
            total_recall += len(set(retrieved) & expected) / len(expected)
        n = len(questions)
        print(f"{mode}: source hit@5={hits/n:.3f}, source recall@5={total_recall/n:.3f}, "
              f"MRR@5={reciprocal/n:.3f}, seconds/query={(time.perf_counter()-started)/n:.2f}")
    print("Source-level labels only; timings include cold model loads. Demo labels are not a benchmark.")

if __name__ == "__main__":
    main()
