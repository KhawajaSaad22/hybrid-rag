"""Compare retrieval modes against question/source labels (not answer quality)."""
import argparse
import json
import time
from rag import ROOT, load_index, search

def evaluate(questions, index, mode="bm25", search_fn=search):
    """Empty relevant_sources labels a negative query; no results means abstention."""
    if not questions: raise ValueError("Provide nonempty evaluation questions")
    hits=reciprocal=recall=answerable=negative=false_positives=0
    started=time.perf_counter()
    for q in questions:
        if not isinstance(q.get('relevant_sources'),list) or not q.get('question','').strip():
            raise ValueError("Each question needs text and a relevant_sources list (empty for negatives)")
        results=search_fn(q['question'],index,mode=mode)
        expected=set(q['relevant_sources'])
        if not expected:
            negative+=1; false_positives+=bool(results); continue
        answerable+=1
        retrieved=[r['source'] for r in results]
        ranks=[i for i,source in enumerate(retrieved,1) if source in expected]
        hits+=bool(ranks); reciprocal+=1/ranks[0] if ranks else 0
        recall+=len(set(retrieved)&expected)/len(expected)
    return dict(mode=mode,answerable_queries=answerable,negative_queries=negative,
                source_hit_at_5=hits/answerable if answerable else None,
                source_recall_at_5=recall/answerable if answerable else None,
                mrr_at_5=reciprocal/answerable if answerable else None,
                negative_false_positive_rate=false_positives/negative if negative else None,
                negative_abstention_rate=1-false_positives/negative if negative else None,
                seconds_per_query=(time.perf_counter()-started)/len(questions))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions',default=str(ROOT/'questions.json'))
    parser.add_argument('--mode',choices=['all','bm25','vector','hybrid','hybrid-rerank'],default='all')
    parser.add_argument('--output',help='Optional JSON evaluation report')
    args=parser.parse_args()
    from pathlib import Path
    questions=json.loads(Path(args.questions).read_text(encoding='utf-8'))
    index=load_index()
    modes=['bm25','vector','hybrid','hybrid-rerank'] if args.mode=='all' else [args.mode]
    reports=[evaluate(questions,index,mode) for mode in modes]
    print(json.dumps(reports,indent=2))
    if args.output: Path(args.output).write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print('Source retrieval only, not answer correctness. Empty output counts as abstention. Timings include cold loads.')

if __name__ == "__main__":
    main()
