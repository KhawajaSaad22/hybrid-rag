"""Exercise real cached embedding/reranking models on an explicitly selected device."""
import argparse
import json
import os
from pathlib import Path
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--allow-download',action='store_true')
    args=p.parse_args();out=Path(args.output)
    if out.exists():p.error('Choose a new report path')
    os.environ['HF_HUB_OFFLINE']='0' if args.allow_download else '1'
    import numpy as np
    import torch
    from rag import build_index,load_index,search,embedder,reranker,EMBEDDER,RERANKER
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);docs=root/'docs';docs.mkdir()
        (docs/'support.md').write_text('Error E104 means an expired access token. Renew the token to reconnect.')
        (docs/'inspection.md').write_text('Inspect bridge surfaces for cracks and corrosion. Record image evidence.')
        build_index(docs,root/'index.json',args.device)
        index=load_index(root/'index.json')
        results=search('What does E104 mean?',index,device=args.device)
        encoder=embedder(EMBEDDER,args.device);cross=reranker(args.device)
        assert encoder.device.type==args.device and cross.model.device.type==args.device
        assert np.isfinite(index['vectors']).all() and results
        assert all(np.isfinite(r['rerank_score']) for r in results)
        report=dict(device=args.device,embedding_device=str(encoder.device),reranking_device=str(cross.model.device),
                    torch=torch.__version__,cuda_runtime=torch.version.cuda,models=[EMBEDDER,RERANKER],
                    gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
                    passages=len(index['chunks']),results=[r['source'] for r in results],
                    note='Real model execution on synthetic text; not an accuracy or speed benchmark.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
