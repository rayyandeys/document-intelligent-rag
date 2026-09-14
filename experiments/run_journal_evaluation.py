"""New, isolated BEIR-format benchmark runs; never overwrites an output folder."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from src.evaluation.journal_metrics import metrics, interval
from src.retrieval.bm25 import BM25


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True, help='BEIR folder with corpus.jsonl, queries.jsonl, qrels/test.tsv')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dense', action='store_true', help='Add MiniLM whole-document, 400 and 600 character conditions')
    p.add_argument('--revision', help='Required immutable model commit for dense runs')
    args = p.parse_args()
    if args.dense and (not args.revision or len(args.revision)!=40 or any(c not in '0123456789abcdef' for c in args.revision)):
        p.error('--dense requires a 40-character model commit via --revision')
    paths = [args.data/'corpus.jsonl', args.data/'queries.jsonl', args.data/'qrels/test.tsv']
    corpus = {str(r['_id']): f"{r.get('title','')}. {r['text']}" for r in map(json.loads, paths[0].read_text().splitlines())}
    queries = {str(r['_id']):r['text'] for r in map(json.loads, paths[1].read_text().splitlines())}
    qrels = {}
    with paths[2].open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            qrels.setdefault(str(row['query-id']), {})[str(row['corpus-id'])] = int(row['score'])
    qrels = {q:r for q,r in qrels.items() if any(g>0 for g in r.values())}
    if not qrels: raise ValueError('No evaluated queries')
    for q,r in qrels.items():
        if q not in queries or not set(r).issubset(corpus): raise ValueError('Missing query or judged document')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'created_at':datetime.now(timezone.utc).isoformat(), 'python':platform.python_version(),
                'data_hashes':{str(x.name):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
                'queries':len(qrels), 'documents':len(corpus), 'model_revision':args.revision,
                'bm25':{'k1':1.2,'b':.75,'tokens':'lowercase Unicode regex word tokens; unique query terms; no stemming/stopwords'},
                'dense_ranking':'exact max score per document; differs from historical top-50 chunk deduplication',
                'bootstrap':'10000 query-resampling percentile samples; seed 42; exploratory comparisons, no multiple-testing adjustment',
                'status':'running'}
    manifest['packages'] = {d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    manifest['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    manifest['git_dirty'] = bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    def save_manifest(): (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    save_manifest()
    rows = {}
    def record(name, retrieve):
        rows[name] = {}
        with (args.output/f'{name}.jsonl').open('x') as f:
            for q in sorted(qrels):
                ranking = retrieve(queries[q])
                score = metrics([d for d,s in ranking], qrels[q])
                rows[name][q] = score
                f.write(json.dumps({'query_id':q,'ranking':ranking,'metrics':score})+'\n')
        print(name, {m:round(float(np.mean([r[m] for r in rows[name].values()])),4) for m in next(iter(rows[name].values()))}, flush=True)
    bm25 = BM25(corpus)
    record('bm25', bm25.retrieve)
    if args.dense:
        from sentence_transformers import SentenceTransformer
        from src.ingestion.smart_chunker import smart_chunk_pages
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', revision=args.revision, device='cpu')
        manifest['max_seq_length'] = model.max_seq_length
        query_vectors = dict(zip(sorted(qrels), model.encode([queries[q] for q in sorted(qrels)], normalize_embeddings=True)))
        query_lookup = {queries[q]:query_vectors[q] for q in sorted(qrels)}
        manifest['truncation'] = {}
        for budget in (0,400,600):
            ids, texts = [], []
            for d in sorted(corpus):
                chunks = [{'text':corpus[d]}] if not budget else smart_chunk_pages([{'page':0,'text':corpus[d],'source':d}],budget,1)
                for c in chunks: ids.append(d); texts.append(c['text'])
            lengths = [len(model.tokenizer.encode(t, truncation=False)) for t in texts]
            name = 'minilm_document' if not budget else f'minilm_{budget}'
            manifest['truncation'][name] = {'units':len(texts),'over_limit':sum(n>model.max_seq_length for n in lengths)}
            vectors = model.encode(texts,batch_size=32,normalize_embeddings=True,show_progress_bar=True)
            def retrieve(text):
                scores = vectors @ query_lookup[text]
                best = {}
                for d,s in zip(ids,scores): best[d] = max(best.get(d,-float('inf')), float(s))
                return sorted(best.items(),key=lambda x:(-x[1],x[0]))[:10]
            record(name,retrieve)
            save_manifest()
    summary = {}
    for name,per_query in rows.items():
        summary[name] = {m:interval([per_query[q][m] for q in sorted(qrels)]) for m in next(iter(per_query.values()))}
    comparisons = {}
    pairs = [(n,'bm25') for n in rows if n!='bm25']
    if 'minilm_document' in rows: pairs += [(n,'minilm_document') for n in rows if n.startswith('minilm_') and n!='minilm_document']
    for a,b in pairs:
        comparisons[f'{a}_minus_{b}'] = {m:interval([rows[a][q][m]-rows[b][q][m] for q in sorted(qrels)]) for m in rows[a][next(iter(qrels))]}
    (args.output/'summary.json').write_text(json.dumps({'metrics':summary,'paired_differences':comparisons},indent=2))
    manifest['status']='complete'; save_manifest()


if __name__=='__main__': main()
