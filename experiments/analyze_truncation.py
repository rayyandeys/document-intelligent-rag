"""Post-hoc relevant-document length strata; no embedding or generation calls."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from src.evaluation.journal_metrics import interval, metrics


def stratum(lengths, limit):
    if not lengths:
        raise ValueError('No positively judged documents')
    flags = [n > limit for n in lengths]
    return 'all_over_limit' if all(flags) else ('all_within_limit' if not any(flags) else 'mixed')


def analyze(run, output):
    from transformers import AutoTokenizer
    manifest = json.loads((run/'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError(f'Incomplete run: {run}')
    dataset = run.name.split('_dense_')[0]
    if dataset not in ('scifact', 'nfcorpus'):
        raise ValueError('Unknown dataset')
    data = Path('data/processed/journal')/dataset
    for name, expected in manifest['data_hashes'].items():
        path = data/('qrels/test.tsv' if name == 'test.tsv' else name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Dataset mismatch: {path}')
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2',
                    revision=manifest['model_revision'], local_files_only=True)
    limit = manifest['max_seq_length']
    lengths = {}
    for line in (data/'corpus.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        text = f"{row.get('title', '')}. {row['text']}"
        lengths[str(row['_id'])] = len(tokenizer.encode(text, truncation=False))
    recorded = manifest['truncation']['minilm_document']
    if len(lengths) != recorded['units'] or sum(n > limit for n in lengths.values()) != recorded['over_limit']:
        raise ValueError('Tokenizer length counts disagree with original run')
    qrels = {}
    with (data/'qrels/test.tsv').open(encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            qrels.setdefault(row['query-id'], {})[row['corpus-id']] = int(row['score'])
    qrels = {q:r for q,r in qrels.items() if any(g > 0 for g in r.values())}
    methods = {}
    for name in ('bm25', 'minilm_document', 'minilm_400', 'minilm_600'):
        rows = list(map(json.loads, (run/f'{name}.jsonl').read_text().splitlines()))
        if len(rows) != len(qrels) or len({r['query_id'] for r in rows}) != len(qrels):
            raise ValueError('Query coverage mismatch')
        methods[name] = {}
        for row in rows:
            q = row['query_id']
            computed = metrics([d for d,s in row['ranking']], qrels[q])
            if any(abs(v-row['metrics'][m]) > 1e-12 for m,v in computed.items()):
                raise ValueError('Saved metric mismatch')
            methods[name][q] = computed
    groups = {g:[] for g in ('all_within_limit', 'mixed', 'all_over_limit')}
    query_rows = []
    for q in sorted(qrels):
        relevant = [d for d,g in qrels[q].items() if g > 0]
        group = stratum([lengths[d] for d in relevant], limit)
        groups[group].append(q)
        query_rows.append({'query_id':q, 'stratum':group, 'relevant_documents':len(relevant),
                           'relevant_over_limit':sum(lengths[d] > limit for d in relevant)})
    report = {'dataset':dataset, 'source_run':str(run), 'model_revision':manifest['model_revision'],
              'limit':limit, 'groups':{}, 'interpretation':'Post-hoc exploratory analysis. Strata use all positively judged relevant documents, including tokenizer special tokens. Not a causal control: competing documents can still be truncated. Small strata have unstable intervals; no multiple-comparison correction.'}
    for group, ids in groups.items():
        section = {'queries':len(ids), 'small_stratum':len(ids)<30, 'paired_differences':{}}
        if ids:
            for treatment in ('minilm_400','minilm_600'):
                for base in ('minilm_document','bm25'):
                    section['paired_differences'][f'{treatment}_minus_{base}'] = {
                        metric:interval([methods[treatment][q][metric]-methods[base][q][metric] for q in ids])
                        for metric in ('ndcg_at_10','mrr_at_10','hit_at_1')}
        report['groups'][group] = section
    output.mkdir(parents=True, exist_ok=False)
    (output/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'document_token_lengths.json').write_text(json.dumps(lengths),encoding='utf-8')
    with (output/'query_strata.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(query_rows[0]));w.writeheader();w.writerows(query_rows)
    print(dataset, {g:len(ids) for g,ids in groups.items()}, flush=True)
    print('Saved:',output,flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs',nargs='+',type=Path,help='Optional explicit dense run folders')
    args = parser.parse_args()
    runs = args.runs or []
    if not runs:
        for dataset in ('scifact','nfcorpus'):
            candidates = list(Path('experiments/journal').glob(dataset+'_dense_*'))
            complete = [p for p in candidates if (p/'manifest.json').exists() and json.loads((p/'manifest.json').read_text()).get('status')=='complete']
            if len(complete)!=1:
                parser.error(f'Found {len(complete)} completed {dataset} runs. Use --runs with the intended folder paths.')
            runs.extend(complete)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = Path('experiments/journal')/('truncation_analysis_'+stamp)
    for run in runs:
        analyze(run,destination/run.name.split('_dense_')[0])
    print('COMPLETE. Zip and return:',destination)


if __name__=='__main__': main()
