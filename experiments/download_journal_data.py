"""Download official BEIR SciFact/NFCorpus archives; refuse existing folders."""
import argparse
import io
from pathlib import Path
import urllib.request
import zipfile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['scifact','nfcorpus'],required=True)
    p.add_argument('--destination',type=Path,default=Path('data/processed/journal'))
    args=p.parse_args()
    target=args.destination/args.dataset
    if target.exists(): raise FileExistsError(target)
    url=f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip'
    with urllib.request.urlopen(url,timeout=90) as response: data=response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in ['corpus.jsonl','queries.jsonl','qrels/test.tsv']:
            if f'{args.dataset}/{name}' not in archive.namelist(): raise ValueError('Unexpected archive layout')
        target.mkdir(parents=True,exist_ok=False)
        for name in ['corpus.jsonl','queries.jsonl','qrels/test.tsv']:
            path=target/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(archive.read(f'{args.dataset}/{name}'))
    print(target)


if __name__=='__main__': main()
