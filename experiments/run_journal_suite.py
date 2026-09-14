"""Run the complete new evaluation locally using existing research dependencies."""
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    from huggingface_hub import model_info
    import sentence_transformers  # Fail before running if dependencies are unavailable.
    revision=model_info('sentence-transformers/all-MiniLM-L6-v2').sha
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for dataset in ['scifact','nfcorpus']:
        data=Path('data/processed/journal')/dataset
        if not data.exists():
            subprocess.run([sys.executable,'-m','experiments.download_journal_data','--dataset',dataset],check=True)
        subprocess.run([sys.executable,'-m','experiments.run_journal_evaluation','--data',str(data),
                        '--output',f'experiments/journal/{dataset}_dense_{stamp}',
                        '--dense','--revision',revision],check=True)


if __name__=='__main__': main()
