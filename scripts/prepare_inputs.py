"""Download only public scientific inputs; record exact bytes and provenance."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://oruk.ai/connectome-reservoir/v1/'
REVISION = 'cab853a1dbdf4c42c2b3ef2173804746df8825fe'
ORIGINAL_MODEL_SHA = 'f6750dcd1312b8794480130571095ad097ccb93b17fcefe2578f1811bfdbe9b8'


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(Request(url, headers={'User-Agent': 'FlyRead-research/1.0'}), timeout=90) as response:
            content = response.read(30_000_001)
        if len(content) > 30_000_000:
            raise ValueError('Unexpectedly large input: ' + url)
        path.write_bytes(content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f'{path.name}: {path.stat().st_size:,} bytes, sha256 {digest}', flush=True)
    return {'url': url, 'sha256': digest, 'bytes': path.stat().st_size}


def main():
    files = {}
    for name in ('meta.json', 'model.bin', 'anatomy.bin', 'credits.json', 'emotion-means.json'):
        files['oruk/' + name] = download(BASE + name, ROOT / 'data/oruk' / name)
    if files['oruk/model.bin']['sha256'] != ORIGINAL_MODEL_SHA:
        raise ValueError('Circuit bytes differ from the published experiment checksum. Review before training.')
    for split in ('train', 'validation', 'test'):
        name = f'{split}-00000-of-00001.parquet'
        url = f'https://huggingface.co/datasets/dair-ai/emotion/resolve/{REVISION}/split/{name}'
        files['emotion/' + split + '.parquet'] = download(url, ROOT / 'data/emotion' / (split + '.parquet'))
    download(f'https://huggingface.co/datasets/dair-ai/emotion/raw/{REVISION}/README.md', ROOT / 'data/emotion/DATASET_CARD.md')
    (ROOT / 'data/sources.json').write_text(json.dumps({'dataset_revision': REVISION, 'files': files}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
