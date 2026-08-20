from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

KAGGLE_URL = 'https://www.kaggle.com/api/v1/datasets/download/spreadsheets600/poketwo-spawn-images-v1'
OFFICIAL_URL = 'https://github.com/poketwo/data.git'


def main() -> None:
    ap = argparse.ArgumentParser(description='Download the documented Pokétwo sources used by the reproducible pipeline.')
    ap.add_argument('--output-dir', type=Path, default=Path('data/raw'))
    ap.add_argument('--skip-official', action='store_true')
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / 'poketwo-spawn-images-v1.zip'
    extracted = args.output_dir / 'kaggle' / 'extracted'
    if not extracted.exists():
        print(f'Downloading {KAGGLE_URL}')
        urllib.request.urlretrieve(KAGGLE_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f'Corrupt archive member: {bad}')
            zf.extractall(extracted)
    if not args.skip_official:
        official = args.output_dir / 'poketwo-data'
        if not official.exists():
            subprocess.run(['git', 'clone', '--depth', '1', OFFICIAL_URL, str(official)], check=True)
    print(f'Captured source: {extracted}')
    if not args.skip_official:
        print(f'Official source: {args.output_dir / "poketwo-data"}')


if __name__ == '__main__':
    main()
