from __future__ import annotations

import argparse
import json
from pathlib import Path
import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dhruv2015/poketwo-datset-3')
    ap.add_argument('--version', default='2')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    endpoint = f'https://www.kaggle.com/api/v1/datasets/download/{args.dataset}?datasetVersionNumber={args.version}'
    offset = args.output.stat().st_size if args.output.exists() else 0
    r = requests.get(endpoint, allow_redirects=False, timeout=60)
    r.raise_for_status()
    url = r.headers['location']
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(60, 300)) as resp:
        resp.raise_for_status()
        mode = 'ab' if offset else 'wb'
        total = offset + int(resp.headers.get('content-length', '0'))
        written = offset
        with args.output.open(mode) as f:
            for chunk in resp.iter_content(chunk_size=16 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if written // (256 * 1024 * 1024) != (written - len(chunk)) // (256 * 1024 * 1024):
                    print(json.dumps({'written_bytes': written, 'expected_bytes': total}), flush=True)
    print(json.dumps({'complete_bytes': args.output.stat().st_size, 'expected_bytes': total}), flush=True)


if __name__ == '__main__':
    main()
