from __future__ import annotations

import argparse
from pathlib import Path
import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dhruv2015/poketwo-datset-3')
    ap.add_argument('--version', default='2')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    url = f'https://www.kaggle.com/api/v1/datasets/download/{args.dataset}?datasetVersionNumber={args.version}'
    r = requests.get(url, allow_redirects=False, timeout=30)
    result = {'endpoint_status': r.status_code, 'endpoint_content_type': r.headers.get('content-type')}
    location = r.headers.get('location')
    if location:
        result['signed_url_present'] = True
        h = requests.head(location, timeout=30)
        result['archive_status'] = h.status_code
        result['archive_content_length'] = h.headers.get('content-length')
        result['archive_content_type'] = h.headers.get('content-type')
        result['archive_accept_ranges'] = h.headers.get('accept-ranges')
    else:
        result['signed_url_present'] = False
        result['response_prefix'] = r.text[:500]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import json
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
