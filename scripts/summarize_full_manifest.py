from __future__ import annotations
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('manifest',type=Path); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    rows=list(csv.DictReader(a.manifest.open(encoding='utf-8')))
    splits=Counter(r['split_original'] for r in rows); sources=Counter(r['source_type'] for r in rows)
    source_by_split=Counter((r['source_type'], r['split_original']) for r in rows)
    by_class=defaultdict(Counter)
    for r in rows: by_class[r['label']][r['source_type']]+=1
    real_classes=sorted(k for k,v in by_class.items() if v['kaggle_real_render']>0)
    ref_only=sorted(k for k,v in by_class.items() if v['kaggle_real_render']==0)

    out={'rows':len(rows),'splits':dict(splits),'source_types':dict(sources),'source_by_split':{f'{k[0]}:{k[1]}':v for k,v in source_by_split.items()},'classes':len(by_class),'real_render_classes':len(real_classes),'reference_only_classes':len(ref_only),'real_render_labels':real_classes,'reference_only_labels':ref_only}
    a.output.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in out.items() if k not in ('real_render_labels','reference_only_labels')},indent=2))
if __name__=='__main__': main()
