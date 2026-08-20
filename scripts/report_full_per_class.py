from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--evaluation',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    import torch
    d=json.loads(a.evaluation.read_text()); ck=torch.load(a.checkpoint,map_location='cpu'); labels=ck['labels']; idx={x:i for i,x in enumerate(labels)}
    rows=[r for r in csv.DictReader(a.manifest.open(encoding='utf-8')) if r['split_original']=='test']
    y=[idx[r['label']] for r in rows]; pred=d['test']['predictions']; cm=d['test']['confusion_matrix']
    supports=[0]*len(labels)
    for i in y: supports[i]+=1
    out=[]
    for i,name in enumerate(labels):
        row=cm[i]; support=supports[i]; correct=row[i] if i<len(row) else 0
        out.append({'index':i,'label':name,'test_support':support,'correct':correct,'per_class_accuracy':(correct/support if support else None),'independent_real_tested':bool(support)})
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2),encoding='utf-8')
    with a.output.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    print({'classes':len(labels),'tested_classes':sum(x['independent_real_tested'] for x in out),'untested_classes':sum(not x['independent_real_tested'] for x in out),'output':str(a.output)})
if __name__=='__main__': main()
