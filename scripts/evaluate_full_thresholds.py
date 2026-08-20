from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--evaluation',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    d=json.loads(a.evaluation.read_text()); ck=json.loads('{}')
    import torch
    checkpoint=torch.load(a.checkpoint,map_location='cpu'); labels=checkpoint['labels']; idx={x:i for i,x in enumerate(labels)}
    rows=[r for r in csv.DictReader(a.manifest.open(encoding='utf-8')) if r['split_original']=='test']
    y=np.asarray([idx[r['label']] for r in rows],dtype=int)
    pred=np.asarray(d['test']['predictions'],dtype=int); scores=np.asarray(d['test']['scores'],dtype=float)
    assert len(y)==len(pred)==len(scores), (len(y),len(pred),len(scores))
    out=[]
    for t in [0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.95]:
        accepted=scores>=t
        if accepted.any():
            p,r,f,_=precision_recall_fscore_support(y[accepted],pred[accepted],labels=np.arange(len(labels)),average='macro',zero_division=0)
            acc=float(np.mean(pred[accepted]==y[accepted]))
        else: p=r=f=acc=0.0
        out.append({'threshold':t,'accepted_count':int(accepted.sum()),'count':int(len(y)),'accepted_rate':float(accepted.mean()),'unknown_rate':float(1-accepted.mean()),'accepted_top1':acc,'precision_macro':float(p),'recall_macro':float(r),'f1_macro':float(f),'overall_top1':float(np.mean(pred==y))})
    result={'classes':len(labels),'test_count':len(y),'thresholds':out}
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
