from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch

def metrics(y,p,s,t):
    a=s>=t
    if not a.any(): return {'accepted_count':0,'count':len(y),'unknown_rate':1.0,'accepted_top1':0.0,'precision_macro':0.0,'recall_macro':0.0,'f1_macro':0.0}
    pr,re,f,_=precision_recall_fscore_support(y[a],p[a],labels=np.arange(int(p.max())+1),average='macro',zero_division=0)
    return {'accepted_count':int(a.sum()),'count':int(len(y)),'unknown_rate':float(1-a.mean()),'accepted_top1':float(np.mean(p[a]==y[a])),'precision_macro':float(pr),'recall_macro':float(re),'f1_macro':float(f)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--evaluation',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--threshold',type=float,default=.30); a=ap.parse_args()
    d=json.loads(a.evaluation.read_text()); ck=torch.load(a.checkpoint,map_location='cpu'); labels=ck['labels']; idx={x:i for i,x in enumerate(labels)}
    rows=[r for r in csv.DictReader(a.manifest.open(encoding='utf-8')) if r['split_original']=='test']; y0=np.asarray([idx[r['label']] for r in rows],int)
    p=np.asarray(d['hard_case']['predictions'],int); s=np.asarray(d['hard_case']['scores'],float); y=np.repeat(y0,8)
    assert len(y)==len(p)==len(s)
    out={'threshold':a.threshold,'overall':metrics(y,p,s,a.threshold),'by_transform':{}}
    names=['brightness_low','brightness_high','contrast_low','blur','jpeg_like','crop','dark_ui','light_ui']
    for i,n in enumerate(names): out['by_transform'][n]=metrics(y0,p[i::8],s[i::8],a.threshold)
    a.output.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
