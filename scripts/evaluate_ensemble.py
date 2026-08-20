from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from train import build_model, read_manifest
from evaluate import EvalDataset, metric_dict, load_rows, eval_hard

def predict(model, loader):
    model.eval(); ys=[]; probs=[]
    with torch.no_grad():
        for x,y,_ in loader:
            ys.extend(y.numpy().tolist()); probs.append(torch.softmax(model(x),1).numpy())
    return np.array(ys), np.concatenate(probs)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--mobile-checkpoint',type=Path,required=True); ap.add_argument('--resnet-checkpoint',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    mck=torch.load(args.mobile_checkpoint,map_location='cpu'); rck=torch.load(args.resnet_checkpoint,map_location='cpu'); assert mck['labels']==rck['labels']
    labels=mck['labels']; lab={x:i for i,x in enumerate(labels)}; mm=build_model(mck['arch'],len(labels)); rm=build_model(rck['arch'],len(labels)); mm.load_state_dict(mck['state_dict']); rm.load_state_dict(rck['state_dict'])
    val=load_rows(args.manifest,'val'); test=load_rows(args.manifest,'test'); vl=DataLoader(EvalDataset(val,lab),batch_size=64); tl=DataLoader(EvalDataset(test,lab),batch_size=64)
    yv,pmv=predict(mm,vl); _,prv=predict(rm,vl); yt,pmt=predict(mm,tl); _,prt=predict(rm,tl)
    for alpha in [0.0,0.25,0.5,0.75,1.0]:
        pv=alpha*pmv+(1-alpha)*prv; pt=alpha*pmt+(1-alpha)*prt
        result={'alpha_mobile':alpha,'val':metric_dict(yv,pv),'test':metric_dict(yt,pt)}
        (args.output_dir/f'alpha_{alpha:.2f}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(alpha, {k:result['test'][k] for k in ('top1','top3','top5','precision_macro','recall_macro','f1_macro')})

if __name__=='__main__': main()
