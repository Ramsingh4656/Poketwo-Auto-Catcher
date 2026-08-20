from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train import build_model, eval_transform, read_manifest


def topk_acc(y, prob, k):
    top = np.argsort(-prob, axis=1)[:, :k]
    return float(np.mean([true in row for true, row in zip(y, top)])) if len(y) else 0.0


def load_rows(path, split=None):
    rows = read_manifest(path)
    return [x for x in rows if split is None or x['split_original'] == split]


class EvalDataset(Dataset):
    def __init__(self, rows, labels, transform=None, data_root=Path('.')):
        self.rows=rows; self.labels=labels; self.data_root=data_root; self.transform=transform or eval_transform()
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r=self.rows[i]
        image_path = Path(r['path']) if Path(r['path']).is_absolute() else self.data_root / r['path']
        with Image.open(image_path) as im: im=im.convert('RGB')
        return self.transform(im), self.labels[r['label']], str(image_path)


def predict(model, loader):
    model.eval(); ys=[]; probs=[]; paths=[]
    with torch.no_grad():
        for x,y,p in loader:
            probs.append(torch.softmax(model(x),1).numpy()); ys.extend(y.numpy().tolist()); paths.extend(p)
    return np.array(ys), np.concatenate(probs), paths


def threshold_for_val(y, prob):
    scores=prob.max(1); pred=prob.argmax(1); best={'threshold':0.0,'f1':-1.0}
    for t in np.linspace(0.05,0.99,95):
        accepted=scores>=t
        if not accepted.any(): continue
        pp,rr,ff,_=precision_recall_fscore_support(y[accepted],pred[accepted],labels=np.arange(prob.shape[1]),average='macro',zero_division=0)
        # optimize accepted macro-F1 while exposing coverage explicitly
        objective=float(ff) * float(accepted.mean())
        if objective>best['f1']: best={'threshold':float(t),'f1':objective,'accepted_rate':float(accepted.mean()),'precision':float(pp),'recall':float(rr),'macro_f1':float(ff)}
    return best


def metric_dict(y, prob, threshold=None):
    pred=prob.argmax(1); scores=prob.max(1); accepted=np.ones(len(y), dtype=bool) if threshold is None else scores>=threshold
    p,r,f,_=precision_recall_fscore_support(y[accepted],pred[accepted],labels=np.arange(prob.shape[1]),average='macro',zero_division=0) if accepted.any() else (0,0,0,None)
    cm=confusion_matrix(y,pred,labels=np.arange(prob.shape[1]))
    return {'count':int(len(y)),'accepted_count':int(accepted.sum()),'unknown_rate':float(1-accepted.mean()),'top1':topk_acc(y,prob,1),'top3':topk_acc(y,prob,3),'top5':topk_acc(y,prob,5),'precision_macro':float(p),'recall_macro':float(r),'f1_macro':float(f),'confusion_matrix':cm.tolist(),'predictions':pred.tolist(),'scores':scores.tolist(),'accepted':accepted.tolist()}


def hard_images(im):
    base=im.convert('RGB')
    variants=[]
    variants.append(('brightness_low', ImageEnhance.Brightness(base).enhance(0.65)))
    variants.append(('brightness_high', ImageEnhance.Brightness(base).enhance(1.35)))
    variants.append(('contrast_low', ImageEnhance.Contrast(base).enhance(0.65)))
    variants.append(('blur', base.filter(ImageFilter.GaussianBlur(1.2))))
    variants.append(('jpeg_like', base.resize((112,112),Image.Resampling.BILINEAR).resize((224,224),Image.Resampling.BICUBIC)))
    variants.append(('crop', ImageOps.fit(base,(224,224),method=Image.Resampling.BICUBIC,centering=(0.48,0.5),bleed=0.08)))
    variants.append(('dark_ui', Image.blend(base, Image.new('RGB',base.size,(25,25,30)),0.28)))
    variants.append(('light_ui', Image.blend(base, Image.new('RGB',base.size,(235,235,235)),0.12)))
    return variants


def eval_hard(model, rows, labels, data_root=Path('.')):
    tf=eval_transform(); ys=[]; probs=[]; names=[]
    with torch.no_grad():
        for row in rows:
            image_path = Path(row['path']) if Path(row['path']).is_absolute() else data_root / row['path']
            with Image.open(image_path) as im: im=im.convert('RGB')
            for name,variant in hard_images(im):
                probs.append(torch.softmax(model(tf(variant).unsqueeze(0)),1)[0].numpy()); ys.append(labels[row['label']]); names.append(name)
    return np.array(ys), np.array(probs), names


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--data-root',type=Path,default=Path('.')); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--batch-size',type=int,default=64); args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True); proc=psutil.Process(os.getpid()); rss_before=proc.memory_info().rss; ck=torch.load(args.checkpoint,map_location='cpu'); labels=ck['labels']; label_to_index={x:i for i,x in enumerate(labels)}; model=build_model(ck['arch'],len(labels)); model.load_state_dict(ck['state_dict']); model.eval(); rss_after_model=proc.memory_info().rss
    val=load_rows(args.manifest,'val'); test=load_rows(args.manifest,'test'); loader=DataLoader(EvalDataset(test,label_to_index,data_root=args.data_root),batch_size=args.batch_size,shuffle=False)
    yv,pv,_=predict(model,DataLoader(EvalDataset(val,label_to_index,data_root=args.data_root),batch_size=args.batch_size,shuffle=False)); yt,pt,paths=predict(model,loader)
    cal=threshold_for_val(yv,pv); hard_y,hard_p,hard_names=eval_hard(model,test,label_to_index,args.data_root)
    proc=psutil.Process(os.getpid()); before=proc.memory_info().rss; x=torch.zeros(1,3,224,224); times=[]
    with torch.no_grad():
        for _ in range(5): model(x)
        for _ in range(50):
            t=time.perf_counter(); model(x); times.append((time.perf_counter()-t)*1000)
    after=proc.memory_info().rss; rss_peak=max(rss_before,rss_after_model,after)
    result={'arch':ck['arch'],'checkpoint':str(args.checkpoint),'classes':len(labels),'labels':labels,'validation_threshold':cal,'val':metric_dict(yv,pv),'test':metric_dict(yt,pt,cal['threshold']),'hard_case':metric_dict(hard_y,hard_p,cal['threshold']),'hard_case_by_transform':{},'cpu_inference_ms_mean':float(np.mean(times)),'cpu_inference_ms_p95':float(np.percentile(times,95)),'ram_rss_before_model_mb':float(rss_before/1024/1024),'ram_rss_after_model_mb':float(rss_after_model/1024/1024),'ram_rss_after_eval_mb':float(after/1024/1024),'ram_rss_peak_mb':float(rss_peak/1024/1024),'ram_rss_model_delta_mb':float((rss_after_model-rss_before)/1024/1024),'model_size_mb':float(args.checkpoint.stat().st_size/1024/1024)}
    for name in sorted(set(hard_names)):
        mask=np.array([x==name for x in hard_names]); result['hard_case_by_transform'][name]=metric_dict(hard_y[mask],hard_p[mask],cal['threshold'])
    (args.output_dir/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (args.output_dir/'confusion_matrix.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['true\\pred']+labels); w.writerows([[labels[i]]+row for i,row in enumerate(result['test']['confusion_matrix'])])
    print(json.dumps({k:result[k] for k in ('arch','validation_threshold','val','test','hard_case','cpu_inference_ms_mean','cpu_inference_ms_p95','ram_rss_peak_mb','ram_rss_model_delta_mb','model_size_mb')},indent=2))

if __name__=='__main__': main()
