from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 20260820

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def read_manifest(path: Path):
    with path.open(encoding='utf-8', newline='') as f: return list(csv.DictReader(f))


def discord_aug():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.82, 1.0), ratio=(0.92, 1.08)),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.92, 1.08), fill=(35, 35, 35)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12)], p=0.65),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
        transforms.RandomHorizontalFlip(p=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class ManifestDataset(Dataset):
    def __init__(self, rows, labels, train=False, data_root=Path('.')):
        self.rows = rows; self.labels = labels; self.data_root = data_root
        self.transform = discord_aug() if train else eval_transform()
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = Path(row['path']) if Path(row['path']).is_absolute() else self.data_root / row['path']
        with Image.open(image_path) as im:
            if 'A' in im.getbands():
                rgba = im.convert('RGBA')
                bg = Image.new('RGBA', rgba.size, (54, 54, 54, 255))
                im = Image.alpha_composite(bg, rgba).convert('RGB')
            else:
                im = im.convert('RGB')
        return self.transform(im), self.labels[row['label']], str(image_path)


def build_model(arch: str, nclasses: int):
    if arch == 'simple_cnn':
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, nclasses),
        )
    if arch == 'mobilenet_v3_small':
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, nclasses)
        return m
    if arch == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, nclasses)
        return m
    raise ValueError(arch)


def split_rows(rows, split): return [r for r in rows if r['split_original'] == split]


def evaluate(model, loader, device, nclasses):
    model.eval(); ys=[]; ps=[]; probs=[]; paths=[]
    with torch.no_grad():
        for x,y,p in loader:
            out = model(x.to(device)); pr = torch.softmax(out, 1).cpu()
            ys.extend(y.tolist()); ps.extend(pr.argmax(1).tolist()); probs.append(pr); paths.extend(p)
    prob = torch.cat(probs) if probs else torch.empty((0,nclasses))
    return np.array(ys), np.array(ps), prob.numpy(), paths


def topk_acc(y, prob, k):
    if len(y) == 0: return 0.0
    top = np.argsort(-prob, axis=1)[:, :k]
    return float(np.mean([true in row for true,row in zip(y, top)]))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest', type=Path, required=True); ap.add_argument('--data-root', type=Path, default=Path('.')); ap.add_argument('--arch', choices=['simple_cnn','mobilenet_v3_small','resnet18'], required=True); ap.add_argument('--output-dir', type=Path, required=True); ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--batch-size', type=int, default=32); ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--workers', type=int, default=0); ap.add_argument('--freeze-backbone', action='store_true'); args=ap.parse_args()
    seed_everything(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows=read_manifest(args.manifest)
    label_names=sorted({r['label'] for r in rows}); labels={n:i for i,n in enumerate(label_names)}
    (args.output_dir/'labels.json').write_text(json.dumps({'index_to_label': label_names, 'label_to_index': labels}, indent=2), encoding='utf-8')
    train_rows=split_rows(rows,'train'); val_rows=split_rows(rows,'val'); test_rows=split_rows(rows,'test')
    train_loader=DataLoader(ManifestDataset(train_rows,labels,True,args.data_root), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader=DataLoader(ManifestDataset(val_rows,labels,False,args.data_root), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader=DataLoader(ManifestDataset(test_rows,labels,False,args.data_root), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device=torch.device('cpu'); model=build_model(args.arch,len(labels)).to(device)
    if args.freeze_backbone and args.arch != 'simple_cnn':
        for name,p in model.named_parameters():
            if ('classifier' not in name and not (args.arch=='resnet18' and name.startswith('fc'))): p.requires_grad=False
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    criterion=nn.CrossEntropyLoss(label_smoothing=0.05)
    best=-1; history=[]; start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train(); running=0.0; n=0
        for x,y,_ in train_loader:
            opt.zero_grad(); out=model(x.to(device)); loss=criterion(out,y.to(device)); loss.backward(); opt.step(); running += loss.item()*len(y); n += len(y)
        yv,pv,probv,_=evaluate(model,val_loader,device,len(labels)); score=topk_acc(yv,probv,1)
        row={'epoch':epoch,'train_loss':running/max(1,n),'val_top1':score,'val_top3':topk_acc(yv,probv,3),'val_top5':topk_acc(yv,probv,5)}; history.append(row); print(json.dumps(row))
        if score > best:
            best=score; torch.save({'state_dict':model.state_dict(),'arch':args.arch,'labels':label_names,'img_size':224,'seed':SEED,'epoch':epoch}, args.output_dir/'best.pt')
    model.load_state_dict(torch.load(args.output_dir/'best.pt', map_location='cpu')['state_dict'])
    yt,pt,probt,paths=evaluate(model,test_loader,device,len(labels))
    result={'arch':args.arch,'epochs':args.epochs,'batch_size':args.batch_size,'lr':args.lr,'train_images':len(train_rows),'val_images':len(val_rows),'test_images':len(test_rows),'classes':len(labels),'best_val_top1':best,'test_top1':topk_acc(yt,probt,1),'test_top3':topk_acc(yt,probt,3),'test_top5':topk_acc(yt,probt,5),'history':history,'wall_seconds':time.perf_counter()-start}
    (args.output_dir/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
