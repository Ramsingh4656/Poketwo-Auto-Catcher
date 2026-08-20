from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train import build_model


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    ck=torch.load(args.checkpoint,map_location='cpu')
    model=build_model(ck['arch'],len(ck['labels'])); model.load_state_dict(ck['state_dict']); model.eval()
    out=args.output_dir/'pokemon_detector.onnx'
    dummy=torch.zeros(1,3,224,224)
    torch.onnx.export(model,dummy,str(out),input_names=['images'],output_names=['logits'],opset_version=17,dynamic_axes={'images':{0:'batch'},'logits':{0:'batch'}},do_constant_folding=True)
    metadata={'arch':ck['arch'],'labels':ck['labels'],'input_size':[224,224],'input_layout':'NCHW','normalization':{'mean':[0.485,0.456,0.406],'std':[0.229,0.224,0.225]},'output':'softmax(logits)','checkpoint_epoch':ck.get('epoch'),'source_checkpoint':str(args.checkpoint)}
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({'onnx':str(out),'bytes':out.stat().st_size,'classes':len(ck['labels']),'arch':ck['arch']},indent=2))

if __name__=='__main__': main()
