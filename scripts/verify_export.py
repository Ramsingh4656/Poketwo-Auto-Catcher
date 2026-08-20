from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from train import build_model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--onnx',type=Path,required=True); args=ap.parse_args()
    ck=torch.load(args.checkpoint,map_location='cpu'); model=build_model(ck['arch'],len(ck['labels'])); model.load_state_dict(ck['state_dict']); model.eval()
    m=onnx.load(str(args.onnx)); onnx.checker.check_model(m)
    x=np.random.default_rng(42).normal(size=(1,3,224,224)).astype('float32')
    with torch.no_grad(): pt=model(torch.from_numpy(x)).numpy()
    sess=ort.InferenceSession(str(args.onnx),providers=['CPUExecutionProvider']); ort_out=sess.run(['logits'],{'images':x})[0]
    diff=np.abs(pt-ort_out)
    result={'onnx_valid':True,'providers':sess.get_providers(),'input':sess.get_inputs()[0].shape,'output':sess.get_outputs()[0].shape,'max_abs_diff':float(diff.max()),'mean_abs_diff':float(diff.mean()),'classes':len(ck['labels'])}
    print(json.dumps(result,indent=2))
    assert diff.max() < 1e-3, result

if __name__=='__main__': main()
