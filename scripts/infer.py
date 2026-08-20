from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model-dir',type=Path,default=Path('model/poketwo_detector')); ap.add_argument('--image',type=Path,required=True); ap.add_argument('--top-k',type=int,default=5); args=ap.parse_args()
    metadata=json.loads((args.model_dir/'metadata.json').read_text(encoding='utf-8')); labels=metadata['labels']; sess=ort.InferenceSession(str(args.model_dir/'pokemon_detector.onnx'),providers=['CPUExecutionProvider'])
    with Image.open(args.image) as im: im=im.convert('RGB').resize((224,224),Image.Resampling.LANCZOS)
    arr=np.asarray(im,dtype=np.float32)/255.0; arr=(arr-np.asarray(metadata['normalization']['mean'],dtype=np.float32))/np.asarray(metadata['normalization']['std'],dtype=np.float32); arr=np.transpose(arr,(2,0,1))[None].astype('float32')
    logits=sess.run(['logits'],{'images':arr})[0][0]; logits=logits-logits.max(); p=np.exp(logits); p=p/p.sum(); ids=np.argsort(p)[-min(args.top_k,len(labels)):][::-1]
    print(json.dumps([{'label':labels[int(i)],'confidence':float(p[i])} for i in ids],indent=2))

if __name__=='__main__': main()
