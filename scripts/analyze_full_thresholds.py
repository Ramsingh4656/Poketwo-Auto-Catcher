from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('evaluation',type=Path); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    d=json.loads(a.evaluation.read_text()); y=np.asarray(d['test']['predictions'])
    # The evaluator stores true labels only indirectly in the prediction artifact, so
    # use accepted statistics already computed for calibrated threshold and report the
    # score/coverage distribution directly for deployment review.
    scores=np.asarray(d['test']['scores'],dtype=float)
    rows=[]
    for t in [0.05,0.25,0.5,0.7,0.8,0.85,0.9,0.95,0.99]:
        accepted=scores>=t
        rows.append({'threshold':t,'accepted_count':int(accepted.sum()),'count':int(len(scores)),'accepted_rate':float(accepted.mean()),'unknown_rate':float(1-accepted.mean()),'score_min_accepted':float(scores[accepted].min()) if accepted.any() else None})
    out={'validation_calibrated_threshold':d['validation_threshold'],'thresholds':rows,'note':'This score-only report does not recompute precision/recall because the compact evaluator artifact does not store true-label arrays; full top-k and macro metrics remain in evaluation.json.'}
    a.output.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
