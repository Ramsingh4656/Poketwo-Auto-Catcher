import json, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
for key in ['arch','validation_threshold','val','test','hard_case','cpu_inference_ms_mean','cpu_inference_ms_p95','ram_rss_delta_mb','model_size_mb']:
    if key in d:
        v=d[key]
        if isinstance(v,dict):
            v={k:v[k] for k in v if k not in ('confusion_matrix','predictions','scores','accepted')}
        print(key, json.dumps(v, sort_keys=True))
if 'test' in d:
    cm=d['test']['confusion_matrix']; wrong=[]
    for i,row in enumerate(cm):
        for j,n in enumerate(row):
            if i!=j and n: wrong.append((n,i,j))
    print('largest_confusions', sorted(wrong, reverse=True)[:20])
if 'hard_case_by_transform' in d:
    for k,v in d['hard_case_by_transform'].items():
        print('hard_transform',k,{x:v[x] for x in ('top1','precision_macro','recall_macro','f1_macro','unknown_rate')})
