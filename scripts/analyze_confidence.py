import json, sys, numpy as np
D=json.load(open(sys.argv[1]));
for name in ['val','test','hard_case']:
    d=D[name]; scores=np.array(d['scores']); pred=np.array(d['predictions']);
    # scores are aligned with hidden y only in current evaluator's no-threshold output; reconstruct accepted isn't enough.
    print(name, 'score_quantiles', {q:float(np.quantile(scores,q)) for q in [0,.01,.05,.1,.25,.5,.75,.9,.99,1]})
