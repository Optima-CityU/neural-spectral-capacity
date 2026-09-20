import os
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
from scipy.stats import spearmanr
P=Path(os.environ.get('NSC_ROOT','.'))/'paper';OUT=Path(os.environ.get('NSC_OUT','outputs'));OUT.mkdir(parents=True,exist_ok=True)
from nsc_utils import psi_mp
source=P/'dataset/autoformer/data/autoformer_tiny_1k.json'
raw=json.loads(source.read_text());archived={r['idx']:r for r in json.loads((OUT/'autoformer_tiny_scores.json').read_text())}
rows=[]
for key,a in sorted(raw.items(),key=lambda x:int(x[0])):
 ns=a['net_setting'];layers=[]
 def at(k,i):return ns[k][i] if isinstance(ns[k],list) else ns[k]
 for i in range(ns['layer_num']):
  d=at('embed_dim',i);h=at('num_heads',i);f=int(d*at('mlp_ratio',i))
  layers.append(3*h*psi_mp(d,64)+psi_mp(d,64*h)+psi_mp(f,d)+psi_mp(d,f))
 rows.append(dict(id=int(key),accuracy=a['performance']['Imagenet']['clean'],layers=layers,sum=sum(layers),min=min(layers),log_product=sum(math.log(v+1e-30) for v in layers)))
assert len(rows)==1001
assert np.allclose([r['sum'] for r in rows],[archived[r['id']]['nsc_mp'] for r in rows],rtol=1e-12,atol=1e-8)
out={'benchmark':'AutoFormer-Tiny','n':len(rows),'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'protocol':'NSC fixed head_dim 64, per-head QKV; layer sums then sum/min/product. Product evaluated as sum(log(layer sum)); distinct scores round 6 decimals, matching archived aggregation ablation.','results':{}}
for k in ['sum','min','log_product']:
 v=np.array([r[k] for r in rows]);out['results'][k]={'spearman':float(spearmanr(np.round(v,6),[r['accuracy'] for r in rows]).statistic),'spearman_raw_float':float(spearmanr(v,[r['accuracy'] for r in rows]).statistic),'distinct':int(len(np.unique(np.round(v,6))))}
# Monotone transform consistency: same correlation with explicit finite product after rounding numerical ties in log domain.
lp=np.round([r['log_product'] for r in rows],6)
out['product_tie_stable_spearman']=float(spearmanr(lp,[r['accuracy'] for r in rows]).statistic)
D=OUT;(D/'aggregation_autoformer_results.json').write_text(json.dumps(out,indent=2));(D/'aggregation_autoformer_per_architecture.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(out,indent=2))
