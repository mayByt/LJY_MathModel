from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import pandas as pd
import config as C
ROOT = Path(__file__).resolve().parents[2]
R = C.RESULTS['q2']
B = C.RAW / 'B_scaling_laws'
BFILES = {'B1':'pythia_training_log_existing.csv','B4':'scaling_baseline.csv','B5':'published_scaling_data.csv','B7':'supplementary_NQ_experiment_expanded.csv','B8':'supplementary_NQ_experiment_large.csv','B9':'supplementary_large_models.csv','B10':'supplementary_large_baseline.csv'}
@lru_cache(None)
def table(name): return pd.read_csv(R/name)
@lru_cache(None)
def raw(key): return pd.read_csv(B/BFILES[key])
@lru_cache(None)
def obj(name): return json.loads((R/name).read_text())
def path(name): return str(R/name)
def rawpath(key): return str(B/BFILES[key])
def cp(): return obj('classic_scaling_parameters.json')['parameters']
def qp(): return obj('quality_model_parameters.json')['parameters']
def l0(n,d):
 p=cp(); return p['E']+p['A']*np.asarray(n)**(-p['alpha'])+p['B']*np.asarray(d)**(-p['beta'])
def quality(q,s=1):
 ref=obj('generalized_scaling_parameters.json')['Q_ref']; eff=np.clip(ref+s*(np.asarray(q)-ref),1e-6,1); p=qp(); return p['c_Q']*(1-eff)**p['nu_Q']
def pred(n,d,q,s=1): return l0(n,d)+quality(q,s)
def cv_predictions(split='leave_cell'):
 a=table('quality_block_predictions.csv.gz');return a[(a.model=='GQ2')&(a.split_type==split)].copy()
def valid_classic():
 a=table('classic_scaling_bootstrap.csv.gz');return a[a.success].copy()
def rstar_draws():
 z=np.load(R/'problem1_mixture_response_bootstrap.npz',allow_pickle=False)
 domains=[str(a) for a in z['domains']]; ps=obj('problem1_bridge.json')['p_star']; p=np.array([ps[d] for d in domains]);p/=p.sum()
 f=np.array(list(p)+[p[i]*p[j] for i in range(len(p)) for j in range(i+1,len(p))])
 j=np.mean(np.einsum('bkf,f->bk',z['coefficients'],f)/z['loss_iqr'][None,:],axis=1)
 vals=(j-z['j_ref'])/z['r_scale'];return vals[z['valid']]
