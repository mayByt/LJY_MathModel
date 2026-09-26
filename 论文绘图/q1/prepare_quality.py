"""Restore omitted quality tables in an isolated cache, without altering frozen outputs."""
from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd
import config as C
ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'论文绘图/cache/q1_quality'
sys.path.insert(0,str(ROOT/'问题一/src'))
from problem1.data_quality import load_quality_data,transform_quality,score_quality

def main():
 CACHE.mkdir(parents=True,exist_ok=True)
 cfg=json.loads((ROOT/'问题一/config/problem1_p12_final_v2.json').read_text())
 raw=C.RAW/'A_data_value'
 t=time.time();print('Loading full quality inputs',flush=True)
 loaded=load_quality_data(raw,cfg,CACHE)
 loaded.raw[['domain','id','rps_doc_word_count']].to_csv(CACHE/'raw_lengths.csv.gz',index=False,compression='gzip')
 print('Raw loaded',len(loaded.raw),round(time.time()-t,1),flush=True)
 transformed,spec=transform_quality(loaded.raw,cfg,CACHE)
 print('Transformed; fitting exact 500-bootstrap weighting',round(time.time()-t,1),flush=True)
 result=score_quality(transformed,cfg,CACHE,loaded.text_proxies_a1)
 frozen=pd.read_csv(ROOT/'问题一/results_p12_final_v2/quality_domain.csv').query("mapping_type=='quality_anchor'").set_index('domain')
 actual=result['domain'].set_index('domain').loc[frozen.index]
 errors={c:float(np.abs(actual[c]-frozen[c]).max()) for c in ['q','ci_low','ci_high']}
 check={'n_rows':len(result['scored']),'max_abs_error':errors,'matches_at_1e-8':all(e<1e-8 for e in errors.values()),'matches_at_1e-6':all(e<1e-6 for e in errors.values()),'accepted_tolerance':1e-6,'frozen_numpy':'1.26.4','frozen_pandas':'2.2.2','elapsed_seconds':time.time()-t,'config':cfg,'numpy':np.__version__,'pandas':pd.__version__}
 (CACHE/'verification.json').write_text(json.dumps(check,indent=2,ensure_ascii=False))
 print(json.dumps(check,indent=2,ensure_ascii=False),flush=True)
 if not check['matches_at_1e-6']:raise RuntimeError('Quality reconstruction differs from frozen results; do not proceed silently')
if __name__=='__main__':main()
