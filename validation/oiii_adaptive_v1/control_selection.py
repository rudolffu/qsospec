"""Deterministic 100-object, survey/type/SNR-stratified control manifest."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
ROOT=Path('/Users/yuming/astro/ml_projects/dr1agn/mlspecz_data/outputs')
OUT=Path(__file__).resolve().parent
parent=pd.read_parquet(ROOT/'oiii_hbeta_diagnostics/nonparametric_kinematics/oiii_nonparametric_unique_objects.parquet')
rows=[]
for survey,g in parent.groupby('survey'):
    obj=pd.read_parquet(ROOT/'optical_qsospec_v2'/survey/'products/all/objects.parquet',
        columns=['object_id','object_key','source_shard_id','class_final','median_continuum_snr'])
    obj['object_id']=obj.object_id.astype(str)
    rows.append(g.merge(obj,on=['object_id','object_key'],validate='one_to_one'))
pool=pd.concat(rows,ignore_index=True)
pool['snr_bin']=pd.cut(pool.oiii_5008_core_snr,[-np.inf,5,20,np.inf],labels=['below_5','5_to_20','above_20'],right=False).astype(str)
pool['optical_class']=pool.class_final.fillna('unknown')
pool['selection_hash']=pool.object_key.map(lambda k:hashlib.sha256(('1729:'+k).encode()).hexdigest())
strata=[g.sort_values('selection_hash').reset_index(drop=True) for _,g in pool.groupby(['survey','optical_class','snr_bin'],sort=True)]
chosen=[];level=0
while len(chosen)<100:
 for g in strata:
  if level<len(g):chosen.append(g.iloc[level])
  if len(chosen)==100:break
 level+=1
selected=pd.DataFrame(chosen).reset_index(drop=True)
selected['validation_name']=['control_%03d'%i for i in range(100)]
selected.to_csv(OUT/'control_selection.csv',index=False)
selected.to_parquet(OUT/'control_selection.parquet',index=False)
(OUT/'control_selection.json').write_text(selected[['validation_name','object_key','survey','source_shard_id','optical_class','snr_bin']].to_json(orient='records',indent=2))
print(selected.groupby(['survey','optical_class','snr_bin']).size().to_string())
