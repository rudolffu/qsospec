import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,pickle
import pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from validate_real import OUT,load,compare

def worker(row):
    source=OUT/'control_inputs'/row['survey']/'runs'/('shard-%03d'%int(row['source_shard_id']))
    w=load(str(source),row['object_key'])
    answer=compare(row['validation_name'],w,str(source),row['object_key'],group='controls')
    answer.update(optical_class=row['optical_class'],snr_bin=row['snr_bin'],survey=row['survey'])
    return answer

if __name__=='__main__':
    jobs=json.loads((OUT/'control_selection.json').read_text());rows=[];failures=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(worker,j):j for j in jobs}
        for future in as_completed(pending):
            try:rows.append(future.result())
            except Exception as exc:
                failures.append(dict(job=pending[future],error=repr(exc)));print('FAILURE',failures[-1],flush=True)
            pd.DataFrame(rows).to_csv(OUT/'control_comparison.csv',index=False)
            (OUT/'control_failures.json').write_text(json.dumps(failures,indent=2))
    print('COMPLETE',len(rows),'failed',len(failures),flush=True)
