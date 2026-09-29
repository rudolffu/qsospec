"""Audit optimizer-free measurements and time explicit legacy comparison refits."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import time,pickle
import numpy as np,pandas as pd
from validate_real import OUT,inputs,load
from qsospec import HbetaComplexConfig,fit_hbeta_complex
from qsospec.complex_recipes import ComponentRecipe,ComplexRecipe
from qsospec.oiii_measurements import record_oiii_measurements
from qsospec.fitting import global_fit

if __name__=='__main__':
    rows=[]
    for name,source,key in inputs():
        w=load(source,key)
        start=time.perf_counter()
        legacy=fit_hbeta_complex(w.spectrum,w.continuum,HbetaComplexConfig(oiii_profile_mode='legacy'))
        elapsed=time.perf_counter()-start
        with (OUT/'fixed_continuum'/name/'adaptive.pkl').open('rb') as f:fit=pickle.load(f)
        data=dict(fit.metadata['model_definition']);data['components']=tuple(ComponentRecipe(**{**x,'line_ids':tuple(x['line_ids'])}) for x in data['components'])
        recipe=ComplexRecipe(**data)
        calls=[];original=global_fit._solve_once_with_fallback
        def forbidden(*args,**kwargs):
            calls.append(1);raise AssertionError('Unexpected optimizer in measurement')
        global_fit._solve_once_with_fallback=forbidden
        fit.metadata.pop('_peak_parameter_state',None)
        start=time.perf_counter()
        try:record_oiii_measurements(fit,w.spectrum,w.continuum,recipe)
        finally:global_fit._solve_once_with_fallback=original
        processing=time.perf_counter()-start
        rows.append(dict(name=name,legacy_refit_runtime_s=elapsed,legacy_refit_chi2=legacy.chi2,
            legacy_matches_frozen_model=bool(np.allclose(legacy.model,w.hbeta.model,rtol=1e-5,atol=1e-7)),
            adaptive_fit_runtime_s=fit.metadata['oiii_runtime_seconds'],measurement_runtime_s=processing,
            measurement_optimizer_calls=len(calls),input_redshift_unchanged=w.spectrum.z==fit.metadata['input_redshift']))
        pd.DataFrame(rows).to_csv(OUT/'runtime_and_postprocessing.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
