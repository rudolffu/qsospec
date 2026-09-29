"""Read-only comparisons against frozen archives; all writes stay in this folder."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/tmp/qsospec-adaptive-mpl'
from pathlib import Path
import json,pickle,time,hashlib,argparse
from dataclasses import replace
import numpy as np
import pandas as pd
from qsospec import fit_hbeta_complex,HbetaComplexConfig
from qsospec.io.run_store import load_model
from qsospec.fitting.adaptive_oiii import residual_diagnostics
from qsospec.oiii_measurements import profile_distribution

OUT=Path(__file__).resolve().parent
QSO=Path('/Users/yuming/astro/prpsl_sched/jwstc6/targets/qsofeed/qsospec')
BASE=Path('/Users/yuming/astro/ml_projects/dr1agn/mlspecz_data/outputs/oiii_hbeta_diagnostics')


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n')


def inputs():
    for row in pd.read_parquet(QSO/'spectra_input.parquet').to_dict('records'):
        yield row['short_id'],str(QSO/'run'),row['qsospec_object_key']
    yield 'J1242',str(BASE/'nonparametric_kinematics/top_tens_spectra/j1242_vi_nobalmer_5500/run'),'desi:coadd:39633524751336740::z_vi_nobalmer_5500'
    yield 'J1743',str(BASE/'outflow_dominated_selection/spectral_panels/J1743_workflow.pkl'),None


def load(source,key):
    if key is None:
        with open(source,'rb') as f:return pickle.load(f)
    return load_model(source,key)


def compare(name,w,source,key,*,config=None,group='fixed_continuum'):
    folder=OUT/group/name;folder.mkdir(parents=True,exist_ok=True)
    cfg=config or HbetaComplexConfig()
    old=w.hbeta
    original={k:hashlib.sha256(np.asarray(a).tobytes()).hexdigest() for k,a in
              [('flux',w.spectrum.flux),('continuum',w.continuum.model),('line_model',old.model)]}
    path=folder/'adaptive.pkl'
    if path.exists():
        with path.open('rb') as f:new=pickle.load(f)
    else:
        new=fit_hbeta_complex(w.spectrum,w.continuum,cfg)
        with path.open('wb') as f:pickle.dump(new,f)
    # Explicitly refresh measurements/quality in *new validation products* only;
    # recorded models and optimizer searches are unchanged.
    from qsospec.complex_recipes import ComplexRecipe,ComponentRecipe
    from qsospec.fitting.adaptive_oiii import finalize_quality
    from qsospec.oiii_measurements import record_oiii_measurements
    definition=dict(new.metadata['model_definition'])
    definition['components']=tuple(ComponentRecipe(**{**d,'line_ids':tuple(d['line_ids'])}) for d in definition['components'])
    recipe=ComplexRecipe(**definition)
    finalize_quality(new,w.spectrum,w.continuum,recipe,new.metadata['candidate_selection'])
    record_oiii_measurements(new,w.spectrum,w.continuum,recipe)
    with path.open('wb') as f:pickle.dump(new,f)
    assert w.spectrum.z==new.metadata['input_redshift']
    assert original=={k:hashlib.sha256(np.asarray(a).tobytes()).hexdigest() for k,a in
              [('flux',w.spectrum.flux),('continuum',w.continuum.model),('line_model',old.model)]}
    core=old.param_values.get('narrow.velocity_kms',0.)
    row=dict(name=name,source=source,object_key=key,z=w.spectrum.z,success=new.success,
        selected_model=new.selected_model,n_components=len(new.metadata['oiii_components']),
        profile_adequate=new.metadata['profile_adequate'],warnings='|'.join(new.metadata['profile_quality_flags']),
        runtime_s=new.metadata['oiii_runtime_seconds'],core_reference_reliable=new.metadata['oiii_core_reference']['reliable'],
        baseline_chi2=old.chi2,adaptive_chi2=new.chi2,
        baseline_reduced_chi2=old.reduced_chi2,adaptive_reduced_chi2=new.reduced_chi2,
        resolution_status=new.metadata['line_lsf']['status'])
    for label,fit in [('baseline',old),('adaptive',new)]:
        diag=residual_diagnostics(w.spectrum.wave_rest,(w.spectrum.flux-w.continuum.model-fit.model)/w.spectrum.err,
                                  old.fit_mask & new.fit_mask,core)
        for line in ('4960','5008'):row[f'{label}_rms_{line}']=np.sqrt(diag[line]['mean_chi2'])
        if label=='baseline':
            n=2 if 'OIII5007_wing.flux' in fit.param_values else 1
            q=profile_distribution(fit.param_values,n,reference=fit.param_values['narrow.velocity_kms'])
        else:
            q=profile_distribution(fit.param_values,len(fit.metadata['oiii_components']),reference=fit.param_values['narrow.velocity_kms'])
            if not fit.metadata['oiii_core_reference']['reliable']:
                q['v02_kms']=np.nan;q['fraction_beyond_500_kms']=np.nan
            row['adaptive_input_frame_v02_kms']=fit.metrics['oiii_5008_full_observed_input_frame_v02_kms']
        for k in ('w80_kms','w90_kms','v02_kms','fraction_beyond_500_kms'):row[label+'_'+k]=q[k]
    dump(folder/'summary.json',row);dump(folder/'adaptive.json',new.summary())
    dump(folder/'provenance.json',dict(source=source,object_key=key,frozen_arrays_sha256=original,
        seed=cfg.oiii_random_seed,redshift_unchanged=True,host_continuum_fixed=True))
    print(json.dumps(row),flush=True)
    return row


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--names',nargs='*');args=parser.parse_args()
    rows=[]
    for name,source,key in inputs():
        if args.names and name not in args.names:continue
        rows.append(compare(name,load(source,key),source,key))
        pd.DataFrame(rows).to_csv(OUT/'fixed_continuum_comparison.csv',index=False)
