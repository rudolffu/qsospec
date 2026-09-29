"""Full host/continuum workflow and explicit continuum-sensitivity comparisons."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/tmp/qsospec-adaptive-mpl'
import json,pickle,time,typing,dataclasses
from pathlib import Path
import numpy as np,pandas as pd
import qsospec
from qsospec.workflows.host.config import HostDecompConfig
from qsospec.workflows.host_workflow import _run_global_fit_with_optional_host
from qsospec.workflows.host.io import SpectrumData
from validate_real import OUT,QSO,inputs,load,dump,compare


def convert(value, hint):
    if value is None:return None
    args=typing.get_args(hint);origin=typing.get_origin(hint)
    candidates=(hint,)+args
    if isinstance(value,dict):
        sub=next((x for x in candidates if isinstance(x,type) and dataclasses.is_dataclass(x)),None)
        if sub:return decode(sub,value)
    if isinstance(value,list):
        subtype=args[0] if args else object
        converted=[convert(x,subtype) for x in value]
        return tuple(converted) if origin is tuple else converted
    return value


def decode(cls,raw):
    types=typing.get_type_hints(cls)
    return cls(**{key:convert(value,types[key]) for key,value in raw.items()})


def run(name,source,key):
    original=load(source,key);folder=OUT/'full_workflow'/name;folder.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((QSO/'resolved_scientific_config.json').read_text())
    cfg=decode(qsospec.GlobalContinuumConfig,manifest['global_config'])
    host=decode(HostDecompConfig,manifest['host_config'])
    if name=='J1242':
        cfg=dataclasses.replace(cfg,power_law=dataclasses.replace(cfg.power_law,mode='single'),
            balmer_pseudocontinuum=dataclasses.replace(cfg.balmer_pseudocontinuum,enabled=False),
            continuum_windows=tuple(w for w in cfg.continuum_windows if w[0]<5500))
    total=original.total_spectrum or original.spectrum
    scale=1+total.z if total.flux_frame=='rest' else 1.
    metadata=dataclasses.replace(total.metadata,flux_frame='observed').to_dict()
    sd=SpectrumData(wave_obs=total.wave_obs,flux=total.flux/scale,error=total.err/scale,
        redshift=total.z,object_id=name,ra=total.metadata.ra,dec=total.metadata.dec,
        mask=(~total.valid_mask).astype(int),resolution=total.resolution,
        metadata={'spectrum_metadata':metadata,'galactic_extinction':{'status':'caller_preprocessed','original_correction':metadata.get('galactic_extinction',{}),'source':'saved corrected total-spectrum arrays'},
                  'galactic_extinction_corrected':True})
    path=folder/'workflow.pkl'
    if path.exists():
        with path.open('rb') as f:result=pickle.load(f)
    else:
        start=time.perf_counter()
        result=_run_global_fit_with_optional_host(sd,source=source,input_path=source,
            object_id=name,run_host_decomp=original.host_decomp_enabled,host_config=host,
            global_config=cfg,hbeta_config=qsospec.HbetaComplexConfig(),
            galactic_extinction_config=qsospec.GalacticExtinctionConfig(enabled=False),
            uncertainty_config=qsospec.UncertaintyConfig(monte_carlo_trials=0),complexes=('hbeta_oiii',))
        result.metadata['validation_full_runtime_s']=time.perf_counter()-start
        with path.open('wb') as f:pickle.dump(result,f)
    assert result.spectrum.z==original.spectrum.z
    with (OUT/'fixed_continuum'/name/'adaptive.pkl').open('rb') as f:fixed=pickle.load(f)
    row=dict(name=name,z=result.spectrum.z,continuum_success=result.continuum.success,
        host_refitted=original.host_decomp_enabled,selected=result.hbeta.selected_model,
        adequate=result.hbeta.metadata['profile_adequate'],runtime_s=result.metadata['validation_full_runtime_s'],
        w80_fixed=fixed.metrics['oiii_5008_full_w80_kms'],w80_full=result.hbeta.metrics['oiii_5008_full_w80_kms'],
        local_continuum_fixed=fixed.param_values['continuum.constant'],local_continuum_full=result.hbeta.param_values['continuum.constant'],
        host_global_relative_rms=float(np.sqrt(np.mean((result.spectrum.flux-original.spectrum.flux)**2))/np.nanmedian(abs(original.spectrum.flux))),
        continuum_relative_rms=float(np.sqrt(np.mean((result.continuum.model-original.continuum.model)**2))/np.nanmedian(abs(original.continuum.model))))
    dump(folder/'summary.json',row);dump(folder/'hbeta.json',result.hbeta.summary());print(json.dumps(row),flush=True)
    return row

if __name__=='__main__':
    rows=[];failed=[]
    for name,source,key in inputs():
        try:rows.append(run(name,source,key))
        except Exception as exc:
            import traceback;traceback.print_exc();failed.append(dict(name=name,error=repr(exc)))
        pd.DataFrame(rows).to_csv(OUT/'full_workflow_comparison.csv',index=False)
        dump(OUT/'full_workflow_failures.json',failed)
