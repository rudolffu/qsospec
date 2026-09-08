"""Compare the four native modes on explicitly extracted archived spectra.

Input JSONL fields: object_id, wave_rest, flux, error, mask, z,
spectrum_metadata, host_model, run_root. Flux is the archived host-subtracted
rest-frame flux. This is a conditional fixed-host model comparison.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import qsospec
from qsospec.metadata import resolve_spectrum_metadata


def compare(row):
    spectrum=qsospec.Spectrum.from_arrays(row['wave_rest'],row['flux'],err=row['error'],
        z=row['z'],wave_frame='rest',mask=row['mask'],metadata=resolve_spectrum_metadata(metadata=row['spectrum_metadata']))
    host=None if row.get('host_model') is None else np.asarray(row['host_model'])
    base=qsospec.GlobalContinuumConfig()
    report={'object_id':str(row['object_id']),'z':row['z'],'source_run':row['run_root'],
        'host_policy':'fixed_archived_host_for_common_comparison','modes':{}}
    arrays={'wave_rest':spectrum.wave_rest,'flux':spectrum.flux,'error':spectrum.err,'valid':spectrum.valid_mask}
    for name,bridge,soft in [('legacy',False,False),('bridge_only',True,False),('soft_hgamma_only',False,True),('combined',True,True)]:
        config=replace(base,regional_iron=qsospec.RegionalIronConfig(enabled=bridge),
            balmer_pseudocontinuum=replace(base.balmer_pseudocontinuum,sync_with_hgamma='soft' if soft else 'hard_legacy'))
        start=perf_counter()
        result=qsospec.fit_global_lines(spectrum,config,host_model_on_grid=host,
            complexes=('mgii','hbeta_oiii','oii_nev_neiii_hgamma'))
        total=result.continuum.model.copy()
        for fit in result.line_complexes.values():
            total+=fit.model
        residual=(spectrum.flux-total)/spectrum.err
        regions={}
        for lo,hi in [(3400,3600),(3600,4000),(4000,4400),(2700,2900),(4700,5100)]:
            good=spectrum.valid_mask&(spectrum.wave_rest>=lo)&(spectrum.wave_rest<hi)
            values=residual[good]
            regions[f'{lo}-{hi}']={'n':int(good.sum()),'signed_bias_sigma':float(values.mean()),
                'rms_sigma':float(np.sqrt(np.mean(values**2))),
                'lag1_correlation':float(np.corrcoef(values[:-1],values[1:])[0,1])}
        report['modes'][name]={'seconds':perf_counter()-start,'success':result.continuum.success,
            'regions':regions,'samples':result.metadata['continuum_samples'],
            'sample_errors':result.metadata['continuum_sample_errors'],'warnings':result.warning_codes(),
            'bridge':result.continuum.metadata.get('regional_iron'),
            'hgamma_status':result.continuum.metadata.get('hgamma_joint_status'),
            'width_diagnostics':result.continuum.metadata.get('iron_width_diagnostics'),
            'parameters':result.continuum.param_values}
        arrays[name]=total
        print(row['object_id'],name,round(perf_counter()-start,2),flush=True)
    return report,arrays


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    output=Path(args.output_dir);output.mkdir(exist_ok=False)
    for line in Path(args.input).read_text().splitlines():
        row=json.loads(line)
        report,arrays=compare(row)
        (output/(row['object_id']+'.json')).write_text(json.dumps(report,indent=2))
        np.savez_compressed(output/(row['object_id']+'.npz'),**arrays)
