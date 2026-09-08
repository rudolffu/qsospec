"""Reproducible small regional-iron / soft-Hgamma comparison.

Run: python examples/iron_balmer_uncertainty.py --output /tmp/iron-validation.json
Add --trials 20 (or more) for a convergence experiment. Small trial counts only
exercise orchestration. No private spectra or external stellar library needed.
"""
from dataclasses import replace
import argparse
import json
from time import perf_counter
import numpy as np
import qsospec
from qsospec.fitting.global_fit import _ContinuumContext


def run(trials=0):
    wave=np.linspace(2600.,5400.,850)
    err=np.full_like(wave,.03)
    spectrum=qsospec.Spectrum.from_arrays(wave,np.ones_like(wave),err=err,
        wave_frame='rest',flux_unit='relative',source='iron-balmer-synthetic')
    base=qsospec.GlobalContinuumConfig(
        power_law=qsospec.PowerLawConfig(norm=2.,slope=-1.,mode='single'),
        uv_iron=qsospec.IronTemplateConfig.vw01(amp=100.),
        optical_iron=qsospec.IronTemplateConfig.park22(amp=100.),
        regional_iron=qsospec.RegionalIronConfig(amp=40.),
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(amplitude=20.,fwhm_kms=3500.,sync_with_hbeta='never'),
        clip_passes=0,blue_absorption_clip_enabled=False)
    context=_ContinuumContext(spectrum,base)
    theta=context.initial.copy()
    for name,value in {'power_law.norm':2.,'power_law.slope':-1.,'uv_iron.amp':100.,'optical_iron.amp':100.,'middle_iron.amp':40.,'balmer_pseudocontinuum.amp':20.}.items():
        theta[context.index[name]]=value
    model=context.model(theta,wave)
    sigma=4341.68*3500./(299792.458*2.35482)
    # Deliberately non-adopted Hgamma ratio and a small omitted spectral feature.
    model+=18.*np.exp(-.5*((wave-4341.68)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
    model+=.015*np.exp(-.5*((wave-3800.)/70.)**2)
    spectrum=replace(spectrum,flux=model+np.random.default_rng(21).normal(0.,err))
    report={'fixture':'non-adopted Hgamma ratio plus omitted smooth feature','trials':trials,'modes':{},
        'real_data_status':'Synthetic-only execution; real comparisons are produced separately by compare_archived_iron_balmer.py.'}
    for label,bridge,soft in [('legacy',False,False),('bridge_only',True,False),('soft_hgamma_only',False,True),('combined',True,True)]:
        config=replace(base,regional_iron=replace(base.regional_iron,enabled=bridge),
            balmer_pseudocontinuum=replace(base.balmer_pseudocontinuum,sync_with_hgamma='soft' if soft else 'hard_legacy'))
        start=perf_counter()
        result=qsospec.fit_global_lines(spectrum,config,complexes=('oii_nev_neiii_hgamma',),
            uncertainty_config=qsospec.UncertaintyConfig(monte_carlo_trials=trials,random_seed=21))
        total=result.continuum.model.copy()
        for fit in result.line_complexes.values():
            total+=fit.model
        residual=(spectrum.flux-total)/err
        regions={}
        for lo,hi in [(3400,3600),(3600,4000),(4000,4400),(2700,2900),(4700,5100)]:
            selected=(wave>=lo)&(wave<hi)&spectrum.valid_mask
            values=residual[selected]
            regions[f'{lo}-{hi}']={'n':int(selected.sum()),'signed_bias_sigma':float(values.mean()),
                'rms_sigma':float(np.sqrt(np.mean(values**2))),
                'lag1_correlation':float(np.corrcoef(values[:-1],values[1:])[0,1])}
        report['modes'][label]={'seconds':perf_counter()-start,'success':result.continuum.success,
            'assessment_regions':regions,'samples':result.metadata['continuum_samples'],
            'sample_errors':result.metadata['continuum_sample_errors'],
            'bridge':result.continuum.metadata.get('regional_iron'),
            'hgamma_status':result.continuum.metadata.get('hgamma_joint_status'),
            'warnings':result.warning_codes(),'bootstrap':result.monte_carlo}
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--trials',type=int,default=0)
    args=parser.parse_args()
    report=run(args.trials)
    with open(args.output,'x') as stream:
        json.dump(report,stream,indent=2)
