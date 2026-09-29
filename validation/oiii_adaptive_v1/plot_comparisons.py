"""Both doublet lines, residuals, and final-state decisions at fixed continuum."""
import os
os.environ['MPLCONFIGDIR']='/tmp/qsospec-adaptive-mpl'
from pathlib import Path
import pickle,json,textwrap
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from validate_real import OUT,inputs,load
from qsospec import lines

plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':11,
    'axes.labelsize':12,'axes.titlesize':13,'xtick.direction':'in','ytick.direction':'in',
    'xtick.top':True,'ytick.right':True,'xtick.minor.visible':True,'ytick.minor.visible':True,
    'axes.grid':False,'savefig.dpi':180})


def plot(name,workflow,adaptive):
    baseline=workflow.hbeta;wave=workflow.spectrum.wave_rest
    flux=workflow.spectrum.flux-workflow.continuum.model
    valid=workflow.spectrum.valid_mask
    fig,axes=plt.subplots(3,2,figsize=(7.09,7.1),sharex='col',layout='constrained',gridspec_kw={'height_ratios':[2.5,1,1]})
    for j,(feature,label) in enumerate([('oiii_4960','4960'),('oiii_5008','5008')]):
        v=299792.458*np.log(wave/lines.get(feature).vacuum_wavelength)
        mask=(v>-2200)&(v<2200);x=v[mask];good=valid[mask]
        y=flux[mask];y=np.where(good,y,np.nan)
        axes[0,j].plot(x,y,c='.6',lw=.7,label='Data',rasterized=True)
        axes[0,j].plot(x,baseline.model[mask],c='#D55E00',ls='--',lw=1.2,label='Baseline')
        axes[0,j].plot(x,adaptive.model[mask],c='#0072B2',lw=1.4,label='Adaptive')
        # Display only this doublet member's components; other lines remain in the total.
        prefix='OIII4959_' if j==0 else 'OIII5007_'
        for key,value in sorted(adaptive.component_models.items()):
            if key.startswith(prefix):axes[0,j].plot(x,value[mask],lw=.8,alpha=.8,ls=':',label=key.split('_',1)[1])
        for i,(fit,color) in enumerate([(baseline,'#D55E00'),(adaptive,'#0072B2')],1):
            res=(flux-fit.model)/workflow.spectrum.err
            axes[i,j].plot(x,np.where(good,res[mask],np.nan),c=color,lw=.8,rasterized=True)
            axes[i,j].axhline(0,color='.4',lw=.6)
            axes[i,j].axhspan(-3,3,color='.9',alpha=.6,zorder=-1)
            scale=max(4.,np.nanpercentile(abs(res[mask][good]),99)*1.15)
            axes[i,j].set_ylim(-scale,scale)
        axes[0,j].set_title('[O III] '+label)
        axes[2,j].set_xlabel(r'Input-frame velocity [km s$^{-1}$]')
        axes[2,j].set_xlim(-2200,2200)
    axes[0,0].set_ylabel(r'$F_\lambda$ (saved input units)')
    axes[1,0].set_ylabel(r'Baseline / $\sigma$')
    axes[2,0].set_ylabel(r'Adaptive / $\sigma$')
    axes[0,0].legend(fontsize=8,ncol=1,loc='upper right')
    n=len(adaptive.metadata['oiii_components'])
    flags=adaptive.metadata['profile_quality_flags']
    flags=[x for x in flags if x!='observed_profile_resolution_unavailable']
    message=', '.join(x.replace('_',' ') for x in flags) or 'residual checks passed'
    reference='gas reference reliable' if adaptive.metadata['oiii_core_reference']['reliable'] else 'gas reference unreliable'
    fig.suptitle(name+f' | {n} components | '+reference+'\n'+textwrap.fill(message,85),fontsize=12)
    return fig


if __name__=='__main__':
    folder=OUT/'qa';folder.mkdir(exist_ok=True)
    with PdfPages(folder/'fixed_continuum_comparisons.pdf') as pdf:
        for name,source,key in inputs():
            w=load(source,key)
            with (OUT/'fixed_continuum'/name/'adaptive.pkl').open('rb') as stream:fit=pickle.load(stream)
            fig=plot(name,w,fit);pdf.savefig(fig);fig.savefig(folder/(name+'.png'));plt.close(fig)
    with PdfPages(folder/'control_comparisons.pdf') as pdf:
        jobs=json.loads((OUT/'control_selection.json').read_text())
        for row in jobs:
            source=OUT/'control_inputs'/row['survey']/'runs'/('shard-%03d'%int(row['source_shard_id']))
            w=load(str(source),row['object_key'])
            with (OUT/'controls'/row['validation_name']/'adaptive.pkl').open('rb') as stream:fit=pickle.load(stream)
            fig=plot(row['validation_name'],w,fit);pdf.savefig(fig);plt.close(fig)
    print('Wrote 13 target panels and 100 control panels',flush=True)
