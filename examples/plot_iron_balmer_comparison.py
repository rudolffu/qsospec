"""Plot unsmoothed comparison arrays produced by compare_archived_iron_balmer.py."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qsospec-iron-comparison-mpl")
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(source,output):
    output.mkdir(parents=True,exist_ok=True)
    modes=[('legacy','Legacy','#0072B2'),('bridge_only','Bridge only','#E69F00'),
        ('soft_hgamma_only','Soft Hγ only','#009E73'),('combined','Combined','#CC79A7')]
    style={'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':10.,
        'axes.grid':False,'xtick.direction':'in','ytick.direction':'in',
        'xtick.top':True,'ytick.right':True,'xtick.minor.visible':True,'ytick.minor.visible':True}
    with plt.rc_context(style):
        for path in sorted(source.glob('*.npz')):
            arrays=np.load(path);report=json.loads(path.with_suffix('.json').read_text())
            wave=arrays['wave_rest'];good=arrays['valid']&(wave>=3300.)&(wave<=4450.)
            fig,axes=plt.subplots(5,1,figsize=(7.09,8.8),sharex=True,gridspec_kw={'height_ratios':[2.2,1,1,1,1]})
            fig.subplots_adjust(left=.15,right=.985,bottom=.075,top=.93,hspace=.14)
            axes[0].plot(wave,np.where(good,arrays['flux'],np.nan),color='.65',lw=.45,rasterized=True)
            for i,(key,label,color) in enumerate(modes):
                axes[0].plot(wave,np.where(good,arrays[key],np.nan),lw=1.15,color=color,label=label)
                residual=(arrays['flux']-arrays[key])/arrays['error']
                axes[i+1].plot(wave,np.where(good,residual,np.nan),lw=.45,color=color,rasterized=True)
                axes[i+1].axhline(0,color='.3',lw=.6)
                axes[i+1].set_ylim(-5,5)
                axes[i+1].text(.015,.84,label,transform=axes[i+1].transAxes,va='top',bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
                axes[i+1].set_ylabel(r'$(f-m)/\sigma$')
            axes[0].set_title(f"DESI / Euclid {report['object_id']}   z = {report['z']:.4f}\nArchived host held fixed",fontsize=11)
            axes[0].set_ylabel(r'$f_{\lambda,\rm rest}$ [$10^{-17}$ cgs / Å]')
            axes[0].legend(loc='upper right',ncol=2,frameon=True,fontsize=9)
            validflux=arrays['flux'][good]
            lo,hi=np.percentile(validflux,[1,99]);axes[0].set_ylim(lo-.12*(hi-lo),hi+.25*(hi-lo))
            axes[-1].set_xlabel('Rest-frame vacuum wavelength [Å]')
            axes[-1].set_xlim(3300,4450)
            fig.savefig(output/(path.stem+'.pdf'),dpi=200)
            fig.savefig(output/(path.stem+'.png'),dpi=120)
            plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();render(args.source,args.output)
