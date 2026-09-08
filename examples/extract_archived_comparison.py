"""Read selected archived spectra over SSH without modifying the remote runs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = r'''
import sys,json,qsospec
for row in json.load(sys.stdin):
    result=qsospec.load_model(row['run_root'],row['object_key'])
    spectrum=result.spectrum
    payload={**row,'wave_rest':spectrum.wave_rest.tolist(),
        'flux':spectrum.flux.tolist(),'error':spectrum.err.tolist(),
        'mask':spectrum.valid_mask.tolist(),'z':float(spectrum.z),
        'spectrum_metadata':spectrum.metadata.to_dict(),
        'host_model':None if result.host_model_on_quasar_grid is None else result.host_model_on_quasar_grid.tolist()}
    print(json.dumps(payload),flush=True)
'''

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--host',default='yuming@localhost')
    parser.add_argument('--port',type=int,default=6000)
    parser.add_argument('--remote-python',default='/home/yuming/anaconda3/bin/python')
    parser.add_argument('--remote-source',default='/home/yuming/tools/qsospec/src')
    args=parser.parse_args()
    requests=json.loads(args.requests.read_text())
    command=shlex.join(['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+args.remote_source,args.remote_python,'-c',REMOTE])
    with args.output.open('x') as output:
        subprocess.run(['ssh','-p',str(args.port),'-o','BatchMode=yes',args.host,command],
            input=json.dumps(requests),text=True,stdout=output,check=True)
