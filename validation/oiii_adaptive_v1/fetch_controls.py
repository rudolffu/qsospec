"""Stream only selected immutable shards from agnkiaa, without server writes."""
import json,subprocess,shlex,tarfile
from pathlib import Path
OUT=Path(__file__).resolve().parent
jobs=json.loads((OUT/'control_selection.json').read_text())
script='''import json,sys,hashlib,tarfile
from pathlib import Path
root=Path('/home/yuming/data/euclid/mlspecz_data/outputs/optical_qsospec_v2')
jobs=json.load(sys.stdin);paths=set()
for j in jobs:
 folder=Path(j['survey'])/'runs'/('shard-%03d'%int(j['source_shard_id']))
 paths.add(folder/'manifest.json')
 digest=hashlib.sha256(j['object_key'].encode()).hexdigest()[:20]
 for table in ['models','objects','measurements','warnings','inputs']:
  p=folder/'data'/table/('part-'+digest+'.parquet')
  if not (root/p).is_file():raise FileNotFoundError(str(root/p))
  paths.add(p)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for p in sorted(paths):tar.add(root/p,arcname=str(p),recursive=False)
'''
archive=OUT/'control_inputs.tar.gz'
with archive.open('wb') as out:
 subprocess.run(['ssh','-p','6000','-o','BatchMode=yes','yuming@localhost',
     '/home/yuming/anaconda3/bin/python -c '+shlex.quote(script)],input=json.dumps(jobs).encode(),stdout=out,check=True)
destination=OUT/'control_inputs';destination.mkdir(exist_ok=True)
with tarfile.open(archive) as tar:tar.extractall(destination,filter='data')
print(archive.stat().st_size,'bytes for',len(jobs),'controls')
