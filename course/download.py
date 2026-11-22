"""Download fixed private-release assets and verify every file before use."""
import argparse,concurrent.futures,hashlib,json,subprocess,tarfile
from pathlib import Path
from course.environment import ROOT
REPOSITORY='mao455/tight-insertion-policy'
TAG='resources-v1'
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def acquire(row):
 target=ROOT/'outputs/downloads'/row['file'];target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():
  subprocess.run(['gh','release','download',TAG,'--repo',REPOSITORY,'--pattern',row['file'],'--dir',str(target.parent)],check=True)
 if target.stat().st_size!=row['bytes'] or digest(target)!=row['sha256']:raise ValueError('Resource checksum failed: '+row['file'])
 if row.get('extract',False):
  with tarfile.open(target) as archive:archive.extractall(ROOT,filter='data')
 else:
  import shutil
  dest=ROOT/row['destination'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,dest)
 print('Verified',row['file'],flush=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',choices=['demo','smoke','training']);a=p.parse_args()
 manifest=json.loads((ROOT/'configs/resources.json').read_text())
 rows=[r for r in manifest['assets'] if a.bundle in r['bundles']]
 if not rows:raise ValueError('Empty resource bundle')
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(acquire,rows))
 print('Ready:',a.bundle,flush=True)
if __name__=='__main__':main()
