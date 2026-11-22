"""Download fixed public-release assets and verify every file before use."""
import argparse, concurrent.futures, hashlib, json, os, shutil, tarfile, tempfile, time, urllib.request
from pathlib import Path
from course.environment import ROOT
REPOSITORY = 'mao455/tight-insertion-policy'
TAG = 'resources-v1'
def digest(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def verified(path, row):
    return path.is_file() and path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
def acquire(row):
    target = ROOT/'outputs/downloads'/row['file']
    target.parent.mkdir(parents=True, exist_ok=True)
    if not verified(target, row):
        with tempfile.TemporaryDirectory(prefix='acquire-', dir=target.parent) as temporary:
            received = Path(temporary)/row['file']
            url = f'https://github.com/{REPOSITORY}/releases/download/{TAG}/{row["file"]}'
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=120) as response, received.open('wb') as stream:
                        shutil.copyfileobj(response, stream, length=1024*1024)
                    break
                except Exception:
                    if attempt == 2: raise
                    time.sleep(2*(attempt+1))
            if not verified(received, row):
                raise ValueError('Resource checksum failed: '+row['file'])
            os.replace(received, target)
    if row.get('extract', False):
        with tarfile.open(target) as archive:
            archive.extractall(ROOT, filter='data')
    else:
        destination = ROOT/row['destination']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, destination)
    print('Verified', row['file'], flush=True)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', choices=['demo', 'smoke', 'training'])
    args = parser.parse_args()
    manifest = json.loads((ROOT/'configs/resources.json').read_text())
    rows = [row for row in manifest['assets'] if args.bundle in row['bundles']]
    if not rows:
        raise ValueError('Empty resource bundle')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(acquire, rows))
    print('Ready:', args.bundle, flush=True)
if __name__ == '__main__':
    main()
