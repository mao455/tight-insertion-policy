"""Verify declared SDK versions and isolate project imports from the research tree."""
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
from course.environment import ROOT


def main():
    expected={'isaacsim':'5.1.0.0','torch':'2.7.0+cu128','torchvision':'0.22.0+cu128',
        'diffusers':'0.39.0','einops':'0.8.2','h5py':'3.16.0','numpy':'1.26.0'}
    actual={n:metadata.version(n) for n in expected}
    if actual!=expected:
        raise RuntimeError('Expected pinned SDK environment '+json.dumps(expected)+'; observed '+json.dumps(actual))
    import course,visuotactile,evaluation
    imported={m.__name__:str(Path(m.__file__).resolve()) for m in [course,visuotactile,evaluation]}
    if any(not Path(p).is_relative_to(ROOT) for p in imported.values()):
        raise RuntimeError('Course import escaped the clone: '+str(imported))
    forbidden=[p for p in sys.path if 'bc+rl_RSS' in p]
    if forbidden:
        raise RuntimeError('Research PYTHONPATH leaked into this environment: '+str(forbidden))
    print(json.dumps(dict(python=sys.version,versions=actual,project_imports=imported),indent=2))


if __name__=='__main__':main()
