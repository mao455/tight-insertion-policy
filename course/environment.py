"""Load the exact native task overlay used by the frozen evaluation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env_cfg(task='Isaacsimenvs-PreciseAssembly-Direct-v0'):
    if task != 'Isaacsimenvs-PreciseAssembly-Direct-v0':
        raise ValueError('This course package supports Tight-Insertion only')
    import yaml
    from isaacsimenvs.tasks.precise_assembly.precise_assembly_env_cfg import PreciseAssemblyEnvCfg
    cfg = PreciseAssemblyEnvCfg()
    overlay = yaml.safe_load((ROOT/'isaacsimenvs/cfg/task/PreciseAssembly.yaml').read_text())
    overlay.pop('clip_observations', None)
    overlay.pop('clip_actions', None)
    cfg.from_dict(overlay)
    return cfg
