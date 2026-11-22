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
    # One GUI environment needs bounded allocation capacity, retaining native
    # integration, contact, solver, control, render and success parameters.
    # Any PhysX capacity overflow is a failed run, never an accepted terminal.
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**18
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**20
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**18
    cfg.sim.physx.gpu_collision_stack_size = 2**26
    return cfg
