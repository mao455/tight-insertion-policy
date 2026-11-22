"""One complete autonomous episode in the real Isaac Sim main window."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

from course.environment import ROOT


class LocalLease:
    def block(self, **kwargs):
        return nullcontext()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/demo')
    parser.add_argument('--seed', type=int, default=28851006)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--keep-open', action='store_true')
    parser.add_argument('--gpu-wait-seconds',type=float,default=900)
    args = parser.parse_args()
    if not os.environ.get('DISPLAY'):
        raise RuntimeError('A real graphical desktop DISPLAY is required for this demo')
    from course.resources import wait_for_gpu
    wait_for_gpu(5,args.gpu_wait_seconds)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('TMPDIR', str(output/'tmp'))
    Path(os.environ['TMPDIR']).mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_HOME'] = str(ROOT/'outputs/torch')
    from visuotactile.training_resources import atomic_json
    from course.policy import load_pair
    saved = load_pair(args.checkpoint_dir, published=args.checkpoint_dir is None)
    atomic_json(output/'registration.json', dict(seed=args.seed, event_seed=args.seed+100000,
        reset_seed=args.seed+500000, policy='A_130_50', gui=True, expert_used=False,
        physics_hz=120, control_hz=60, planning_hz=5, sampler_seed=7091,
        scope='Known historical success case for demonstration; excluded from historical success rate',
        utc=time.time()))
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=False, enable_cameras=True, device='cuda:0', fast_shutdown=False,
        kit_args='--/rtx/descriptorSets=360000 --/rtx/reservedDescriptors=900000 --/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=4 --/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false').app
    import torch
    from visuotactile.precision import state_fp32
    state_fp32()
    from visuotactile.native_evaluation import make_env
    from visuotactile.environment_rng import IsolatedEnvironment, LaneRandomStreams
    from visuotactile.narrow_data import direct_configuration, configure_direct_reset
    from visuotactile.live_collision import clearance
    from visuotactile.models.native_vision import NativeDinoSpatial
    from visuotactile.gpu_batch_controller import GPUBatchedResidualController
    contract = json.loads((ROOT/'configs/domain.json').read_text())
    config = direct_configuration(contract['domain'], args.seed)
    # Exact native layout: fixed displacement direction from the domain center.
    import numpy as np
    xy = np.array(config['fixed_start_pose'][:2])
    direction = -np.asarray(contract['domain']['center']['relative_xyz'][:2], dtype=float)
    xy += contract['layout']['distance_m']*direction/np.linalg.norm(direction)
    config['fixed_start_pose'][:2] = xy.tolist()
    atomic_json(output/'reset_configuration.json', config)
    env = make_env(True, LocalLease(), num_envs=1, scene_queries=True, camera_content_validation=True,
        execution_backend={'version':'tight_execution_backend_v2','physics_device':'cuda:0',
            'camera_backend':'standard','kit_tasking_threads':4,'physics_threads':2,
            'renderer_descriptor_sets':360000,'renderer_reserved_descriptors':900000})
    from visuotactile.current_render import CurrentRenderReuse
    render_reuse = CurrentRenderReuse(env)
    configure_direct_reset(env, config)
    reset = env._reset_idx
    streams = LaneRandomStreams([args.seed+500000], device=str(env.device))
    def paired_reset(ids):
        with streams.scope(0):
            reset(ids)
    env._reset_idx = paired_reset
    isolated = IsolatedEnvironment(env, args.seed)
    env.clear_recorded_events()
    env.action_control_mode = 'native_command_clipped_pre_delay'
    isolated.reset(args.seed)
    isolated.begin_events(args.seed+100000)
    env.action_control_mode = 'absolute_joint_position_target_rad'
    legal = clearance(env)
    atomic_json(output/'initial_clearance.json', legal)
    if not legal['legal']:
        raise RuntimeError('Predetermined start failed native clearance; no seed substitution')
    metadata = env.research_metadata()
    metadata['config'] = env.cfg.to_dict()
    metadata['execution_adapter'] = saved['configuration']['execution_adapter']
    metadata['env_origins_w'] = env.scene.env_origins.cpu().tolist()
    atomic_json(output/'metadata.json', metadata)
    dino = NativeDinoSpatial().cuda().eval()
    controller = GPUBatchedResidualController(saved, metadata, dino, 1, neural_batch=True,
        scheduler_index_mode='host', vision_microbatch=2)
    from isaaclab.sim import SimulationContext
    env.sim.set_camera_view(eye=(1.25,1.10,1.05), target=(0.,0.,.65))
    initial_q = env._diagnostics()['joint_position_rad'].clone()
    rows=[]
    began=time.perf_counter()
    max_steps=2*env.cfg.termination.episode_length+4
    for tick in range(max_steps):
        if not app.is_running():
            raise RuntimeError('GUI window closed before native evaluator terminal')
        with torch.no_grad():
            packet=env.research_packet
            command=controller.act(packet, env._diagnostics(), [True])
            _,_,terminated,truncated,_=isolated.step(command)
            transition=env.last_transition
            controller.lanes[0].observe_execution(transition)
            reasons={k:bool(v[0]) for k,v in transition['termination_reasons'].items()}
            q=transition['diagnostics_after']['joint_position_rad'][0].cpu()
            rows.append(dict(tick=tick,physics_steps=2*(tick+1),q=q.tolist(),
                native_subgoals=int(transition['diagnostics_after']['successes'][0]),
                object_pose=transition['diagnostics_after']['object_pose_w'][0].cpu().tolist()))
        if tick%30==0:
            print(f'GUI control_tick={tick+1} physics_steps={2*(tick+1)} native_subgoals={rows[-1]["native_subgoals"]}',flush=True)
        if tick in (0,60,180,360):
            try:
                from PIL import ImageGrab
                ImageGrab.grab(xdisplay=os.environ['DISPLAY']).save(output/f'gui_tick_{tick:04d}.png')
            except Exception as error:
                print('Desktop screenshot unavailable:',error,flush=True)
        if bool((terminated|truncated)[0]):
            result=dict(success=reasons['max_successes'],reasons=reasons,software_complete=True,
                legal=True,steps=tick+1,physics_steps=2*(tick+1),native_subgoals=rows[-1]['native_subgoals'],
                policy='A_130_50',seed=args.seed,expert_used=False,gui=True,
                elapsed_s=time.perf_counter()-began,
                maximum_joint_displacement_rad=max(abs(a-b) for row in rows for a,b in zip(row['q'],initial_q[0].cpu().tolist())))
            atomic_json(output/'result.json',result)
            atomic_json(output/'trace.json',rows)
            print('NATIVE EVALUATOR: '+('SUCCESS' if result['success'] else 'FAILURE')+' '+json.dumps(result),flush=True)
            break
    else:
        raise RuntimeError('Native evaluator did not terminate within its original finite bound')
    if args.keep_open:
        while app.is_running(): app.update()
    # Isaac Sim 5.1 can hang while tearing down CUDA render/PhysX contexts.
    # The complete native terminal and trace are durable before process exit.
    import sys
    sys.stdout.flush();sys.stderr.flush();os._exit(0)


if __name__=='__main__':
    main()
