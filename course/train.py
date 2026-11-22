"""Portable continuation of the released exposure50 experiment."""
import argparse,copy,json,shutil,time
from pathlib import Path
from course.environment import ROOT

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 from visuotactile.training_resources import atomic_json
 atomic_json(p,v)
def prepare(root,role,smoke=False,hours=96):
 import torch
 from visuotactile.compact import sha256
 from visuotactile.narrow_data import record,dagger_record,actual_configuration_labels
 from visuotactile.checkpointing import save
 from visuotactile.state_image_loss import geometry_loss_context
 template=ROOT/'resources/training/templates';manifest=read(ROOT/'configs/training130.json')
 root.mkdir(parents=True,exist_ok=False)
 s=read(template/'session.json');s.update(course_runtime=True,deadline=time.time()+hours*3600,
  training_deadline=time.time()+hours*3600,cache_deadline=time.time()+hours*3600,
  dino_checkpoint=str(ROOT/'resources/dinov2_vits14_pretrain.pth'),resource_root=str(root),owned_worker_roots=[str(root)])
 s.pop('expert_checkpoint',None);s.pop('expert_sha256',None)
 for n in ['session.json','metadata.json','training_sampling.json','effective_exposure.json','bounded_extension.json',
           'full_trajectory_batch.json','frozen_visual_io.json','frozen_visual_hotset.json','backend_episode_sampling.json','cuda_allocator_policy.json']:
  if (template/n).exists():shutil.copy2(template/n,root/n)
 write(root/'session.json',s)
 data=read(template/role/'data.json')
 for i,e in data['entries'].items():e['path']=str(ROOT/manifest['entries'][i]['path'])
 ids=data['splits']['train'][:2]+data['splits']['validation'][:1] if smoke else data['splits']['train']+data['splits']['validation']
 records={};contexts={}
 for i in ids:
  e=data['entries'][i]
  if sha256(e['path'])!=e['source_sha256']:raise ValueError('Raw training file hash differs: '+i)
  if e.get('trajectory_kind')=='expert_takeover_v1':r,_,c=dagger_record(e)
  else:
   _,_,r=record(e['path'],episode=e['episode'],privileged=False)
   r.update(q_reached=actual_configuration_labels(e)[0],u_expert=r['command'].clone(),
    supervision_mask=torch.ones(len(r['time']),dtype=torch.bool),control_segment=torch.zeros(len(r['time']),dtype=torch.long))
   c=geometry_loss_context(e)
  records[i]=r;contexts[i]=c
 # Both roles retain the same command-affine population and family split.
 for target in 'AB':
  d=copy.deepcopy(data);d['supervision']='q_reached' if target=='A' else 'u_expert'
  d['normalization_identity']=read(template/target/'data.json')['normalization_identity']
  rr=root/target;dr=rr/'state_data';dr.mkdir(parents=True)
  write(dr/'data.json',d)
  save(dr/'numeric.pt',dict(contract=d['contract'],data_sha256=sha256(dr/'data.json'),records=records))
  save(dr/'loss_context.pt',contexts);shutil.copy2(template/target/'statistics.pt',dr/'statistics.pt')
  continuation=read(template/target/'continuation.json')
  continuation.update(path=str(ROOT/f'resources/training/parents/{target}_72_dp.pt'),
   previous_backend=str(ROOT/f'resources/training/parents/{target}_72_transformer.pt'))
  continuation.pop('source_root',None);continuation.pop('reuse_index',None)
  write(rr/'continuation.json',continuation);write(rr/'session.json',s)
  shutil.copy2(root/'metadata.json',rr/'metadata.json')
  write(rr/'cache_request.json',dict(ids=ids,smoke=smoke))
 batch=read(root/'full_trajectory_batch.json')
 for target in 'AB':batch['roles'][target]['data_sha256']=sha256(root/target/'state_data/data.json')
 write(root/'full_trajectory_batch.json',batch)
 write(root/'prepared.json',dict(role=role,smoke=smoke,ids=ids,train=130,validation=6,
  original_archive400='Distinct historical archive; not the current training pool',parents=manifest['parents']))
 print('Prepared',root,flush=True)
def visual(root,role,smoke=False):
 from visuotactile.native_data import cache_visual_pair
 ids=read(root/role/'cache_request.json')['ids']
 cache_visual_pair(root/role,identities=ids if smoke else None)
 if smoke:shutil.copy2(root/role/'cache_acceptance_index.json',root/role/'cache_index.json')
 from visuotactile.compact import sha256
 other='B' if role=='A' else 'A'
 destination=root/other/'state_data/perception'
 if not destination.exists():destination.symlink_to(root/role/'state_data/perception',target_is_directory=True)
 index=read(root/role/'cache_index.json');index['data_sha256']=sha256(root/other/'state_data/data.json')
 write(root/other/'cache_index.json',index)
 # Only local immutable features are shared; each DP rebuilds learned history.

def dp(root,role):
 from tools.tight_fresh_train import gate,train
 from visuotactile.trajectory_training import profile
 gate(root)
 train(root,role,60,offline_anchor_batch=64,trajectory_batch_profile=profile())
def nominal(root,role):
 from tools.tight_fresh_backend import prepare as backend_prepare
 from tools.tight_finish import residual_cache
 from visuotactile.frozen_visual_io import load_profile
 backend_prepare(root,role,60,continue_rear=True,offline_anchor_batch=64,frozen_visual_io=load_profile(root))
 residual_cache(root/role/'backend')
def rear(root,role):
 from visuotactile.anchor_continue import residual_train
 residual_train(root/role/'backend'/role)
 export(root,role)
def export(root,role):
 from visuotactile.compact import sha256
 p=root/'deploy';p.mkdir(exist_ok=True)
 for src,dst in [(root/role/'training/best.pt',f'{role}_130_50_dp.pt'),
   (root/role/'backend'/role/'training/best.pt',f'{role}_130_50_transformer.pt'),
   (root/'B/state_data/statistics.pt','u_statistics.pt')]:shutil.copy2(src,p/dst)
 write(p/'pair.json',{q.name:sha256(q) for q in p.iterdir() if q.suffix=='.pt'})
 print('Deployable matched pair:',p,flush=True)
def smoke_run(root,role):
 from course.smoke import run
 return run(root,role)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','visual','dp','nominal','transformer','all','smoke'])
 p.add_argument('--root',type=Path,default=ROOT/'outputs/train');p.add_argument('--role',choices=['A','B'],default='A');p.add_argument('--hours',type=float,default=96)
 a=p.parse_args();root=a.root.resolve()
 if a.stage in ('prepare','all','smoke'):prepare(root,a.role,a.stage=='smoke',a.hours)
 if a.stage in ('visual','all','smoke'):visual(root,a.role,a.stage=='smoke')
 if a.stage=='smoke':smoke_run(root,a.role);return
 if a.stage in ('dp','all'):dp(root,a.role)
 if a.stage in ('nominal','all'):nominal(root,a.role)
 if a.stage in ('transformer','all'):rear(root,a.role)
if __name__=='__main__':main()
