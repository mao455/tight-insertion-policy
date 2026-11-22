"""One real B2 DP update, regenerated nominal, real Transformer updates/reload."""
import copy,gc,json,shutil
from pathlib import Path
import torch
from course.environment import ROOT
from course.train import read,write

def run(root,role):
 from visuotactile.precision import state_fp32
 from visuotactile.compact import sha256
 from visuotactile.native_data import visual_records,Episode,cache_residual_episode,residual_batch,residual_previous_targets,residual_supervision_mask
 from visuotactile.checkpointing import save,training_state,cpu_snapshot
 from visuotactile.models.native_dp import load_policy,TargetResidual
 from visuotactile.models.vendor.ema_model import EMAModel
 from visuotactile.trajectory_training import execute_update
 from visuotactile.fresh_visit_sampling import load_profile,visit
 from tools.tight_fresh_train import new_model,fresh_objective
 from visuotactile.anchor_continue import residual_aux,residual_prefix_state
 from course.resources import CourseLease
 state_fp32();dr=root/role/'state_data';data=read(dr/'data.json');ids=read(root/role/'cache_request.json')['ids']
 records=visual_records(dr,ids,'G00',device='cuda:0',target_window_contract='recovery_onset_commit12_v1')
 stats=torch.load(dr/'statistics.pt',map_location='cpu',weights_only=False)
 model=new_model(role,stats);parent=ROOT/f'resources/training/parents/{role}_72_dp.pt'
 old=torch.load(parent,map_location='cpu',weights_only=False);model.load_state_dict(old['ema']);del old
 averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
 opt=torch.optim.AdamW(model.parameters(),lr=3e-5,betas=(.95,.999),weight_decay=1e-6)
 objective,_,_=fresh_objective(root,role,data,records,geometry_sampling_profile=load_profile(root)['geometry'])
 train_ids=ids[:2];episodes=[Episode(dr,i,'I-HistoryDP') for i in train_ids];anchors=[];receipts=[];generators=[]
 for index,i in enumerate(train_ids,1):
  a,r,g=visit(records[i],load_profile(root),identity=i,epoch=1,visit_index=index)
  anchors.append(a);receipts.append(r);generators.append(g)
 before=next(model.parameters()).detach().cpu().clone()
 noise=torch.Generator(device='cuda').manual_seed(2026111403)
 report=execute_update(model,opt,ema,objective,train_ids,[records[i] for i in train_ids],episodes,noise,
  selected_anchors=anchors,anchor_receipts=receipts,geometry_generators=generators,geometry_sampling_profile=load_profile(root)['geometry'])
 changed=float((next(model.parameters()).detach().cpu()-before).abs().max())
 if changed==0:raise AssertionError('DP did not update')
 out=root/'smoke';out.mkdir();dp=out/'dp.pt'
 configuration=dict(statistics_sha256=sha256(dr/'statistics.pt'),learning_rate=3e-5,disposable=True,scope='Real-data course smoke; not a full exposure50 retraining')
 payload=training_state(model,averaged,opt,ema,torch.Generator().manual_seed(2026111402),noise,
  kind='dp',group='I-HistoryDP',updates=1,epoch=1,configuration=configuration,best=None,history=[])
 save(dp,payload);del payload,model,opt,averaged,ema,objective,episodes;gc.collect();torch.cuda.empty_cache()
 frozen=load_policy(torch.load(dp,map_location='cpu',weights_only=False),'cuda').eval().requires_grad_(False)
 lease=CourseLease(root);nominal={}
 for i in ids:
  binding=dict(base_checkpoint_sha256=sha256(dp),statistics_sha256=sha256(dr/'statistics.pt'),
   data_sha256=sha256(dr/'data.json'),residual_statistics_sha256=sha256(root/'B/state_data/statistics.pt'))
  nominal[i]=cache_residual_episode(frozen,records[i],data['entries'][i],lease,cache_identity=binding,
   transfer_mode='episode',scheduler_index_mode='host')
  save(out/(i+'_nominal.pt'),nominal[i]);print('Regenerated nominal with new DP:',i,flush=True)
 del frozen,records;gc.collect();torch.cuda.empty_cache()
 prior=torch.load(ROOT/f'resources/training/parents/{role}_72_transformer.pt',map_location='cpu',weights_only=False)
 cfg=copy.deepcopy(prior['configuration']);stats_path=root/'B/state_data/statistics.pt'
 stats=torch.load(stats_path,map_location='cpu',weights_only=False)
 backend=TargetResidual(stats,cfg['fixed_residual_scales'],execution_adapter=cfg['execution_adapter'],
  loss_contract=cfg['loss_contract'],history_contract=cfg['history_contract'],output_contract=cfg['output_contract']).cuda()
 backend.load_state_dict(prior['model']);del prior
 optimizer=torch.optim.AdamW(backend.parameters(),lr=1e-4,weight_decay=1e-4)
 sample=[nominal[train_ids[0]]];state=None;prefix=[];losses=[];updates=0
 before=next(backend.parameters()).detach().cpu().clone()
 for start in range(0,min(64,len(sample[0]['robot'])),32):
  stop=min(start+32,len(sample[0]['robot']));inputs=residual_batch(sample,start,stop,'cuda:0')
  state=residual_prefix_state(backend,prefix)
  raw,state=backend(inputs,state)
  gate=residual_aux(sample,start,stop,'gate','cuda:0')&inputs['valid']
  limits=torch.stack([r['limits'] for r in sample]).cuda()[:,None]
  chain=backend.compose_chain(residual_aux(sample,start,stop,'base','cuda:0'),raw,gate,limits,
   previous=residual_previous_targets(inputs,limits))
  mask=residual_supervision_mask(sample,start,stop,'cuda:0')&inputs['valid']
  parts=backend.objective_components(chain['u_exec'],residual_aux(sample,start,stop,'target','cuda:0'),chain['delta'],mask,
   u_pre=chain['u_pre'],execution_bounds=chain['execution_bounds'])
  loss=parts['total'];optimizer.zero_grad(set_to_none=True);loss.backward()
  norm=torch.nn.utils.clip_grad_norm_(backend.parameters(),1.)
  if not torch.isfinite(norm):raise ValueError('Nonfinite Transformer gradient')
  optimizer.step();prefix.append(inputs);updates+=1;losses.append(float(loss.detach()))
 rear_changed=float((next(backend.parameters()).detach().cpu()-before).abs().max())
 if rear_changed==0:raise AssertionError('Transformer did not update')
 cfg.update(base_checkpoint=str(dp),base_sha256=sha256(dp),statistics_sha256=sha256(dr/'statistics.pt'),
  residual_statistics_path=str(stats_path),residual_statistics_sha256=sha256(stats_path),disposable=True)
 rear=out/'transformer.pt'
 save(rear,cpu_snapshot(dict(residual_version=backend.version,group='I-HistoryDP',contract=frozen_contract(dp),
  model=backend.state_dict(),configuration=cfg,updates=updates,optimizer=optimizer.state_dict())))
 expected=backend.state_dict();reloaded=torch.load(rear,map_location='cpu',weights_only=False)
 reloaded_model=TargetResidual(stats,cfg['fixed_residual_scales'],execution_adapter=cfg['execution_adapter'],
  loss_contract=cfg['loss_contract'],history_contract=cfg['history_contract'],output_contract=cfg['output_contract']).cuda()
 reloaded_model.load_state_dict(reloaded['model']);reloaded_model.eval();backend.eval()
 check_inputs=residual_batch([nominal[ids[-1]]],0,32,'cuda:0')
 with torch.no_grad():
  expected_output,_=backend(check_inputs,None);actual_output,_=reloaded_model(check_inputs,None)
  reload_error=float((expected_output-actual_output).abs().max())
 if reload_error>1e-6:raise AssertionError('Reloaded Transformer predictions differ')

 if any(not torch.equal(v.cpu(),reloaded['model'][k]) for k,v in expected.items()):raise AssertionError('Saved/reloaded Transformer differs')
 deploy=root/'deploy';deploy.mkdir();shutil.copy2(dp,deploy/f'{role}_130_50_dp.pt');shutil.copy2(rear,deploy/f'{role}_130_50_transformer.pt');shutil.copy2(stats_path,deploy/'u_statistics.pt')
 from course.policy import load_pair
 loaded=load_pair(deploy,published=False,role=role)
 write(out/'result.json',dict(passed=True,scope='Real-data pipeline smoke only; one DP update and two Transformer updates, not full retraining',
  dp_updates=1,dp_parameter_max_change=changed,transformer_updates=updates,transformer_parameter_max_change=rear_changed,
  transformer_losses=losses,nominal_episodes=ids,dp_sha256=sha256(dp),transformer_sha256=sha256(rear),pair_reload=True,transformer_reload_prediction_max_abs=reload_error))
 print(json.dumps(read(out/'result.json')),flush=True)
def frozen_contract(p):return torch.load(p,map_location='cpu',weights_only=False)['contract']
