"""Finite local resource lease; no historical owner or monitoring service."""
from contextlib import contextmanager
import json,time,shutil
from pathlib import Path
class CourseLease:
 def __init__(self,root,deadline=None):self.root=Path(root);self.deadline=deadline;self.work_s=self.wait_s=self.pacing_s=0
 def check(self,deadline=None):
  s=json.loads((self.root/'session.json').read_text())
  if time.time()> (deadline or self.deadline or s['deadline']):raise TimeoutError('Course training deadline reached')
  if (self.root/'STOP').exists():raise InterruptedError('STOP requested')
  if shutil.disk_usage(self.root).free<12*1024**3:raise RuntimeError('Keep 12 GiB disk free')
 def memory(self,cuda=True):return {}
 @contextmanager
 def block(self,deadline=None,cuda=True):
  self.check(deadline)
  if cuda:
   import torch
   if torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('Insufficient free GPU memory; preserve this run')
  began=time.monotonic()
  try:yield
  finally:self.work_s+=time.monotonic()-began
