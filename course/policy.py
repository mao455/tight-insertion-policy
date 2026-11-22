"""Resolve portable resources without changing released checkpoint bytes."""
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DP_SHA = 'df4782e5f1640ef10da89ce95d174e90da0703a969fc29e2a80841bbcd4c78fa'
REAR_SHA = '36e6b6e7bcd06aef7577320a3eb6309b4c4153ce09c173374f2d3bc9587554e9'


def load_pair(directory=None, *, published=True, role='A'):
    import torch
    from visuotactile.compact import sha256
    root = (ROOT/'resources') if directory is None else Path(directory).resolve()
    dp = root/f'{role}_130_50_dp.pt'
    rear = root/f'{role}_130_50_transformer.pt'
    stats = root/'u_statistics.pt'
    if published and role != 'A':
        raise ValueError('Only the selected A pair is distributed for demonstration')
    if published and (sha256(dp) != DP_SHA or sha256(rear) != REAR_SHA):
        raise ValueError('Published complete-policy identity mismatch')
    saved = torch.load(rear, map_location='cpu', weights_only=False)
    saved['configuration'] = copy.deepcopy(saved['configuration'])
    saved['configuration']['base_checkpoint'] = str(dp.resolve())
    if sha256(dp) != saved['configuration']['base_sha256']:
        raise ValueError('Transformer belongs to a different frontend')
    if sha256(stats) != saved['configuration']['residual_statistics_sha256']:
        raise ValueError('Command normalizer identity mismatch')
    saved['configuration']['residual_statistics_path'] = str(stats.resolve())
    return saved
