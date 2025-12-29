# groundingdino/models/GroundingDINO/ops/jittor_ms_deform_attn.py
# Jittor pure-Python port of ms_deform_attn_core_pytorch (for Deformable DETR / GroundingDINO)
# This implementation uses jittor.nn.grid_sample and relies on autograd provided by Jittor.

import jittor as jt
from jittor import nn

def ms_deform_attn_core_jittor(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    value: (N, S, M, D)  -- S = sum(H_l * W_l)
    value_spatial_shapes: iterable/list/var of shape (num_levels, 2) with (H_l, W_l) each
    sampling_locations: (N, Lq, M, L, P, 2), normalized in [0,1] (per level)
    attention_weights: (N, Lq, M, L, P)

    Returns:
        output: (N, Lq, M*D)
    """
    # convert shapes to python list if jt.Var
    if isinstance(value_spatial_shapes, jt.Var):
        vsh = value_spatial_shapes.numpy().int().tolist()
    else:
        vsh = [tuple(map(int, x)) for x in value_spatial_shapes]

    N, S, M, D = value.shape
    _, Lq, M_, L, P, _ = sampling_locations.shape
    assert M == M_, "num_heads mismatch"

    # split value per level by spatial size
    value_splits = []
    start = 0
    for (H, W) in vsh:
        len_hw = int(H) * int(W)
        v = value[:, start:start+len_hw, :, :]  # (N, H*W, M, D)
        # reshape to (N, M, D, H, W) then to (N*M, D, H, W)
        v = v.reshape((N, H, W, M, D)).permute((0,3,4,1,2)).reshape((N*M, D, H, W))
        value_splits.append(v)
        start += len_hw

    # sampling_locations normalized [0,1] -> grid_sample needs [-1,1]
    sampling_grids = sampling_locations * 2.0 - 1.0  # (N, Lq, M, L, P, 2)

    sampling_value_list = []
    for lid, (H, W) in enumerate(vsh):
        # sampling_grid_l: (N, Lq, M, P, 2) -> (N, M, Lq, P, 2) -> (N*M, Lq, P, 2)
        sampling_grid_l = sampling_grids[:, :, :, lid]  # (N, Lq, M, P, 2)
        sampling_grid_l = sampling_grid_l.transpose((0,2,1,3,4)).reshape((N*M, Lq, P, 2))

        # value_l is (N*M, D, H, W)
        value_l = value_splits[lid]

        # grid_sample(input, grid) -> (N*M, D, Lq, P)
        # jittor.nn.grid_sample signature close to pytorch's
        sampling_value_l = nn.grid_sample(value_l, sampling_grid_l, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l)  # each is (N*M, D, Lq, P)

    # stack across levels -> shape (N*M, D, Lq, L, P)
    sampling_value = jt.stack(sampling_value_list, dim=3)  # dim 3 is level dim
    # flatten last two dims => (N*M, D, Lq, L*P)
    sampling_value = sampling_value.reshape((N*M, D, Lq, L*P))

    # reshape attention weights: (N, Lq, M, L, P) -> (N, M, Lq, L*P) -> (N*M, 1, Lq, L*P)
    attn = attention_weights.transpose((0,2,1,3,4)).reshape((N*M, 1, Lq, L*P))

    # weighted sum: (N*M, D, Lq, L*P) * (N*M, 1, Lq, L*P) -> sum over last dim -> (N*M, D, Lq)
    weighted = (sampling_value * attn).sum(-1)  # (N*M, D, Lq)

    # reshape -> (N, M*D, Lq) -> transpose to (N, Lq, M*D)
    out = weighted.reshape((N, M*D, Lq)).transpose((0,2,1)).contiguous()
    return out


# A lightweight Module wrapper similar to the PyTorch version
class MSDeformAttnJittor(nn.Module):
    def __init__(self, embed_dim=256, num_levels=4, num_heads=8, num_points=4, im2col_step=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        assert embed_dim % num_heads == 0
        self.im2col_step = im2col_step

    def execute(self, value, spatial_shapes, sampling_locations, attention_weights):
        # value: (N, S, M, D)
        # spatial_shapes: iterable/list/Var with shape (num_levels, 2)
        # sampling_locations: (N, Lq, M, L, P, 2)
        # attention_weights: (N, Lq, M, L, P)
        return ms_deform_attn_core_jittor(value, spatial_shapes, sampling_locations, attention_weights)
