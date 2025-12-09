# tests_env/test_ms_deform_attn_jittor.py
import jittor as jt
from jittor import nn
import numpy as np

from groundingdino.models.GroundingDINO.ops.jittor_ms_deform_attn import ms_deform_attn_core_jittor

def random_test():
    jt.flags.use_cuda = 0
    N = 2
    num_levels = 2
    HWs = [(6,4), (3,2)]
    S = sum([h*w for h,w in HWs])
    M = 4
    D = 8
    Lq = 5
    P = 3
    L = num_levels
    print("Successfully Create Variables")

    # random inputs
    value = jt.array(np.random.randn(N, S, M, D).astype("float32"))
    spatial_shapes = jt.array(np.array(HWs, dtype=np.int32))
    sampling_locations = jt.array(np.random.rand(N, Lq, M, L, P, 2).astype("float32"))
    attention_weights = jt.array(np.random.rand(N, Lq, M, L, P).astype("float32"))
    # normalize weights similar to usual usage
    attn_sum = attention_weights.sum(-1, keepdims=True).sum(-2, keepdims=True)
    attention_weights = attention_weights / (attn_sum + 1e-6)

    out = ms_deform_attn_core_jittor(value, spatial_shapes, sampling_locations, attention_weights)
    print("output shape:", out.shape)  # expect (N, Lq, M*D)

    # run a backward to ensure autograd works
    out_var = out.sum()
    out_var.backward()
    print("grad on value present:", value.grad is not None)

if __name__ == "__main__":
    random_test()
