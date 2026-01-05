# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import jittor as jt
import numpy as np

from models.GroundingDINO.ops.ms_deform_attn_func import MSDeformAttnFunction
from models.GroundingDINO.ms_deform_attn import ms_deform_attn_core_jittor


N, M, D = 1, 2, 2
Lq, L, P = 2, 2, 2
shapes: jt.Var = jt.array([(6, 4), (3, 2)], dtype=jt.int64)
level_start_index = jt.concat((shapes.new_zeros((1, )), jt.cumsum(shapes.prod(1),0)[:-1]))
S = sum([(H*W).item() for H, W in shapes])



@jt.no_grad()
def check_forward_equal_with_jittor_double():
    value = jt.rand(N, S, M, D) * 0.01
    sampling_locations = jt.rand(N, Lq, M, L, P, 2)
    attention_weights = jt.rand(N, Lq, M, L, P) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    output_jittor = ms_deform_attn_core_jittor(value.double(), shapes, sampling_locations.double(), attention_weights.double())
    output_cuda = MSDeformAttnFunction.apply(value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step)
    assert isinstance(output_cuda, jt.Var)
    assert isinstance(output_jittor, jt.Var)
    fwdok = np.allclose(output_cuda.numpy(), output_jittor.numpy())
    max_abs_err = (output_cuda - output_jittor).abs().max()
    max_rel_err = ((output_cuda - output_jittor).abs() / output_jittor.abs()).max()

    print(f'* {fwdok} check_forward_equal_with_jittor_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


@jt.no_grad()
def check_forward_equal_with_jittor_float():
    value = jt.rand(N, S, M, D) * 0.01
    sampling_locations = jt.rand(N, Lq, M, L, P, 2)
    attention_weights: jt.Var = jt.rand(N, Lq, M, L, P) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdims=True).sum(-2, keepdims=True)
    im2col_step = 2
    output_jittor = ms_deform_attn_core_jittor(value, shapes, sampling_locations, attention_weights)
    output_cuda = MSDeformAttnFunction.apply(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step)
    fwdok = np.allclose(output_cuda, output_jittor, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_cuda - output_jittor).abs().max()
    max_rel_err = ((output_cuda - output_jittor).abs() / output_jittor.abs()).max()

    print(f'* {fwdok} check_forward_equal_with_jittor_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


def gradcheck(func, inputs, eps=1e-4, rtol=1e-3, atol=1e-3):
    """
    Checks if the analytical gradient computed by Jittor matches the 
    numerical gradient computed via finite difference.
    
    Args:
        func (callable): A Python function that takes inputs and returns a scalar or tensor.
        inputs (list or tuple): A list/tuple of jt.Vars.
        eps (float): Perturbation size for finite difference.
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.
        
    Returns:
        bool: True if gradients match, False otherwise.
    """
    # 1. Compute Analytical Gradients (Autograd)
    # Ensure inputs require grad
    for x in inputs:
        x.start_grad()
        
    output = func(*inputs)
    
    # If output is not scalar, sum it to get a scalar for backward
    if output.numel() > 1:
        target = jt.sum(output)
    else:
        target = output
        
    # Get analytical grads
    # jt.grad automatically handles the backward graph
    analytical_grads = jt.grad(target, inputs)
    
    # 2. Compute Numerical Gradients (Finite Difference)
    # Formula: (f(x+eps) - f(x-eps)) / (2*eps)
    
    for i, x in enumerate(inputs):
        # We only check inputs that require grad
        if not x.requires_grad:
            continue
            
        x_np = x.data # Get numpy array (forces sync)
        grad_analytical = analytical_grads[i].data
        grad_numerical = np.zeros_like(x_np)
        
        # Iterate over every element in the input tensor
        it = np.nditer(x_np, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig_val = x_np[idx]
            
            # f(x + eps)
            x_np[idx] = orig_val + eps
            x.data = x_np # Update Jittor var
            y_plus = func(*inputs).sum().data
            
            # f(x - eps)
            x_np[idx] = orig_val - eps
            x.data = x_np # Update Jittor var
            y_minus = func(*inputs).sum().data
            
            # Restore value
            x_np[idx] = orig_val
            x.data = x_np
            
            # Central difference
            grad_numerical[idx] = (y_plus - y_minus) / (2 * eps)
            it.iternext()
            
        # 3. Compare
        if not np.allclose(grad_analytical, grad_numerical, rtol=rtol, atol=atol):
            diff = np.abs(grad_analytical - grad_numerical)
            print(f"Gradient check failed for input {i}!")
            print(f"Max difference: {diff.max()}")
            print(f"Analytical: {grad_analytical}")
            print(f"Numerical:  {grad_numerical}")
            return False

    return True


def check_gradient_numerical(channels=4, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True):

    value: jt.Var = jt.rand(N, S, M, channels) * 0.01
    sampling_locations = jt.rand(N, Lq, M, L, P, 2)
    attention_weights: jt.Var = jt.rand(N, Lq, M, L, P) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdims=True).sum(-2, keepdims=True)
    im2col_step = 2
    func = MSDeformAttnFunction.apply

    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight

    gradok = gradcheck(func, (value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step))

    print(f'* {gradok} check_gradient_numerical(D={channels})')


if __name__ == '__main__':
    check_forward_equal_with_jittor_double()
    check_forward_equal_with_jittor_float()

    for channels in [30, 32, 64, 71]:
        check_gradient_numerical(channels, True, True, True)
