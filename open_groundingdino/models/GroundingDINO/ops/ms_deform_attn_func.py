# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import jittor as jt

from ..ms_deform_attn import MultiScaleDeformableAttention as MSDA
from ..ms_deform_attn import ms_deform_attn_forward


class MSDeformAttnFunction(jt.Function):
    def execute(self, value, value_spatial_shapes, value_level_start_index, 
                sampling_locations, attention_weights, im2col_step):
        # 在Jittor中，ctx被替换为self
        self.im2col_step = im2col_step
        
        output = ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, 
            sampling_locations, attention_weights, self.im2col_step)
        
        # Jittor保存中间变量
        self.save_vars = value, value_spatial_shapes, value_level_start_index, \
                        sampling_locations, attention_weights
        return output

    def grad(self, grad_output):
        # 获取保存的变量
        value, value_spatial_shapes, value_level_start_index, \
        sampling_locations, attention_weights = self.save_vars
        
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, 
                sampling_locations, attention_weights, grad_output, 
                self.im2col_step)
        
        # 返回所有输入的梯度，None表示该输入不需要梯度
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
