# ------------------------------------------------------------------------
# Grounding DINO (Jittor-only MSDeformAttn)
# ------------------------------------------------------------------------

import math
import os
import warnings
from typing import Optional

import jittor as jt
import jittor.nn as nn
from jittor import init


# ------------------------------------------------------------------------
# helpers
def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n <= 0):
        return False
    return (n & (n - 1)) == 0


# ------------------------------------------------------------------------
# Python fallback (core implementation)
# ------------------------------------------------------------------------

def multi_scale_deformable_attn_pytorch(
    value: jt.Var,
    value_spatial_shapes: jt.Var,
    sampling_locations: jt.Var,
    attention_weights: jt.Var,
) -> jt.Var:
    """
    Pure Jittor implementation of Multi-Scale Deformable Attention
    (ported from PyTorch fallback)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    # split value by levels
    spatial_shapes = value_spatial_shapes.tolist()
    split_sizes = [int(h * w) for h, w in spatial_shapes]
    value_list = value.split(split_sizes, dim=1)

    sampling_grids = 2.0 * sampling_locations - 1.0  # [0,1] -> [-1,1]

    sampling_value_list = []

    for level, (H_, W_) in enumerate(spatial_shapes):
        # (bs, H*W, heads, dim) -> (bs*heads, dim, H, W)
        value_l = (
            value_list[level]
            .reshape(bs, H_ * W_, num_heads * embed_dims)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, H_, W_)
        )
        # (bs, queries, heads, points, 2)
        sampling_grid_l = (
            sampling_grids[:, :, :, level]
            .transpose(1, 2)
            .reshape(bs * num_heads, num_queries, num_points, 2)
        )
        sampling_grid_l = sampling_grid_l.float32()

        # grid_sample: (N, C, H, W) + (N, H_out, W_out, 2)
        sampled = nn.grid_sample(
            value_l,
            sampling_grid_l,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # (bs*heads, dim, queries, points)

        sampling_value_list.append(sampled)

    # attention weights
    attention_weights = (
        attention_weights
        .transpose(1, 2)
        .reshape(bs * num_heads, 1, num_queries, num_levels * num_points)
    )

    # stack sampled values
    output = (
        jt.stack(sampling_value_list, dim=-2)
        .reshape(bs * num_heads, embed_dims, num_queries, num_levels * num_points)
        * attention_weights
    ).sum(-1)

    # (bs, queries, heads*dim)
    output = (
        output
        .reshape(bs, num_heads * embed_dims, num_queries)
        .transpose(1, 2)
    )

    return output


# ------------------------------------------------------------------------
# Main Module
# ------------------------------------------------------------------------

class MultiScaleDeformableAttention(nn.Module):
    """
    Jittor-only Multi-Scale Deformable Attention
    (interface compatible with GroundingDINO / Deformable-DETR)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        img2col_step: int = 64,
        batch_first: bool = False,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads, but got {} and {}".format(
                    embed_dim, num_heads
                )
            )
        head_dim = embed_dim // num_heads

        self.batch_first = batch_first

        if not _is_power_of_2(head_dim):
            warnings.warn(
                """
                You'd better set d_model in MSDeformAttn to make sure that
                each dim of the attention head a power of 2, which is more efficient.
                """
            )

        self.im2col_step = img2col_step
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.init_weights()

    # ------------------------------------------------------------------

    def init_weights(self):
        init.constant_(self.sampling_offsets.weight, 0.0)

        thetas = jt.arange(self.num_heads, dtype=jt.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = jt.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = (
            (grid_init / grid_init.abs().max(dim=-1, keepdims=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with jt.no_grad():
            self.sampling_offsets.bias = grid_init.view(-1)
        init.constant_(self.attention_weights.weight, 0.0)
        init.constant_(self.attention_weights.bias, 0.0)
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0.0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0.0)

    # ------------------------------------------------------------------

    def execute(
        self,
        query: jt.Var,
        key: Optional[jt.Var] = None,
        value: Optional[jt.Var] = None,
        query_pos: Optional[jt.Var] = None,
        key_padding_mask: Optional[jt.Var] = None,
        reference_points: Optional[jt.Var] = None,
        spatial_shapes: Optional[jt.Var] = None,
        level_start_index: Optional[jt.Var] = None,
        **kwargs
    ) -> jt.Var:

        if value is None:
            value = query

        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = jt.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError("reference_points must have last dim 2 or 4")

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output

def ms_deform_attn_forward(
    value: jt.Var, 
    spatial_shapes: jt.Var,
    level_start_index: jt.Var,
    sampling_loc: jt.Var,
    attn_weight: jt.Var,
    grad_output: jt.Var,
    im2col_step: int
):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cuda_src_path = os.path.join(current_dir, "src", "ms_deform_attn.cu")

    with open(cuda_src_path, "r") as f:
        # Read the file content
        kernel_source = f.read()
    # ... (Asserts omitted as in original code) ...

    batch = value.shape[0]
    spatial_size = value.shape[1]
    num_heads = value.shape[2]
    channels = value.shape[3]

    num_levels = spatial_shapes.shape[0]

    num_query = sampling_loc.shape[1]
    num_point = sampling_loc.shape[4]

    im2col_step_ = min(batch, im2col_step)

    if batch % im2col_step_ != 0:
        raise KeyError(f"batch({batch}) must divide im2col_step({im2col_step_})")

    # Initialize outputs
    grad_value = jt.zeros_like(value)
    grad_sampling_loc = jt.zeros_like(sampling_loc)
    grad_attn_weight = jt.zeros_like(attn_weight)

    batch_n = im2col_step_
    per_value_size = spatial_size * num_heads * channels
    per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2
    per_attn_weight_size = num_query * num_heads * num_levels * num_point
    
    # Reshape grad_output for batching
    grad_output_n = grad_output.view((batch // im2col_step_, batch_n, num_query, num_heads, channels))
    
    # Define the C++ kernel signature (Header)
    # Ensure the actual implementation of 'ms_deformable_col2im_cuda' is available 
    # to the compiler (e.g., included in a .cu file you load, or linked).
    cuda_header = f"""
    #include <cuda_runtime.h>
    #include <vector>
    #include <algorithm>
    
    // Inject the content of your .cu file here
    {kernel_source}
    """

    for n in range(0, batch // im2col_step_):
        grad_output_g = grad_output_n[n]
        
        # We construct the C++ source code string for this iteration
        # @inX_p gives the raw pointer to the X-th input variable
        # in0_type gives the C++ type of the 0-th input (e.g., float)
        cuda_src = f"""
            using scalar_t = in0_type;

            // Cast input pointers to mutable because we are writing to them inplace
            // In Jittor, inputs are const by default in the kernel wrapper
            scalar_t* grad_value_ptr = const_cast<scalar_t*>(@in6_p);
            scalar_t* grad_sampling_loc_ptr = const_cast<scalar_t*>(@in7_p);
            scalar_t* grad_attn_weight_ptr = const_cast<scalar_t*>(@in8_p);

            // Offsets calculation
            int n = {n};
            int im2col_step_ = {im2col_step_};
            long per_value_size = {per_value_size};
            long per_sample_loc_size = {per_sample_loc_size};
            long per_attn_weight_size = {per_attn_weight_size};

            ms_deformable_col2im_cuda(
                0, // Use default stream or q.stream() if managed internally
                @in5_p, // grad_output_g
                @in0_p + n * im2col_step_ * per_value_size, // value
                (int64_t*)@in1_p, // spatial_shapes
                (int64_t*)@in2_p, // level_start_index
                @in3_p + n * im2col_step_ * per_sample_loc_size, // sampling_loc
                @in4_p + n * im2col_step_ * per_attn_weight_size, // attn_weight
                {batch_n}, {spatial_size}, {num_heads}, {channels}, {num_levels}, {num_query}, {num_point},
                grad_value_ptr + n * im2col_step_ * per_value_size,
                grad_sampling_loc_ptr + n * im2col_step_ * per_sample_loc_size,
                grad_attn_weight_ptr + n * im2col_step_ * per_attn_weight_size
            );
        """

        # Execute the kernel
        # We pass the output vars (grad_value etc) as inputs so we can access their pointers
        # and modify them in place via const_cast in the C++ code.
        jt.code(
            inputs=[
                value,              # in0
                spatial_shapes,     # in1
                level_start_index,  # in2
                sampling_loc,       # in3
                attn_weight,        # in4
                grad_output_g,      # in5
                grad_value,         # in6 (Modified inplace)
                grad_sampling_loc,  # in7 (Modified inplace)
                grad_attn_weight    # in8 (Modified inplace)
            ],
            outputs=[], # No new outputs created, side effects only
            cuda_header=cuda_header,
            cuda_src=cuda_src
        )

    return grad_value, grad_sampling_loc, grad_attn_weight
