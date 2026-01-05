# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import Optional, Tuple
import jittor as jt
import jittor.nn as nn


def _linear_bmm_transpose(x: jt.Var, weight: jt.Var, bias: Optional[jt.Var] = None) -> jt.Var:
    """
    Apply a linear projection using bmm_transpose to avoid broadcasted matmul_transpose
    on large 2D inputs.
    """
    x_shape = x.shape
    if len(x_shape) == 2:
        out = jt.bmm_transpose(x[None, ...], weight[None, ...])[0]
        if bias is not None:
            out = out + bias
        return out
    if len(x_shape) == 3:
        bsz, seq_len, dim = x_shape
        x2 = x.reshape(bsz * seq_len, dim)
        out2 = jt.bmm_transpose(x2[None, ...], weight[None, ...])[0]
        if bias is not None:
            out2 = out2 + bias
        return out2.reshape(bsz, seq_len, -1)
    raise ValueError(f"Unsupported input rank for _linear_bmm_transpose: {x_shape}")


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def execute(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = jt.abs(X).sum(dim=dim, keepdims=True) + eps
    X = X / norm
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = jt.pow(X, 2).sum(dim=dim, keepdims=True).sqrt() + eps
    X = X / norm
    return X


def func_attention(query, context, smooth=1, raw_feature_norm="softmax", eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.shape[0], query.shape[1]
    batch_size, sourceL = context.shape[0], context.shape[1]

    # Get attention
    # --> (batch, d, queryL)
    queryT = jt.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = jt.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.softmax(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = jt.contiguous(jt.transpose(attn, 1, 2))
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.softmax(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = jt.contiguous(jt.transpose(attn, 1, 2))

    # --> (batch, d, sourceL)
    contextT = jt.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = jt.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = jt.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: jt.Var, seq_len: int, bsz: int) -> jt.Var:
        return jt.contiguous(tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2))

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zero_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.l_proj.weight)
        nn.init.zero_(self.l_proj.bias)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        nn.init.zero_(self.values_v_proj.bias)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        nn.init.zero_(self.values_l_proj.bias)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        nn.init.zero_(self.out_v_proj.bias)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        nn.init.zero_(self.out_l_proj.bias)

    def execute(self, v: jt.Var, l: jt.Var, attention_mask_v: Optional[jt.Var], attention_mask_l: Optional[jt.Var]):
        """_summary_

        Args:
            v (_type_): bs, n_img, dim
            l (_type_): bs, n_text, dim
            attention_mask_v (_type_, optional): _description_. bs, n_img
            attention_mask_l (_type_, optional): _description_. bs, n_text

        Returns:
            _type_: _description_
        """
        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()
        bsz, tgt_len, _ = v.shape

        query_states: jt.Var = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states: jt.Var = key_states.view(*proj_shape)
        value_v_states: jt.Var = value_v_states.view(*proj_shape)
        value_l_states: jt.Var = value_l_states.view(*proj_shape)

        src_len = key_states.shape[1]
        attn_weights: jt.Var = jt.bmm(query_states, key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.shape}"
            )

        if self.stable_softmax_2d:
            attn_weights: jt.Var = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = jt.clamp(
                attn_weights, min_v=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = jt.clamp(
                attn_weights, max_v=50000
            )  # Do not increase 50000, data type half has quite limited range

        attn_weights_T: jt.Var = attn_weights.transpose(1, 2)
        attn_weights_l: jt.Var = attn_weights_T - attn_weights_T.max(dim=-1, keepdims=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_l = jt.clamp(
                attn_weights_l, min_v=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = jt.clamp(
                attn_weights_l, max_v=50000
            )  # Do not increase 50000, data type half has quite limited range

        # mask vison for language
        if attention_mask_v is not None:
            attention_mask_v = (
                attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights_l: jt.Var = jt.masked_fill(attn_weights_l, attention_mask_v, float("-inf"))

        attn_weights_l: jt.Var = nn.softmax(attn_weights_l, dim=-1)

        # mask language for vision
        if attention_mask_l is not None:
            attention_mask_l = ( # though it has no highlight, it can actually work out
                attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights = jt.masked_fill(attn_weights, attention_mask_l, float("-inf"))
        attn_weights_v: jt.Var = nn.softmax(attn_weights, dim=-1)

        attn_probs_v: jt.Var = nn.dropout(attn_weights_v, p=self.dropout, is_train=self.is_training)
        attn_probs_l: jt.Var = nn.dropout(attn_weights_l, p=self.dropout, is_train=self.is_training)

        attn_output_v: jt.Var = jt.bmm(attn_probs_v, value_l_states)
        attn_output_l: jt.Var = jt.bmm(attn_probs_l, value_v_states)

        if attn_output_v.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.shape}"
            )

        if attn_output_l.shape != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.shape}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = _linear_bmm_transpose(
            attn_output_v, self.out_v_proj.weight, self.out_v_proj.bias
        )
        attn_output_l = _linear_bmm_transpose(
            attn_output_l, self.out_l_proj.weight, self.out_l_proj.bias
        )

        return attn_output_v, attn_output_l


# Bi-Direction MHA (text->image, image->text)
class BiAttentionBlock(nn.Module):
    def __init__(
        self,
        v_dim: int,
        l_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        cfg=None,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # add layer scale for training stability
        self.drop_path = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v: jt.Var = init_values * jt.ones((v_dim))
        self.gamma_l: jt.Var = init_values * jt.ones((l_dim))

    def execute(self, v: jt.Var, l: jt.Var, attention_mask_v: Optional[jt.Var], attention_mask_l: Optional[jt.Var]) -> Tuple[jt.Var, jt.Var]:
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(
            v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l
        )
        assert isinstance(delta_v, jt.Var)
        assert isinstance(delta_l, jt.Var)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l

    # def execute(self, v:List[jt.Var], l, attention_mask_v=None, attention_mask_l=None)
