# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""

from typing import Dict, List
from collections import OrderedDict

import jittor as jt
import jittor.nn as nn
from jittor.models import resnet

from util.misc import NestedTensor, clean_state_dict, is_main_process

from .position_encoding import PositionEmbeddingLearned, PositionEmbeddingSineHW, build_position_encoding
from .swin_transformer import build_swin_transformer

import jittor as jt
from jittor import nn

class FrozenBatchNorm2d(nn.Module):
    """
    Jittor version of FrozenBatchNorm2d.
    Fixed: Uses jt.no_grad() to prevent graph overhead for constant calculations.
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        # 初始化参数
        self.weight = jt.ones(n)
        self.bias = jt.zeros(n)
        self.running_mean = jt.zeros(n)
        self.running_var = jt.ones(n)
        
        # 确保这些变量永远不计算梯度
        self.weight.stop_grad()
        self.bias.stop_grad()
        self.running_mean.stop_grad()
        self.running_var.stop_grad()

    def execute(self, x):
        # 1. 使用 no_grad 上下文，防止 scale 和 bias 的计算生成无用的计算图节点
        with jt.no_grad():
            w = self.weight.reshape(1, -1, 1, 1)
            b = self.bias.reshape(1, -1, 1, 1)
            rv = self.running_var.reshape(1, -1, 1, 1)
            rm = self.running_mean.reshape(1, -1, 1, 1)
            eps = 1e-5
            scale = w * (rv + eps).rsqrt()
            bias = b - rm * scale
        
        # 2. 只有这一步涉及输入 x，需要保留在计算图中
        return x * scale + bias


class IntermediateLayerGetter(nn.Module):
    """
    Module wrapper that returns intermediate layers from a model.
    Adapted for Jittor from torchvision.models._utils.IntermediateLayerGetter.
    """
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_indices: list,
    ):
        super().__init__()
        # 冻结参数逻辑保持不变，这部分是对的
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.stop_grad() 

        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update(
                {"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)}
            )

        # 假设 IntermediateLayerGetter 已经适配了 Jittor
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def execute(self, tensor_list: NestedTensor):
        # 修改点1：不要随意使用 jt.array() 
        # 假设 tensor_list.tensors 已经是 Jittor Var (从 DataLoader 出来通常就是)
        # 如果不确定，可以用 x = tensor_list.tensors; if not isinstance(x, jt.Var): x = jt.array(x)
        xs = self.body(tensor_list.tensors)
        
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            
            # 修改点2：Mask 的插值与类型转换
            # 1. 确保 m 是 var 并转为 float (interpolate 需要 float)
            # 2. unsqueeze(1) 增加 channel 维度: (B, H, W) -> (B, 1, H, W)
            if not isinstance(m, jt.Var):
                m = jt.array(m)
            
            # 使用 nearest 插值保持 mask 的二值特性
            mask = nn.interpolate(m.float().unsqueeze(1), size=x.shape[-2:], mode="nearest")
            
            # 3. 移除 channel 维度并转回 bool/int
            mask = mask.squeeze(1).bool()
            
            # 修改点3：关键！停止梯度
            # Mask 只是位置编码的辅助信息，不需要参与反向传播
            mask = mask.stop_grad()
            
            # 修改点4：致命修复！
            # 绝对不要在这里调用 .numpy()
            # 必须将 Jittor Var 传给 NestedTensor，以保持计算图连通
            out[name] = NestedTensor(x, mask)
            
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        dilation: bool,
        return_interm_indices: list,
        batch_norm=FrozenBatchNorm2d,
    ):
        if name in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            if name == "resnet18":
                rn = resnet.resnet18
            elif name == "resnet34":
                rn = resnet.resnet34
            elif name == "resnet50":
                rn = resnet.resnet50
            else:
                rn = resnet.resnet101
            backbone = rn(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(),
                norm_layer=batch_norm,
            )
        else:
            raise NotImplementedError("Why you can get here with name {}".format(name))
        
            
        assert name not in ("resnet18", "resnet34"), "Only resnet50 and resnet101 are available."
        assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
        num_channels_all = [256, 512, 1024, 2048]
        num_channels = num_channels_all[4 - len(return_interm_indices) :]
        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)


class Joiner(nn.Module):
    def __init__(self, backbone, position_embedding):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding

    def __getitem__(self, index):
        if index == 0:
            return self.backbone
        elif index == 1:
            return self.position_embedding
        else:
            raise TypeError("Wrong Index Called")

    def execute(self, tensor_list: NestedTensor):
        xs = self.backbone(tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(jt.type_as(self.position_embedding(x), x.tensors))

        return out, pos


def build_backbone(args):
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone:
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords:
        - use_checkpoint: for swin only for now
    
    """
    position_embedding = build_position_encoding(args)
    assert isinstance(position_embedding, (PositionEmbeddingSineHW, PositionEmbeddingLearned))
    train_backbone = True
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices: list = args.return_interm_indices
    assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
    
    args.backbone_freeze_keywords # unused
    use_checkpoint: bool = getattr(args, "use_checkpoint", False)

    if args.backbone in ["resnet50", "resnet101"]:
        backbone = Backbone(
            args.backbone,
            train_backbone,
            args.dilation,
            return_interm_indices,
            batch_norm=FrozenBatchNorm2d,
        )
        bb_num_channels = backbone.num_channels
    elif args.backbone in [
        "swin_T_224_1k", # we use this
        "swin_B_224_22k",
        "swin_B_384_22k",
        "swin_L_224_22k",
        "swin_L_384_22k",
    ]:
        pretrain_img_size = int(args.backbone.split("_")[-2])
        backbone = build_swin_transformer(
            args.backbone,
            pretrain_img_size=pretrain_img_size,
            out_indices=tuple(return_interm_indices),
            dilation=False,
            use_checkpoint=use_checkpoint,
        )

        bb_num_channels = backbone.num_features[4 - len(return_interm_indices) :]
    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))

    assert len(bb_num_channels) == len(
        return_interm_indices
    ), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"

    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels
    assert isinstance(
        bb_num_channels, List
    ), "bb_num_channels is expected to be a List but {}".format(type(bb_num_channels))
    # import ipdb; ipdb.set_trace()
    return model
