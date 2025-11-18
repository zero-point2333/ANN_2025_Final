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

import jittor as jt
import jittor.nn as nn
from jittor.models import resnet

from groundingdino.util.misc import NestedTensor, clean_state_dict, is_main_process

from .position_encoding import build_position_encoding
from .swin_transformer import build_swin_transformer


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.weight = jt.ones(n)
        self.bias = jt.zeros(n)
        self.running_mean = jt.zeros(n)
        self.running_var = jt.ones(n)
        
        # 在Jittor中设置为不计算梯度
        self.weight.stop_grad()
        self.bias.stop_grad()
        self.running_mean.stop_grad()
        self.running_var.stop_grad()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def execute(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class IntermediateLayerGetter(nn.Module):
    """
    Jittor版本的IntermediateLayerGetter
    用于从backbone中提取多层特征
    """
    
    def __init__(self, model, return_layers):
        super().__init__()
        self.model = model
        self.return_layers = return_layers

    def execute(self, x):
        out = {}
        
        # Stem部分
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        # 逐层处理
        layers = [
            ('layer1', self.model.layer1),
            ('layer2', self.model.layer2),
            ('layer3', self.model.layer3),
            ('layer4', self.model.layer4)
        ]
        
        for layer_name, layer_func in layers:
            x = layer_func(x)
            if layer_name in self.return_layers:
                out_key = self.return_layers[layer_name]
                out[out_key] = x
                
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
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.stop_grad()  # Jittor中使用stop_grad

        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update(
                {"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)}
            )

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def execute(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # Jittor的interpolate用法
            mask = nn.interpolate(m.float().unsqueeze(0), size=x.shape[-2:]).to(jt.bool).squeeze(0)
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
        # 使用Jittor的resnet模型
        if name == "resnet50":
            backbone = resnet.resnet50(pretrained=is_main_process())
        elif name == "resnet101":
            backbone = resnet.resnet101(pretrained=is_main_process())
        elif name == "resnet18":
            backbone = resnet.resnet18(pretrained=is_main_process())
        elif name == "resnet34":
            backbone = resnet.resnet34(pretrained=is_main_process())
        else:
            raise NotImplementedError("Why you can get here with name {}".format(name))
        
        # 处理dilation参数 - 需要根据Jittor的resnet实现进行调整
        if dilation:
            # 这里需要根据Jittor resnet的具体实现来修改dilation
            # 可能需要手动修改相关层的配置
            pass
            
        assert name not in ("resnet18", "resnet34"), "Only resnet50 and resnet101 are available."
        assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
        num_channels_all = [256, 512, 1024, 2048]
        num_channels = num_channels_all[4 - len(return_interm_indices):]
        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)


class Joiner(nn.Module):
    def __init__(self, backbone, position_embedding):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding

    def execute(self, tensor_list: NestedTensor):
        xs = self.backbone(tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self.position_embedding(x).to(x.tensors.dtype))

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
    train_backbone = True
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
    
    if hasattr(args, "backbone_freeze_keywords"):
        args.backbone_freeze_keywords
    use_checkpoint = getattr(args, "use_checkpoint", False)

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
        "swin_T_224_1k",
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

        bb_num_channels = backbone.num_features[4 - len(return_interm_indices):]
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
    
    return model