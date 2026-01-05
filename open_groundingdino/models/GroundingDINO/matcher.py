# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modules to compute the matching cost and solve the corresponding LSAP.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


import os
import numpy as np
from scipy.optimize import linear_sum_assignment

import jittor as jt
from jittor import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.utils import cdist

# [修改点 1] 引入 log_text 用于输出日志
from util.debug_tools import log_text

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha = 0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @jt.no_grad()
    def execute(self, outputs, targets, label_map):
        # 1. 预先检查 sizes
        sizes = [len(v["boxes"]) for v in targets]
        if sum(sizes) == 0:
            log_text("[Matcher Warning] No targets found in current batch! Returning empty matches.")
            return [(jt.array([], dtype=jt.int64), jt.array([], dtype=jt.int64)) for _ in targets]

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 2. Flatten
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  

        # 3. Concat targets
        # [优化] 确保 reshape 和类型正确
        tgt_ids = jt.concat([v["labels"] for v in targets]).int()
        tgt_bbox = jt.concat([v["boxes"] for v in targets])

        # 4. Compute Classification Cost (重点修改区域)
        alpha = self.focal_alpha
        gamma = 2.0

        # 取出对应的 label_map
        new_label_map = label_map[tgt_ids] # shape: [Total_Targets, Num_Classes]

        # 计算 Focal Loss 组件
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        
        cost_bbox = cdist(out_bbox, tgt_bbox, p=1)

        cost_class=[]
        for idx_map in new_label_map:       
            idx_map = idx_map / idx_map.sum()
            cost_class.append(pos_cost_class @ idx_map - neg_cost_class@ idx_map)
        if cost_class:
            cost_class=jt.stack(cost_class,dim=0).transpose()
        else:
            cost_class=jt.zeros_like(cost_bbox)

        # 6. Compute GIoU Cost
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        invalid = jt.isnan(C) | jt.isinf(C)
        C = jt.where(invalid, jt.zeros_like(C), C)

        sizes = [len(v["boxes"]) for v in targets]
        try:
            indices = [linear_sum_assignment(c[i].numpy()) for i, c in enumerate(C.split(sizes, -1))]
        except:
            print("warning: use SimpleMinsumMatcher")
            indices = []
            for i, (c, _size) in enumerate(zip(C.split(sizes, -1), sizes)):
                weight_mat = c[i]
                idx_i = weight_mat.min(0)[1]
                idx_j = jt.arange(_size)
                indices.append((idx_i, idx_j))
        return [(jt.array(i, dtype=jt.int64), jt.array(j, dtype=jt.int64)) for i, j in indices]


class SimpleMinsumMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha = 0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @jt.no_grad()
    def execute(self, outputs, targets, label_map):
        # [修改点 2] 预先检查 sizes，处理整个 batch 都为空的极端情况
        sizes = [len(v["boxes"]) for v in targets]
        if sum(sizes) == 0:
            log_text("[Matcher Warning] No targets found in current batch! Returning empty matches.")
            return [(jt.array([], dtype=jt.int64), jt.array([], dtype=jt.int64)) for _ in targets]

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = jt.concat([v["labels"] for v in targets]).int()
        tgt_bbox = jt.concat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0

        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())

        cost_bbox = cdist(out_bbox, tgt_bbox, p=1)

        # -------------------------------------------------------------------------
        # [重要修改] 将 Python 循环改为矩阵运算，极大降低编译器压力，防止 Segfault
        # -------------------------------------------------------------------------
        # 原逻辑：逐个 Target 取出 label_map 向量，归一化后做点积
        # 新逻辑：对整个 new_label_map 矩阵归一化，然后直接矩阵相乘
        
        # 1. 归一化 (注意 dim=1, 加上 eps 防止除零)
        new_label_map = label_map[tgt_ids]
        new_label_map = new_label_map / (new_label_map.sum(dim=1, keepdims=True) + 1e-6)
        
        # 2. 矩阵乘法替代循环
        # pos_cost_class: [Total_Queries, Num_Classes]
        # new_label_map.T: [Num_Classes, Total_Targets]
        # 结果: [Total_Queries, Total_Targets]
        cost_class = (pos_cost_class @ new_label_map.transpose(0, 1)) - \
                     (neg_cost_class @ new_label_map.transpose(0, 1))
        
        # -------------------------------------------------------------------------

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        assert isinstance(C, jt.Var)
        C = C.view(bs, num_queries, -1)
        
        # 处理无效值
        invalid = jt.isnan(C) | jt.isinf(C)
        C = jt.where(invalid, jt.zeros_like(C), C)

        # 校验尺寸逻辑 (保持你的原样)
        total_targets = int(sum(sizes))
        if total_targets != int(C.shape[-1]):
            log_text(f"[Matcher Error] Shape mismatch: Total targets {total_targets} vs Cost Matrix width {C.shape[-1]}")
            # ... (Debug print 保持不变) ...

        C_split = C.split(sizes, -1)
        
        # [修改点 3] 显式循环处理 (保持你的修改，非常完美)
        indices = []
        for i, (c_tensor, count) in enumerate(zip(C_split, sizes)):
            # 3.1 检查单张图是否没有目标
            if count == 0:
                log_text(f"[Matcher Info] Batch index {i} has NO targets. Skipping assignment.")
                indices.append((np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
                continue

            # 3.2 提取当前 batch 的 Cost Matrix
            weight_mat = c_tensor[i] 
            
            # 3.3 强制同步 (关键步骤！)
            weight_mat.sync()
            
            # 3.4 转换为 numpy
            weight_np = weight_mat.numpy()
            
            # 3.5 防御性编程
            weight_np = np.nan_to_num(weight_np, posinf=1e10, neginf=-1e10)

            try:
                # 正常匹配
                indices.append(linear_sum_assignment(weight_np))
            except Exception as e:
                log_text(f"[Matcher Failed] Scipy match failed at batch {i}. Error: {e}")
                log_text(f"  - Weight shape: {weight_np.shape}, Has Nan: {np.isnan(weight_np).any()}")
                # 降级方案：Simple MinSum
                idx_i = np.argmin(weight_np, axis=0)
                idx_j = np.arange(count)
                indices.append((idx_i, idx_j))

        return [(jt.array(i, dtype=jt.int64), jt.array(j, dtype=jt.int64)) for i, j in indices]


def build_matcher(args):
    assert args.matcher_type in ['HungarianMatcher', 'SimpleMinsumMatcher'], "Unknown args.matcher_type: {}".format(args.matcher_type)
    if args.matcher_type == 'HungarianMatcher':
        print("Use Hungarian Matcher")
        return HungarianMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha
        )
    elif args.matcher_type == 'SimpleMinsumMatcher':
        print("Use Simple Minsum Matcher")
        return SimpleMinsumMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha
        )    
    else:
        raise NotImplementedError("Unknown args.matcher_type: {}".format(args.matcher_type))