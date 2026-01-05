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
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1,
                 cost_giou: float = 1, focal_alpha=0.25):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0

        self.focal_alpha = focal_alpha

    @jt.no_grad()
    def execute(self, outputs, targets, label_map):
        sizes = [len(v["boxes"]) for v in targets]
        if sum(sizes) == 0:
            return [(jt.array([], dtype=jt.int64),
                     jt.array([], dtype=jt.int64)) for _ in targets]

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # flatten
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = jt.concat([v["labels"] for v in targets]).int()
        tgt_bbox = jt.concat([v["boxes"] for v in targets])

        alpha = self.focal_alpha
        gamma = 2.0

        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())

        # ✅【关键对齐点】严格等价 PyTorch
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_bbox = cdist(out_bbox, tgt_bbox, p=1)

        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )

        C = self.cost_bbox * cost_bbox + \
            self.cost_class * cost_class + \
            self.cost_giou * cost_giou

        C = C.view(bs, num_queries, -1)

        invalid = jt.isnan(C) | jt.isinf(C)
        C = jt.where(invalid, jt.zeros_like(C), C)

        indices = []
        C_split = C.split(sizes, -1)

        for i, (c_tensor, count) in enumerate(zip(C_split, sizes)):
            if count == 0:
                indices.append((jt.array([], dtype=jt.int64),
                                jt.array([], dtype=jt.int64)))
                continue

            weight_mat = c_tensor[i]
            weight_mat.sync()
            weight_np = np.nan_to_num(weight_mat.numpy())

            idx_i = np.argmin(weight_np, axis=0)
            idx_j = np.arange(count)

            indices.append((idx_i, idx_j))

        return [(jt.array(i, dtype=jt.int64),
                 jt.array(j, dtype=jt.int64)) for i, j in indices]



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