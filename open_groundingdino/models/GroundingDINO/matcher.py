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


import jittor as jt
from jittor import nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import os

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


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

    def execute(self, outputs, targets, label_map):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = jt.concat([v["labels"] for v in targets])
        tgt_bbox = jt.concat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0

        new_label_map = label_map[tgt_ids.numpy()]
        new_label_map = jt.array(new_label_map)  # 转换为jittor tensor

        # 计算分类代价
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        
        # 确保new_label_map在同一设备上
        if new_label_map.place is not None and new_label_map.place != out_prob.place:
            new_label_map = new_label_map.to(out_prob.place)
        
        # 计算bbox距离代价
        # Jittor的cdist函数需要从jittor.nn导入
        from jittor.nn import cdist
        cost_bbox = cdist(out_bbox, tgt_bbox, p=1)

        # 计算分类代价
        cost_class = []
        for idx_map in new_label_map:       
            idx_map = idx_map / idx_map.sum()
            cost_class.append(pos_cost_class @ idx_map - neg_cost_class @ idx_map)
        
        if len(cost_class) > 0:
            cost_class = jt.stack(cost_class, dim=0).transpose(0, 1)
        else:
            cost_class = jt.zeros_like(cost_bbox)

        # Compute the giou cost between boxes
        out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
        tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
        cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1)
        
        # 处理NaN和Inf
        C = jt.ternary(jt.isnan(C), jt.zeros_like(C), C)
        C = jt.ternary(jt.isinf(C), jt.zeros_like(C), C)
        
        # 转换为numpy用于scipy的linear_sum_assignment
        C_np = C.numpy()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        
        try:
            for i, c in enumerate(C_np):
                # 分割出当前batch的cost矩阵
                c_split = np.split(c, np.cumsum(sizes)[:-1], axis=-1)[i]
                row_ind, col_ind = linear_sum_assignment(c_split)
                indices.append((jt.array(row_ind, dtype=jt.int64), 
                               jt.array(col_ind, dtype=jt.int64)))
        except:
            print("warning: use SimpleMinsumMatcher")
            indices = []
            for i, (c, _size) in enumerate(zip(C.split(sizes, dim=-1), sizes)):
                weight_mat = c[i]
                idx_i = weight_mat.min(dim=0)[1]
                idx_j = jt.arange(_size, dtype=jt.int64)
                indices.append((idx_i, idx_j))
        
        return indices


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

    def execute(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = jt.concat([v["labels"] for v in targets])
        tgt_bbox = jt.concat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        
        # 通过索引选择对应的分类代价
        tgt_ids_expanded = tgt_ids.view(1, -1).expand(out_prob.shape[0], -1)
        pos_selected = jt.zeros_like(out_prob)
        neg_selected = jt.zeros_like(out_prob)
        
        # 使用gather操作
        batch_indices = jt.arange(out_prob.shape[0]).view(-1, 1).expand(-1, tgt_ids.shape[0])
        pos_selected = pos_cost_class[batch_indices, tgt_ids_expanded]
        neg_selected = neg_cost_class[batch_indices, tgt_ids_expanded]
        cost_class = pos_selected - neg_selected

        # Compute the L1 cost between boxes
        from jittor.nn import cdist
        cost_bbox = cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute the giou cost between boxes            
        out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
        tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
        cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1)

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        
        for i, (c, _size) in enumerate(C.split(sizes, dim=-1)):
            weight_mat = c[i]
            idx_i = weight_mat.min(dim=0)[1]
            idx_j = jt.arange(_size, dtype=jt.int64)
            indices.append((idx_i, idx_j))

        return indices


def build_matcher(args):
    assert args.matcher_type in ['HungarianMatcher', 'SimpleMinsumMatcher'], \
        f"Unknown args.matcher_type: {args.matcher_type}"
    
    if args.matcher_type == 'HungarianMatcher':
        return HungarianMatcher(
            cost_class=args.set_cost_class, 
            cost_bbox=args.set_cost_bbox, 
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha
        )
    elif args.matcher_type == 'SimpleMinsumMatcher':
        return SimpleMinsumMatcher(
            cost_class=args.set_cost_class, 
            cost_bbox=args.set_cost_bbox, 
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha
        )    
    else:
        raise NotImplementedError(f"Unknown args.matcher_type: {args.matcher_type}")