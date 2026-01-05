# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import jittor as jt


def box_cxcywh_to_xyxy(x):
    assert isinstance(x, jt.Var)
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return jt.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return jt.stack(b, dim=-1)

def box_area(boxes):
    """
    计算一组边界框（XYXY格式）的面积。

    参数:
        boxes (jt.Var): 形状为 (N, 4) 的张量，表示N个边界框。
                        每个边界框格式为 (x1, y1, x2, y2)。

    返回:
        jt.Var: 形状为 (N,) 的张量，包含每个边界框的面积。
    """
    # 确保输入是浮点数类型，避免整数相减溢出
    boxes = boxes.float()
    
    # 计算宽度和高度
    width = boxes[:, 2] - boxes[:, 0]  # x2 - x1
    height = boxes[:, 3] - boxes[:, 1] # y2 - y1
    
    # 计算面积
    area = width * height
    return area
# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = jt.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = jt.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = jt.clamp(rb - lt, min_v=0.0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert jt.all(boxes1[:, 2:] >= boxes1[:, :2]).item()
    assert jt.all(boxes2[:, 2:] >= boxes2[:, :2]).item()

    iou, union = box_iou(boxes1, boxes2)

    lt = jt.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = jt.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = jt.clamp(rb - lt, min_v=0.0)  # [N, M, 2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


# modified from torchvision to also return the union
def box_iou_pairwise(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = jt.maximum(boxes1[:, :2], boxes2[:, :2])  # [N, 2]
    rb = jt.minimum(boxes1[:, 2:], boxes2[:, 2:])  # [N, 2]

    wh = jt.clamp(rb - lt, min_v=0.0)  # [N, 2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou_pairwise(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    Input:
        - boxes1, boxes2: [N, 4], in xyxy format
    Output:
        - giou: [N]
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert jt.all(boxes1[:, 2:] >= boxes1[:, :2]).item()
    assert jt.all(boxes2[:, 2:] >= boxes2[:, :2]).item()
    assert boxes1.shape == boxes2.shape

    iou, union = box_iou_pairwise(boxes1, boxes2)  # [N]

    lt = jt.minimum(boxes1[:, :2], boxes2[:, :2])
    rb = jt.maximum(boxes1[:, 2:], boxes2[:, 2:])

    wh = jt.clamp(rb - lt, min_v=0.0)  # [N, 2]
    area = wh[:, 0] * wh[:, 1]  # [N]

    return iou - (area - union) / area


def masks_to_boxes(masks: jt.Var):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    # 使用 jt.numel() 替代 torch.numel()
    if masks.numel() == 0:
        # Jittor 张量创建使用 jt.zeros，且无需指定 device
        return jt.zeros((0, 4))

    h, w = masks.shape[-2:]

    y = jt.arange(0, h, dtype=jt.float32)
    x = jt.arange(0, w, dtype=jt.float32)
    y, x = jt.meshgrid([y, x]) 

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0] # Jittor的max返回(value, index)
    x_min = x_mask.masked_fill(jt.logical_not(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(jt.logical_not(masks.bool()), 1e8).flatten(1).min(-1)[0]
    
    return jt.stack([x_min, y_min, x_max, y_max], dim=1)


if __name__ == "__main__":
    x = jt.rand(5, 4)
    y = jt.rand(3, 4)
    iou, union = box_iou(x, y)
    # import ipdb

    # ipdb.set_trace()
