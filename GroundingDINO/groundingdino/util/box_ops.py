# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import jittor as jt


def box_cxcywh_to_xyxy(x):
    # x: [..., 4] in (cx, cy, w, h)
    x_c = x[..., 0]
    y_c = x[..., 1]
    w = x[..., 2]
    h = x[..., 3]
    b = [
        x_c - 0.5 * w,
        y_c - 0.5 * h,
        x_c + 0.5 * w,
        y_c + 0.5 * h,
    ]
    return jt.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    # x: [..., 4] in (x0, y0, x1, y1)
    x0 = x[..., 0]
    y0 = x[..., 1]
    x1 = x[..., 2]
    y1 = x[..., 3]
    b = [
        (x0 + x1) / 2.0,
        (y0 + y1) / 2.0,
        (x1 - x0),
        (y1 - y0),
    ]
    return jt.stack(b, dim=-1)


def box_area(boxes):
    """
    Compute the area of a set of bounding boxes.

    The boxes are expected to be in (x0, y0, x1, y1) format.

    Args:
        boxes: jt.Var with shape [..., 4].

    Returns:
        jt.Var with shape [...] containing the area of each box.
    """
    w = jt.clamp(boxes[..., 2] - boxes[..., 0], min_v=0.0)
    h = jt.clamp(boxes[..., 3] - boxes[..., 1], min_v=0.0)
    return w * h


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    """
    Args:
        boxes1: [N, 4] in xyxy
        boxes2: [M, 4] in xyxy

    Returns:
        iou:   [N, M]
        union: [N, M]
    """
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

    The boxes should be in [x0, y0, x1, y1] format.

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
    """
    Args:
        boxes1: [N, 4] in xyxy
        boxes2: [N, 4] in xyxy

    Returns:
        iou:   [N]
        union: [N]
    """
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


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks.

    The masks should be in format [N, H, W] where N is the number of masks,
    (H, W) are the spatial dimensions.

    Returns a [N, 4] tensor, with the boxes in xyxy format.
    """
    # 与原实现保持语义一致：当没有 mask 时返回空 tensor
    if masks.shape[0] == 0:
        return jt.zeros((0, 4))

    h, w = masks.shape[-2:]
    n = masks.shape[0]

    # 坐标网格（与原来 torch.arange + meshgrid 等价）
    ys = jt.arange(h).float32().reshape(1, h, 1)  # [1, H, 1]
    xs = jt.arange(w).float32().reshape(1, 1, w)  # [1, 1, W]

    mask_bool = masks.bool()  # [N, H, W]

    # x 最大值：mask 处取坐标，其余位置为 0
    x_max_tensor = jt.where(mask_bool, xs, jt.zeros_like(xs))
    x_max = x_max_tensor.reshape(n, -1).max(dim=1)

    # x 最小值：mask 处取坐标，其余位置为一个很大的数
    large = 1e8
    x_min_tensor = jt.where(mask_bool, xs, jt.full(xs.shape, large))
    x_min = x_min_tensor.reshape(n, -1).min(dim=1)

    # y 最大值
    y_max_tensor = jt.where(mask_bool, ys, jt.zeros_like(ys))
    y_max = y_max_tensor.reshape(n, -1).max(dim=1)

    # y 最小值
    y_min_tensor = jt.where(mask_bool, ys, jt.full(ys.shape, large))
    y_min = y_min_tensor.reshape(n, -1).min(dim=1)

    return jt.stack([x_min, y_min, x_max, y_max], dim=1)


if __name__ == "__main__":
    x = jt.rand(5, 4)
    y = jt.rand(3, 4)
    iou, union = box_iou(x, y)
    import ipdb

    ipdb.set_trace()