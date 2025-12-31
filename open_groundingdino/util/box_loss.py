# borrow from https://github.com/Zzh-tju/CIoU/blob/master/layers/modules/multibox_loss.py

import math
import sys
import jittor as jt
from jittor import Var

def _to_jt_var(x):
    if isinstance(x, Var):
        return x
    torch = sys.modules.get("torch", None)
    if torch is not None and isinstance(x, getattr(torch, "Tensor", ())):
        return jt.array(x.detach().cpu().numpy())
    return jt.array(x)


def ciou(bboxes1, bboxes2):
    """Compute CIoU distance (1 - CIoU) between two sets of boxes.
    Boxes expected in format [cx, cy, log(w), log(h)] or similar (same as original code).
    Returns a matrix of shape [N, M] where N = len(bboxes1), M = len(bboxes2).
    """
    bboxes1 = _to_jt_var(bboxes1)
    bboxes2 = _to_jt_var(bboxes2)
    bboxes1 = jt.sigmoid(bboxes1)
    bboxes2 = jt.sigmoid(bboxes2)

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = jt.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if rows > cols:
        bboxes1, bboxes2 = bboxes2, bboxes1
        rows, cols = cols, rows
        cious = jt.zeros((rows, cols))
        exchange = True

    w1 = jt.exp(bboxes1[:, 2])
    h1 = jt.exp(bboxes1[:, 3])
    w2 = jt.exp(bboxes2[:, 2])
    h2 = jt.exp(bboxes2[:, 3])
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = bboxes1[:, 0]
    center_y1 = bboxes1[:, 1]
    center_x2 = bboxes2[:, 0]
    center_y2 = bboxes2[:, 1]

    # make pairwise by expanding dims
    cx1 = center_x1[:, None]
    cy1 = center_y1[:, None]
    cx2 = center_x2[None, :]
    cy2 = center_y2[None, :]
    w1_e = w1[:, None]
    h1_e = h1[:, None]
    w2_e = w2[None, :]
    h2_e = h2[None, :]

    inter_l = jt.maximum(cx1 - w1_e / 2, cx2 - w2_e / 2)
    inter_r = jt.minimum(cx1 + w1_e / 2, cx2 + w2_e / 2)
    inter_t = jt.maximum(cy1 - h1_e / 2, cy2 - h2_e / 2)
    inter_b = jt.minimum(cy1 + h1_e / 2, cy2 + h2_e / 2)
    inter_area = jt.clamp((inter_r - inter_l), min_v=0.0) * jt.clamp((inter_b - inter_t), min_v=0.0)

    c_l = jt.minimum(cx1 - w1_e / 2, cx2 - w2_e / 2)
    c_r = jt.maximum(cx1 + w1_e / 2, cx2 + w2_e / 2)
    c_t = jt.minimum(cy1 - h1_e / 2, cy2 - h2_e / 2)
    c_b = jt.maximum(cy1 + h1_e / 2, cy2 + h2_e / 2)

    inter_diag = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
    c_diag = jt.clamp((c_r - c_l), min_v=0.0) ** 2 + jt.clamp((c_b - c_t), min_v=0.0) ** 2

    union = area1[:, None] + area2[None, :] - inter_area
    u = inter_diag / c_diag
    iou = inter_area / union

    v = (4.0 / (math.pi ** 2)) * jt.pow((jt.atan(w2_e / h2_e) - jt.atan(w1_e / h1_e)), 2)
    with jt.no_grad():
        S = (iou > 0.5).float()
        alpha = S * v / (1.0 - iou + v)

    cious = iou - u - alpha * v
    cious = jt.clamp(cious, min_v=-1.0, max_v=1.0)
    if exchange:
        cious = cious.T
    return 1.0 - cious


def diou(bboxes1, bboxes2):
    """Compute DIoU distance (1 - DIoU) between two sets of boxes."""
    bboxes1 = _to_jt_var(bboxes1)
    bboxes2 = _to_jt_var(bboxes2)
    bboxes1 = jt.sigmoid(bboxes1)
    bboxes2 = jt.sigmoid(bboxes2)

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = jt.zeros((rows, cols))
    if rows * cols == 0:
        return dious
    exchange = False
    if rows > cols:
        bboxes1, bboxes2 = bboxes2, bboxes1
        rows, cols = cols, rows
        dious = jt.zeros((rows, cols))
        exchange = True

    w1 = jt.exp(bboxes1[:, 2])
    h1 = jt.exp(bboxes1[:, 3])
    w2 = jt.exp(bboxes2[:, 2])
    h2 = jt.exp(bboxes2[:, 3])
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = bboxes1[:, 0]
    center_y1 = bboxes1[:, 1]
    center_x2 = bboxes2[:, 0]
    center_y2 = bboxes2[:, 1]

    cx1 = center_x1[:, None]
    cy1 = center_y1[:, None]
    cx2 = center_x2[None, :]
    cy2 = center_y2[None, :]
    w1_e = w1[:, None]
    h1_e = h1[:, None]
    w2_e = w2[None, :]
    h2_e = h2[None, :]

    inter_l = jt.maximum(cx1 - w1_e / 2, cx2 - w2_e / 2)
    inter_r = jt.minimum(cx1 + w1_e / 2, cx2 + w2_e / 2)
    inter_t = jt.maximum(cy1 - h1_e / 2, cy2 - h2_e / 2)
    inter_b = jt.minimum(cy1 + h1_e / 2, cy2 + h2_e / 2)
    inter_area = jt.clamp((inter_r - inter_l), min_v=0.0) * jt.clamp((inter_b - inter_t), min_v=0.0)

    c_l = jt.minimum(cx1 - w1_e / 2, cx2 - w2_e / 2)
    c_r = jt.maximum(cx1 + w1_e / 2, cx2 + w2_e / 2)
    c_t = jt.minimum(cy1 - h1_e / 2, cy2 - h2_e / 2)
    c_b = jt.maximum(cy1 + h1_e / 2, cy2 + h2_e / 2)

    inter_diag = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
    c_diag = jt.clamp((c_r - c_l), min_v=0.0) ** 2 + jt.clamp((c_b - c_t), min_v=0.0) ** 2

    union = area1[:, None] + area2[None, :] - inter_area
    u = inter_diag / c_diag
    iou = inter_area / union
    dious = iou - u
    dious = jt.clamp(dious, min_v=-1.0, max_v=1.0)
    if exchange:
        dious = dious.T
    return 1.0 - dious


if __name__ == "__main__":
    x = jt.rand((10, 4))
    y = jt.rand((10, 4))
    cxy = ciou(x, y)
    dxy = diou(x, y)
    print(cxy.shape, dxy.shape)
