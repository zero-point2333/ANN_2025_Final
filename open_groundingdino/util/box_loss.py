# borrow from https://github.com/Zzh-tju/CIoU/blob/master/layers/modules/multibox_loss.py

import math
import sys
import jittor as jt


def ciou(bboxes1, bboxes2):
    """Compute CIoU distance (1 - CIoU) between two sets of boxes.
    Boxes expected in format [cx, cy, log(w), log(h)] or similar (same as original code).
    Returns a matrix of shape [N, M] where N = len(bboxes1), M = len(bboxes2).
    """
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
    inter_l = jt.maximum(center_x1 - w1 / 2,center_x2 - w2 / 2)
    inter_r = jt.minimum(center_x1 + w1 / 2,center_x2 + w2 / 2)
    inter_t = jt.maximum(center_y1 - h1 / 2,center_y2 - h2 / 2)
    inter_b = jt.minimum(center_y1 + h1 / 2,center_y2 + h2 / 2)
    inter_area = jt.clamp((inter_r - inter_l),min_v=0) * jt.clamp((inter_b - inter_t),min_v=0)

    c_l = jt.minimum(center_x1 - w1 / 2,center_x2 - w2 / 2)
    c_r = jt.maximum(center_x1 + w1 / 2,center_x2 + w2 / 2)
    c_t = jt.minimum(center_y1 - h1 / 2,center_y2 - h2 / 2)
    c_b = jt.maximum(center_y1 + h1 / 2,center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    c_diag = jt.clamp((c_r - c_l), min_v=0.0) ** 2 + jt.clamp((c_b - c_t), min_v=0.0) ** 2

    union = area1+area2-inter_area
    u = inter_diag / c_diag
    iou = inter_area / union

    v = (4.0 / (math.pi ** 2)) * jt.pow((jt.atan(w2 / h2) - jt.atan(w1 / h1)), 2)
    with jt.no_grad():
        S = (iou > 0.5).float()
        alpha = S * v / (1.0 - iou + v)

    cious = iou - u - alpha * v
    cious = jt.clamp(cious, min_v=-1.0, max_v=1.0)
    if exchange:
        cious = jt.transpose(cious)
    return 1.0 - cious


def diou(bboxes1, bboxes2):
    """Compute DIoU distance (1 - DIoU) between two sets of boxes."""
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

    inter_l = jt.maximum(center_x1 - w1 / 2,center_x2 - w2 / 2)
    inter_r = jt.minimum(center_x1 + w1 / 2,center_x2 + w2 / 2)
    inter_t = jt.maximum(center_y1 - h1 / 2,center_y2 - h2 / 2)
    inter_b = jt.minimum(center_y1 + h1 / 2,center_y2 + h2 / 2)
    inter_area = jt.clamp((inter_r - inter_l), min_v=0.0) * jt.clamp((inter_b - inter_t), min_v=0.0)

    c_l = jt.minimum(center_x1 - w1 / 2,center_x2 - w2 / 2)
    c_r = jt.maximum(center_x1 + w1 / 2,center_x2 + w2 / 2)
    c_t = jt.minimum(center_y1 - h1 / 2,center_y2 - h2 / 2)
    c_b = jt.maximum(center_y1 + h1 / 2,center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    c_diag = jt.clamp((c_r - c_l), min_v=0.0) ** 2 + jt.clamp((c_b - c_t), min_v=0.0) ** 2

    union = area1+area2-inter_area
    u = inter_diag / c_diag
    iou = inter_area / union
    dious = iou - u
    dious = jt.clamp(dious, min_v=-1.0, max_v=1.0)
    if exchange:
        dious = jt.transpose(dious)
    return 1.0 - dious


if __name__ == "__main__":
    x = jt.rand((10, 4))
    y = jt.rand((10, 4))
    cxy = ciou(x, y)
    dxy = diou(x, y)
    print(cxy.shape, dxy.shape)
