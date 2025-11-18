# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import os
import random

import PIL
from PIL import Image, ImageOps
import numpy as np
import jittor as jt


def get_random_crop_params(img, output_size):
    """Return parameters (top, left, height, width) for a random crop.

    `output_size` may be an int or a sequence (height, width).
    """
    if isinstance(output_size, (list, tuple)):
        th, tw = output_size
    else:
        th = tw = output_size
    w, h = img.size
    if w == tw and h == th:
        return 0, 0, h, w
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw


def to_jt_tensor(pic):
    """Convert a PIL Image or numpy array to a jittor CHW float tensor in [0,1]."""
    if isinstance(pic, PIL.Image.Image):
        arr = np.array(pic)
    else:
        arr = np.asarray(pic)
    if arr.ndim == 2:
        # grayscale H,W -> 1,H,W
        arr = np.expand_dims(arr, axis=2)
    # arr is H,W,C -> transpose to C,H,W
    if arr.shape[2] <= 4 and arr.shape[0] >= 1 and arr.shape[0] != arr.shape[2]:
        arr = arr.transpose(2, 0, 1)
    arr = arr.astype(np.float32) / 255.0
    return jt.array(arr)


def normalize_tensor(image, mean, std):
    """Normalize a tensor (jittor Var or numpy array) by mean/std per channel.

    Expects input in CHW format.
    """
    if not isinstance(image, jt.Var):
        image = to_jt_tensor(image)
    mean_arr = jt.array(mean, dtype=jt.float32).reshape(-1, 1, 1)
    std_arr = jt.array(std, dtype=jt.float32).reshape(-1, 1, 1)
    return (image - mean_arr) / std_arr

from groundingdino.util.box_ops import box_xyxy_to_cxcywh
from groundingdino.util.misc import interpolate


def crop(image, target, region):
    # region: top, left, height, width
    top, left, height, width = region
    cropped_image = image.crop((left, top, left + width, top + height))

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = jt.array([h, w], dtype=jt.float32)

    fields = ["labels", "area", "iscrowd", "positive_map"]

    if "boxes" in target:
        boxes = target["boxes"]
        boxes = jt.array(boxes) if not isinstance(boxes, jt.Var) else boxes
        max_size = jt.array([w, h], dtype=jt.float32)
        cropped_boxes = boxes - jt.array([j, i, j, i], dtype=jt.float32)
        cropped_boxes = jt.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = jt.maximum(cropped_boxes, 0)
        diff = cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]
        area = diff[:, 0] * diff[:, 1]
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        masks = target["masks"]
        # masks assumed to be (N, H, W) and support slicing
        target["masks"] = masks[:, i : i + h, j : j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target["boxes"].reshape(-1, 2, 2)
            if not isinstance(cropped_boxes, jt.Var):
                cropped_boxes = jt.array(cropped_boxes)
            keep = (cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :]).all(1)
        else:
            masks = target["masks"]
            if isinstance(masks, jt.Var):
                keep = masks.reshape(masks.shape[0], -1).any(1)
            else:
                keep = masks.reshape(masks.shape[0], -1).any(1)

        for field in fields:
            if field in target:
                target[field] = target[field][keep]

    if os.environ.get("IPDB_SHILONG_DEBUG", None) == "INFO":
        # for debug and visualization only.
        if "strings_positive" in target:
            if isinstance(keep, jt.Var):
                keep_np = keep.numpy().tolist()
            else:
                keep_np = list(keep)
            target["strings_positive"] = [_i for _i, _j in zip(target["strings_positive"], keep_np) if _j]

    return cropped_image, target


def hflip(image, target):
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = jt.array(boxes) if not isinstance(boxes, jt.Var) else boxes
        boxes = boxes[:, [2, 1, 0, 3]] * jt.array([-1, 1, -1, 1], dtype=jt.float32) + jt.array([w, 0, w, 0], dtype=jt.float32)
        target["boxes"] = boxes

    if "masks" in target:
        masks = target["masks"]
        if isinstance(masks, jt.Var):
            target["masks"] = masks.flip(-1)
        else:
            target["masks"] = np.flip(masks, axis=-1).copy()

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    # PIL expects (width, height)
    rescaled_image = image.resize(size[::-1], resample=Image.BILINEAR)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = jt.array(boxes) if not isinstance(boxes, jt.Var) else boxes
        scaled_boxes = boxes * jt.array([ratio_width, ratio_height, ratio_width, ratio_height], dtype=jt.float32)
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        area = jt.array(area) if not isinstance(area, jt.Var) else area
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = jt.array([h, w], dtype=jt.float32)

    if "masks" in target:
        masks = target["masks"]
        # rely on project interpolate (which is already jittor-aware)
        target["masks"] = (interpolate(masks[:, None].astype(jt.float32), size, mode="nearest")[:, 0] > 0.5)

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = ImageOps.expand(image, border=(0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = jt.array(padded_image.size[::-1], dtype=jt.float32)
    if "masks" in target:
        masks = target["masks"]
        # masks shape (N, H, W)
        if isinstance(masks, jt.Var):
            masks_np = masks.numpy()
            padded = np.pad(masks_np, ((0, 0), (0, padding[1]), (0, padding[0])), mode="constant")
            target["masks"] = jt.array(padded)
        else:
            target["masks"] = np.pad(masks, ((0, 0), (0, padding[1]), (0, padding[0])), mode="constant")
    return padded_image, target


class ResizeDebug(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return resize(img, target, self.size)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = get_random_crop_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, respect_boxes: bool = False):
        # respect_boxes:    True to keep all boxes
        #                   False to tolerence box filter
        self.min_size = min_size
        self.max_size = max_size
        self.respect_boxes = respect_boxes

    def __call__(self, img: PIL.Image.Image, target: dict):
        init_boxes = len(target["boxes"])
        max_patience = 10
        for i in range(max_patience):
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            region = get_random_crop_params(img, [h, w])
            result_img, result_target = crop(img, target, region)
            if (
                not self.respect_boxes
                or len(result_target["boxes"]) == init_boxes
                or i == max_patience - 1
            ):
                return result_img, result_target
        return result_img, result_target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return to_jt_tensor(img), target


class RandomErasing(object):
    def __init__(self, *args, **kwargs):
        # implement a simple RandomErasing compatible with tensors or PIL images
        self.args = args
        self.kwargs = kwargs

    def __call__(self, img, target):
        # support PIL Image, numpy array (H,W,C) or jt.Var (C,H,W)
        if isinstance(img, PIL.Image.Image):
            arr = np.array(img).copy()
            h, w = arr.shape[0], arr.shape[1]
            area = h * w
            for _ in range(1):
                Se = random.uniform(0.02, 0.4) * area
                re = random.uniform(0.3, 3.3)
                He = int(round(np.sqrt(Se * re)))
                We = int(round(np.sqrt(Se / re)))
                if He < h and We < w:
                    xe = random.randint(0, h - He)
                    ye = random.randint(0, w - We)
                    arr[xe : xe + He, ye : ye + We, :] = 0
                    break
            return PIL.Image.fromarray(arr), target
        else:
            # assume tensor-like
            if isinstance(img, jt.Var):
                arr = img.numpy()
                is_jt = True
            else:
                arr = np.array(img)
                is_jt = False
            # handle CHW
            if arr.ndim == 3 and arr.shape[0] <= 4:
                # CHW -> HWC
                arr = arr.transpose(1, 2, 0)
            h, w = arr.shape[0], arr.shape[1]
            area = h * w
            for _ in range(1):
                Se = random.uniform(0.02, 0.4) * area
                re = random.uniform(0.3, 3.3)
                He = int(round(np.sqrt(Se * re)))
                We = int(round(np.sqrt(Se / re)))
                if He < h and We < w:
                    xe = random.randint(0, h - He)
                    ye = random.randint(0, w - We)
                    arr[xe : xe + He, ye : ye + We, :] = 0
                    break
            if is_jt:
                # back to CHW
                arr = arr.transpose(2, 0, 1)
                return jt.array(arr), target
            else:
                if arr.shape[2] <= 4:
                    arr = arr.transpose(2, 0, 1)
                return arr, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = normalize_tensor(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = jt.array(boxes) if not isinstance(boxes, jt.Var) else boxes
            boxes = boxes / jt.array([w, h, w, h], dtype=jt.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
