import os
import random

import numpy as np
import PIL
import jittor as jt
from jittor import transform as T
from groundingdino.util.box_ops import box_xyxy_to_cxcywh
from groundingdino.util.misc import interpolate


def crop(image, target, region):
    # region: top, left, height, width
    top, left, height, width = region
    cropped_image = image.crop((left, top, left + width, top + height))

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = jt.array([h, w])

    fields = ["labels", "area", "iscrowd", "positive_map"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = jt.array([w, h], dtype=jt.float32)
        cropped_boxes = boxes - jt.array([j, i, j, i])
        cropped_boxes = jt.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.maximum(0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        masks = target["masks"]
        target["masks"] = masks[:, i:i + h, j:j + w]
        fields.append("masks")

    if "boxes" in target or "masks" in target:
        if "boxes" in target:
            cropped_boxes = target["boxes"].reshape(-1, 2, 2)
            keep = jt.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
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
        if "strings_positive" in target:
            if isinstance(keep, jt.Var):
                keep_np = keep.numpy().tolist()
            else:
                keep_np = list(keep)
            target["strings_positive"] = [_i for _i, _j in zip(target["strings_positive"], keep_np) if _j]

    return cropped_image, target


def hflip(image, target):
    flipped_image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * jt.array([-1, 1, -1, 1]) + jt.array(
            [w, 0, w, 0]
        )
        target["boxes"] = boxes

    if "masks" in target:
        masks = target["masks"]
        if isinstance(masks, jt.Var):
            target["masks"] = masks.flip(-1)
        else:
            target["masks"] = np.flip(masks, axis=-1).copy()

    return flipped_image, target


def resize(image, target, size, max_size=None): # used
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

    size = get_size(image.size, size, max_size)[::-1]
    # PIL expects (width, height)
    rescaled_image = image.resize(size, resample=PIL.Image.BILINEAR)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * jt.array(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        area = jt.array(area) if not isinstance(area, jt.Var) else area
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = jt.array([h, w])

    if "masks" in target:
        masks = target["masks"]
        target["masks"] = (
            interpolate(target["masks"][:, None].float(), size, mode="nearest")[:, 0] > 0.5
        )

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = PIL.ImageOps.expand(image, border=(0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    target["size"] = jt.array(padded_image.size[::-1])
    if "masks" in target:
        target["masks"] = jt.nn.pad(target["masks"], (0, padding[0], 0, padding[1]))
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
        if isinstance(img, PIL.Image.Image):
            w, h = img.size
        else:
            h, w = img.shape[-2], img.shape[-1]

        crop_h, crop_w = self.size
        if crop_h > h or crop_w > w:
            crop_h = min(crop_h, h)
            crop_w = min(crop_w, w)

        i = random.randint(0, h - crop_h)
        j = random.randint(0, w - crop_w)
        region = (i, j, crop_h, crop_w)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, respect_boxes: bool = False):
        self.min_size = min_size
        self.max_size = max_size
        self.respect_boxes = respect_boxes

    def __call__(self, img: PIL.Image.Image, target: dict):
        init_boxes = len(target["boxes"])
        max_patience = 10
        for i in range(max_patience):
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))

            i = random.randint(0, img.height - h) if img.height > h else 0
            j = random.randint(0, img.width - w) if img.width > w else 0
            region = (i, j, h, w)

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
        if isinstance(img, PIL.Image.Image):
            image_width, image_height = img.size
        else:
            image_height, image_width = img.shape[-2], img.shape[-1]

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


class RandomResize(object): # used
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
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object): # used
    def __call__(self, img, target):
        if isinstance(img, PIL.Image.Image):
            img = T.to_tensor(img)
        return img, target


class RandomErasing(object):
    def __init__(self, *args, **kwargs):
        # Jittor 目前没有 RandomErasing，可以手动实现或使用其他增强方法
        # 这里先保持空实现，需要时再补充
        pass

    def __call__(self, img, target):
        # TODO: 实现 Jittor 版本的 RandomErasing
        return img, target


class Normalize(object): # used
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = T.image_normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / jt.array([w, h, w, h], dtype=jt.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object): # used partially
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self): # unused
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
