import argparse
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import jittor as jt
from jittor import nn
from transformers import AutoTokenizer

from groundingdino.util.slconfig import SLConfig


def slprint(x, name="x"):
    if isinstance(x, (jt.Var, np.ndarray)):
        print(f"{name}.shape:", x.shape)
    elif isinstance(x, (tuple, list)):
        print("type x:", type(x))
        for i in range(min(10, len(x))):
            slprint(x[i], f"{name}[{i}]")
    elif isinstance(x, dict):
        for k, v in x.items():
            slprint(v, f"{name}[{k}]")
    else:
        print(f"{name}.type:", type(x))


def clean_state_dict(state_dict):
    import jittor as jt
    import torch
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            k = k[7:]  # remove `module.`
        if k.startswith("bert."):
            continue  # Skip BERT parameters as they are handled by transformers
        # Only keep tensors, convert PyTorch tensors to Jittor
        if isinstance(v, jt.Var):
            new_state_dict[k] = v
        elif isinstance(v, torch.Tensor):
            new_state_dict[k] = jt.array(v.detach().cpu().numpy())
        elif isinstance(v, dict):
            # Recursively clean nested dicts
            new_state_dict[k] = clean_state_dict(v)
        # Skip non-tensor objects like modules
    return new_state_dict


def renorm(
    img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    """
    img: jt.Var with shape (3,H,W) or (B,3,H,W)
    return: same shape as img
    """
    # 兼容 jt.Var / numpy
    if isinstance(img, np.ndarray):
        img = jt.array(img)
    assert len(img.shape) in (3, 4), f"img.ndim should be 3 or 4 but {len(img.shape)}"

    if len(img.shape) == 3:
        assert img.shape[0] == 3, f'img.shape[0] should be 3 but {img.shape}'
        img_perm = img.permute(1, 2, 0)  # H,W,3
        mean_var = jt.array(mean)
        std_var = jt.array(std)
        img_res = img_perm * std_var + mean_var
        return img_res.permute(2, 0, 1)
    else:
        assert img.shape[1] == 3, f'img.shape[1] should be 3 but {img.shape}'
        img_perm = img.permute(0, 2, 3, 1)  # B,H,W,3
        mean_var = jt.array(mean)
        std_var = jt.array(std)
        img_res = img_perm * std_var + mean_var
        return img_res.permute(0, 3, 1, 2)


class CocoClassMapper:
    def __init__(self) -> None:
        self.category_map_str = {
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
            "11": 11,
            "13": 12,
            "14": 13,
            "15": 14,
            "16": 15,
            "17": 16,
            "18": 17,
            "19": 18,
            "20": 19,
            "21": 20,
            "22": 21,
            "23": 22,
            "24": 23,
            "25": 24,
            "27": 25,
            "28": 26,
            "31": 27,
            "32": 28,
            "33": 29,
            "34": 30,
            "35": 31,
            "36": 32,
            "37": 33,
            "38": 34,
            "39": 35,
            "40": 36,
            "41": 37,
            "42": 38,
            "43": 39,
            "44": 40,
            "46": 41,
            "47": 42,
            "48": 43,
            "49": 44,
            "50": 45,
            "51": 46,
            "52": 47,
            "53": 48,
            "54": 49,
            "55": 50,
            "56": 51,
            "57": 52,
            "58": 53,
            "59": 54,
            "60": 55,
            "61": 56,
            "62": 57,
            "63": 58,
            "64": 59,
            "65": 60,
            "67": 61,
            "70": 62,
            "72": 63,
            "73": 64,
            "74": 65,
            "75": 66,
            "76": 67,
            "77": 68,
            "78": 69,
            "79": 70,
            "80": 71,
            "81": 72,
            "82": 73,
            "84": 74,
            "85": 75,
            "86": 76,
            "87": 77,
            "88": 78,
            "89": 79,
            "90": 80,
        }
        self.origin2compact_mapper = {int(k): v - 1 for k, v in self.category_map_str.items()}
        self.compact2origin_mapper = {int(v - 1): int(k) for k, v in self.category_map_str.items()}

    def origin2compact(self, idx):
        return self.origin2compact_mapper[int(idx)]

    def compact2origin(self, idx):
        return self.compact2origin_mapper[int(idx)]


def to_device(item, device):

    # Jittor: 全局 jt.flags.use_cuda 控制设备
    if hasattr(item, "to") and not isinstance(item, (list, dict)):
        try:
            return item.to(device)
        except TypeError:
            # Jittor Var.to(...) 可能不存在，直接返回
            return item
    elif isinstance(item, list):
        return [to_device(i, device) for i in item]
    elif isinstance(item, dict):
        return {k: to_device(v, device) for k, v in item.items()}
    else:
        return item


def get_gaussian_mean(x, axis, other_axis, softmax=True):
    """
    Args:
        x: BxCxHxW  (jt.Var)
        axis (int): 要取加权平均的维度
        other_axis (int): 另一空间维度
    Returns:
        BxC 的加权位置（0~1）
    """
    mat2line = jt.sum(x, dim=other_axis)
    if softmax:
        # 原来是 torch.softmax(..., dim=2)
        u = nn.softmax(mat2line, dim=2)
    else:
        u = mat2line / (jt.sum(mat2line, dim=2, keepdims=True) + 1e-6)

    size = x.shape[axis]
    ind = jt.linspace(0.0, 1.0, size)  # [size]
    batch = x.shape[0]
    channel = x.shape[1]
    index = ind.repeat(batch, channel, 1)  # B,C,size
    mean_position = jt.sum(index * u, dim=2)
    return mean_position


def get_expected_points_from_map(hm, softmax=True):
    """
    B,C,H,W -> B,C,2  (x,y in [0,1])
    """
    B, C, H, W = hm.shape
    y_mean = get_gaussian_mean(hm, 2, 3, softmax=softmax)  # B,C
    x_mean = get_gaussian_mean(hm, 3, 2, softmax=softmax)  # B,C
    return jt.stack([x_mean, y_mean], dim=2)


# Positional encoding
# borrow from nerf
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** jt.linspace(0.0, float(max_freq), steps=N_freqs)
        else:
            freq_bands = jt.linspace(2.0 ** 0.0, 2.0 ** float(max_freq), steps=N_freqs)

        # 将频率转换为 python float
        freq_bands_list = [float(v) for v in freq_bands.tolist()]

        for freq in freq_bands_list:
            for p_fn in self.kwargs["periodic_fns"]:
                # 这里 freq 是 float，后面 x 是 jt.Var
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        outs = [fn(inputs) for fn in self.embed_fns]
        return jt.concat(outs, dim=-1)


def get_embedder(multires, i=0):
    if i == -1:
        # 简单 Identity
        class _Identity:
            def __call__(self, x):
                return x

        return _Identity(), 3

    embed_kwargs = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [jt.sin, jt.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class APOPMeter:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, pred, gt):
        """
        Input:
            pred, gt: Var / Tensor with same shape, values in {0,1}
        """
        assert pred.shape == gt.shape
        pred1 = (pred == 1)
        gt1 = (gt == 1)
        pred0 = (pred == 0)
        gt0 = (gt == 0)

        self.tp += (jt.logical_and(pred1, gt1)).sum().item()
        self.fp += (jt.logical_and(pred1, gt0)).sum().item()
        self.tn += (jt.logical_and(pred0, gt0)).sum().item()
        self.tn += (jt.logical_and(pred1, gt0)).sum().item()

    def update_cm(self, tp, fp, tn, fn):
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.tn += fn


def inverse_sigmoid(x, eps=1e-5):
    # x: jt.Var, clamp 到 [0,1]
    x = jt.maximum(x, 0.0)
    x = jt.minimum(x, 1.0)
    x1 = jt.maximum(x, eps)
    x2 = jt.maximum(1.0 - x, eps)
    return jt.log(x1 / x2)


def get_raw_dict(args):
    """
    return the dict contained in args.

    e.g:
        >>> with open(path, 'w') as f:
        ...     json.dump(get_raw_dict(args), f, indent=2)
    """
    if isinstance(args, argparse.Namespace):
        return vars(args)
    elif isinstance(args, dict):
        return args
    elif isinstance(args, SLConfig):
        return args._cfg_dict
    else:
        raise NotImplementedError("Unknown type {}".format(type(args)))


def stat_tensors(tensor):
    """
    输入 1-D jt.Var，输出若干统计量
    """
    assert len(tensor.shape) == 1
    tensor_sm = nn.softmax(tensor, dim=0)
    entropy = (tensor_sm * jt.log(tensor_sm + 1e-9)).sum()

    return {
        "max": tensor.max(),
        "min": tensor.min(),
        "mean": tensor.mean(),
        "var": tensor.var(),
        "std": jt.sqrt(tensor.var()),
        "entropy": entropy,
    }


class NiceRepr:
    """继承后实现 __nice__，可获得统一的 __str__ 和 __repr__ 行为。"""

    def __nice__(self):
        if hasattr(self, "__len__"):
            return str(len(self))
        else:
            raise NotImplementedError(f"Define the __nice__ method for {self.__class__!r}")

    def __repr__(self):
        try:
            nice = self.__nice__()
            classname = self.__class__.__name__
            return f"<{classname}({nice}) at {hex(id(self))}>"
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)

    def __str__(self):
        try:
            classname = self.__class__.__name__
            nice = self.__nice__()
            return f"<{classname}({nice})>"
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)


def ensure_rng(rng=None):
    """
    将输入转换成 numpy.random.RandomState。
    """
    if rng is None:
        rng = np.random.mtrand._rand
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)
    else:
        rng = rng
    return rng


def random_boxes(num=1, scale=1, rng=None):
    """Simple version of kwimage.Boxes.random

    Returns:
        jt.Var: shape (n, 4) in x1, y1, x2, y2 format.
    """
    rng = ensure_rng(rng)

    tlbr = rng.rand(num, 4).astype(np.float32)

    tl_x = np.minimum(tlbr[:, 0], tlbr[:, 2])
    tl_y = np.minimum(tlbr[:, 1], tlbr[:, 3])
    br_x = np.maximum(tlbr[:, 0], tlbr[:, 2])
    br_y = np.maximum(tlbr[:, 1], tlbr[:, 3])

    tlbr[:, 0] = tl_x * scale
    tlbr[:, 1] = tl_y * scale
    tlbr[:, 2] = br_x * scale
    tlbr[:, 3] = br_y * scale

    boxes = jt.array(tlbr)
    return boxes


class ModelEma(nn.Module):

    def __init__(self, model, decay=0.9997, device=None):
        super().__init__()
        # make a copy of the model
        self.module = deepcopy(model)
        self.module.eval()

        self.decay = decay
        self.device = device
        # Jittor 一般不需要 per-module 设置 device，统一由 jt.flags.use_cuda 控制

    def _update(self, model, update_fn):
        # 简化实现：直接深拷贝当前模型
        self.module = deepcopy(model)
        self.module.eval()

    def update(self, model):
        self._update(model, update_fn=None)

    def set(self, model):
        self.module = deepcopy(model)
        self.module.eval()


class BestMetricSingle:
    def __init__(self, init_res=0.0, better="large") -> None:
        self.init_res = init_res
        self.best_res = init_res
        self.best_ep = -1

        self.better = better
        assert better in ["large", "small"]

    def isbetter(self, new_res, old_res):
        if self.better == "large":
            return new_res > old_res
        if self.better == "small":
            return new_res < old_res

    def update(self, new_res, ep):
        if self.isbetter(new_res, self.best_res):
            self.best_res = new_res
            self.best_ep = ep
            return True
        return False

    def __str__(self) -> str:
        return "best_res: {}\t best_ep: {}".format(self.best_res, self.best_ep)

    def __repr__(self) -> str:
        return self.__str__()

    def summary(self) -> dict:
        return {
            "best_res": self.best_res,
            "best_ep": self.best_ep,
        }


class BestMetricHolder:
    def __init__(self, init_res=0.0, better="large", use_ema=False) -> None:
        self.best_all = BestMetricSingle(init_res, better)
        self.use_ema = use_ema
        if use_ema:
            self.best_ema = BestMetricSingle(init_res, better)
            self.best_regular = BestMetricSingle(init_res, better)

    def update(self, new_res, epoch, is_ema=False):
        """
        return if the results is the best.
        """
        if not self.use_ema:
            return self.best_all.update(new_res, epoch)
        else:
            if is_ema:
                self.best_ema.update(new_res, epoch)
                return self.best_all.update(new_res, epoch)
            else:
                self.best_regular.update(new_res, epoch)
                return self.best_all.update(new_res, epoch)

    def summary(self):
        if not self.use_ema:
            return self.best_all.summary()

        res = {}
        res.update({f"all_{k}": v for k, v in self.best_all.summary().items()})
        res.update({f"regular_{k}": v for k, v in self.best_regular.summary().items()})
        res.update({f"ema_{k}": v for k, v in self.best_ema.summary().items()})
        return res

    def __repr__(self) -> str:
        return json.dumps(self.summary(), indent=2)

    def __str__(self) -> str:
        return self.__repr__()


def targets_to(targets: List[Dict[str, Any]], device):

    excluded_keys = [
        "questionId",
        "tokens_positive",
        "strings_positive",
        "tokens",
        "dataset_name",
        "sentence_id",
        "original_img_id",
        "nb_eval",
        "task_id",
        "original_id",
        "token_span",
        "caption",
        "dataset_type",
    ]
    new_targets = []
    for t in targets:
        new_t = {}
        for k, v in t.items():
            if k in excluded_keys:
                new_t[k] = v
            else:
                if hasattr(v, "to"):
                    try:
                        new_t[k] = v.to(device)
                    except TypeError:
                        new_t[k] = v
                else:
                    new_t[k] = v
        new_targets.append(new_t)
    return new_targets


def get_phrases_from_posmap(
    posmap,
    tokenized: Dict,
    tokenizer: AutoTokenizer,
    left_idx: int = 0,
    right_idx: int = 255,
):
    """
    posmap: 1-D bool mask (jt.Var / numpy / list)，True 的位置对应要保留的 token。
    """
    # 转成 numpy
    if hasattr(posmap, "numpy"):
        mask = posmap.numpy()
    else:
        mask = np.asarray(posmap)

    assert mask.ndim == 1, "posmap must be 1-dim"

    # 截掉 [0, left_idx] 和 [right_idx, end)
    mask[: left_idx + 1] = False
    if right_idx < mask.shape[0]:
        mask[right_idx:] = False

    non_zero_idx = np.nonzero(mask)[0].tolist()
    token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
    return tokenizer.decode(token_ids)