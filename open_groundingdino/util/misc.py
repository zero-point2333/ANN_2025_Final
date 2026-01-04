# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including helpers.

Originally adapted from torchvision references; this version has been
ported to work with Jittor instead of PyTorch.
"""
import colorsys
import datetime
import functools
import io
import json
import os
import pickle
import subprocess
import time
from collections import OrderedDict, defaultdict, deque
from typing import List, Optional

import numpy as np
import jittor as jt
import jittor.nn as nn


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(float(value))
        self.count += n
        self.total += float(value) * n

    def synchronize_between_processes(self):
        """
        In the original PyTorch version this synchronized across processes.
        In the Jittor port we run in single-process mode, so this is a no-op.
        """
        return

    @property
    def median(self):
        if len(self.deque) == 0:
            return 0.0
        d = np.asarray(self.deque, dtype=np.float64)
        return float(np.median(d))

    @property
    def avg(self):
        if len(self.deque) == 0:
            return 0.0
        d = np.asarray(self.deque, dtype=np.float32)
        return float(np.mean(d))

    @property
    def global_avg(self):
        # keep the small epsilon logic to avoid division by zero
        if os.environ.get("SHILONG_AMP", None) == "1":
            eps = 1e-4
        else:
            eps = 1e-6
        return float(self.total) / float(self.count + eps)

    @property
    def max(self):
        if len(self.deque) == 0:
            return 0.0
        return max(self.deque)

    @property
    def value(self):
        if len(self.deque) == 0:
            return 0.0
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Placeholder kept for API compatibility. In this Jittor port,
    distributed training is disabled, so this simply returns None.
    """
    return None


def all_gather_cpu(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Jittor port note:
        We only support single-process execution here, so this
        function simply wraps the input in a list.
    """
    return [data]


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Jittor port note:
        We only support single-process execution here, so this
        function simply wraps the input in a list.
    """
    return [data]


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results.

    Jittor port note:
        Since we only run in single-process mode, this is effectively a no-op.
    """
    return input_dict


class MetricLogger(object):
    def __init__(self, delimiter: str = "\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            # Support Jittor Var, numpy scalar and Python scalar
            if isinstance(v, jt.Var):
                # Jittor 标量：先转 numpy，再转成 Python float
                v = float(v.numpy())
            elif isinstance(v, np.generic):
                v = float(v)
            elif hasattr(v, "item"):
                v = float(v.item())
            assert isinstance(v, (float, int)), f"Metric {k} must be float or int, got {type(v)}"
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if meter.count > 0:
                loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, logger=None):
        if logger is None:
            print_func = print
        else:
            print_func = logger.info

        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"

        # In this Jittor port we do not track CUDA memory; keep log format simple.
        log_msg = self.delimiter.join(
            [
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
            ]
        )

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print_func(
                    log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                    )
                )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print_func(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / max(len(iterable), 1)
            )
        )


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    # batch is list of tuples (image, target, ...)
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list: List[List[int]]) -> List[int]:
    maxes = list(the_list[0])
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    """
    Wrapper for a batched tensor and an associated mask.
    
    Jittor Optimization Note:
    We MUST store data as jt.Var to maintain the computational graph.
    Storing as numpy (as in the original code) will break autograd and cause OOM.
    """

    def __init__(self, tensors, mask: Optional[jt.Var] = "auto"):
        # 1. 强制转换为 jt.Var，确保在计算图中
        if not isinstance(tensors, jt.Var):
            tensors = jt.array(tensors)
        self.tensors = tensors
        
        # 2. Mask 处理
        if isinstance(mask, str) and mask == "auto":
            # 自动构建 "全0 (无padding)" mask
            if self.tensors.ndim == 3:
                _, h, w = self.tensors.shape
                # mask 不需要梯度，必须 stop_grad
                self.mask = jt.zeros((h, w)).bool().stop_grad()
            elif self.tensors.ndim == 4:
                b, _, h, w = self.tensors.shape
                self.mask = jt.zeros((b, h, w)).bool().stop_grad()
            else:
                raise ValueError(f"tensors dim must be 3 or 4 but got {self.tensors.ndim}")
        else:
            # 如果传入了 mask
            if mask is not None:
                if not isinstance(mask, jt.Var):
                    mask = jt.array(mask)
                # 关键：确保 mask 不带梯度
                self.mask = mask.stop_grad()
            else:
                self.mask = None

    def imgsize(self):
        """
        Return a list of [H, W] for each image.
        Uses jittor ops to avoid sync, but returns python list for compatibility.
        """
        res = []
        if self.tensors.ndim != 4:
             raise ValueError("imgsize only defined for batched (B, C, H, W) tensors")
        
        # mask: True 表示 padding
        # inv: True 表示 valid pixel
        # 注意：这里 mask 已经是 stop_grad 的了，运算安全
        
        # 为了获取具体的数值（通常用于后续逻辑判断），这里可能需要同步一次
        # 但通常 imgsize 是在后处理阶段用的，所以 sync 影响不大
        # 如果是在 Loss 计算中用到，最好避免 .item()
        
        for i in range(self.tensors.shape[0]):
            mask_i = self.mask[i] 
            inv = ~mask_i # False = padding
            
            # 使用 Jittor 算子计算高宽
            maxH = inv.sum(dim=0).max().item() # .item() 会触发同步，但在后处理中通常是必要的
            maxW = inv.sum(dim=1).max().item()
            res.append([int(maxH), int(maxW)])
            
        return res

    def to(self, device):
        # Jittor 自动管理设备，此函数仅为了兼容 API
        return self

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    @property
    def shape(self):
        return {"tensors.shape": self.tensors.shape, "mask.shape": self.mask.shape}
    
    # 兼容性属性
    @property
    def device(self):
        return "cuda" if jt.flags.use_cuda else "cpu"


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List):
    """
    Build a NestedTensor from a list of CHW tensors.
    
    This function usually runs in the DataLoader collate_fn.
    We build the batch as jt.Var directly.
    """
    if len(tensor_list) == 0:
        raise ValueError("tensor_list must be non-empty")

    # 1. 统一转为 Jittor Var 来获取 shape，或者直接读取 shape
    # 为了效率，我们先读取 shape，不要急着把所有数据塞进 GPU
    
    # 假设输入可能是 numpy, jt.Var, 或其他
    # 先探测第一个元素的属性
    if isinstance(tensor_list[0], jt.Var):
        # 如果已经是 Var (比如在模型内部调用)，直接处理
        # 这种情况比较少见，通常 tensor_list 是 list of Vars
        # 我们需要先看看 shape
        shapes = [list(img.shape) for img in tensor_list]
        dtype = tensor_list[0].dtype
    else:
        # 如果是 numpy (DataLoader 中常见)
        # 保持 numpy 操作直到最后一步，这样最快
        np_list = [np.array(img) for img in tensor_list]
        shapes = [list(img.shape) for img in np_list]
        dtype = "float32" # 默认

    # 检查维度
    if len(shapes[0]) != 3:
        raise ValueError("Only 3D CHW tensors are supported")

    # 计算最大尺寸 (C, H, W)
    max_size = _max_by_axis(shapes)
    batch_shape = [len(tensor_list)] + max_size
    b, c, h, w = batch_shape

    # 2. 构建 Batch Tensor 和 Mask
    # 策略：如果输入是 numpy，我们在 CPU (numpy) 上拼好，然后一次性转 Jittor
    # 这样比在 Jittor 上创建 huge zeros 然后循环赋值要快，且显存碎片少
    
    if isinstance(tensor_list[0], jt.Var):
        # 如果输入已经是 Jittor Var，我们必须在 Jittor 上操作以保持图连接（如果需要的话）
        # 或者在 DataLoader 中，这里其实是创建新的叶子节点
        
        batch_tensor = jt.zeros(batch_shape, dtype=dtype)
        batch_mask = jt.ones((b, h, w), dtype="bool") # 1 = padding
        
        for i, img in enumerate(tensor_list):
            c_i, h_i, w_i = img.shape
            # Jittor 的切片赋值
            batch_tensor[i, :c_i, :h_i, :w_i] = img
            batch_mask[i, :h_i, :w_i] = False # 0 = valid
            
        # Mask 必须切断梯度
        batch_mask = batch_mask.stop_grad()
        
    else:
        # 输入是 Numpy (推荐路径 for DataLoader)
        # 先在 numpy 上拼，快且省显存
        batch_numpy = np.zeros(batch_shape, dtype=np.float32)
        mask_numpy = np.ones((b, h, w), dtype=bool)
        
        for i, img in enumerate(np_list):
            c_i, h_i, w_i = img.shape
            batch_numpy[i, :c_i, :h_i, :w_i] = img
            mask_numpy[i, :h_i, :w_i] = False
            
        # 一次性转 Jittor
        batch_tensor = jt.array(batch_numpy)
        batch_mask = jt.array(mask_numpy).stop_grad() # 显式 stop_grad

    return NestedTensor(batch_tensor, batch_mask)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process.
    In the Jittor single-process port, this still works but is rarely used.
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print_fn(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print_fn


def is_dist_avail_and_initialized():
    """
    Check if distributed training is available and initialized.

    Jittor port note:
        We only support single-process mode here, so this always returns False.
    """
    return False


def get_world_size():
    return 1


def get_rank():
    return 0


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    Save checkpoint on main process.

    Jittor port note:
        For simplicity we disable checkpoint saving here. Keeping the
        function for API compatibility.
    """
    if is_main_process():
        # You can optionally plug in `jt.save` or `pickle.dump` here if needed.
        pass


def init_distributed_mode(args):
    """
    Initialize distributed training.

    Jittor port note:
        Distributed / multi-GPU training is not supported in this port.
        We force single-process mode and set the relevant attributes on args.
    """
    print("Not using distributed mode (Jittor single-process port).")
    args.distributed = False
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    # keep these attributes for compatibility if other code accesses them
    if not hasattr(args, "dist_backend"):
        args.dist_backend = "nccl"
    if not hasattr(args, "dist_url"):
        args.dist_url = "env://"
    # also ensure printing only from main process
    setup_for_distributed(True)


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k.

    Jittor port note:
        Returns a list of jt.Var scalars to keep `.item()` semantics.
    """
    # Convert to numpy
    if isinstance(output, jt.Var):
        output_np = output.numpy()
    else:
        output_np = np.asarray(output)

    if isinstance(target, jt.Var):
        target_np = target.numpy()
    else:
        target_np = np.asarray(target)

    if target_np.size == 0:
        return [jt.float32([0.0]) for _ in topk]

    # Ensure shape (N, C) for output and (N,) for target
    if output_np.ndim != 2:
        raise ValueError(f"accuracy expects output of shape (N, C), got {output_np.shape}")
    target_np = target_np.reshape(-1)
    batch_size = target_np.shape[0]
    maxk = max(topk)

    # top-k indices for each sample (descending scores)
    # shape: (N, maxk)
    topk_idx = np.argsort(-output_np, axis=1)[:, :maxk]

    res = []
    for k in topk:
        pred_k = topk_idx[:, :k]  # (N, k)
        correct = (pred_k == target_np[:, None])
        correct_k = np.any(correct, axis=1).sum()
        acc = float(correct_k) * 100.0 / float(batch_size)
        # 返回 1 元素 Var，外面可以 .item()
        res.append(jt.float32([acc]))
    return res


def accuracy_onehot(pred, gt):
    """
    Accuracy for one-hot predictions / labels.

    Args:
        pred: array-like of shape (N, C)
        gt:   array-like of shape (N, C)

    Returns:
        Tensor-like scalar, percentage.
    """
    if not (hasattr(pred, "abs") and hasattr(pred, "sum") and hasattr(pred, "float")):
        pred = jt.array(pred)
    if not (hasattr(gt, "abs") and hasattr(gt, "sum") and hasattr(gt, "float")):
        gt = jt.array(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have same shape, got {pred.shape} vs {gt.shape}")
    tp = ((pred - gt).abs().sum(-1) < 1e-4).float().sum()
    acc = tp / gt.shape[0] * 100
    return acc
def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """
    Wrapper around jittor.nn.interpolate with a PyTorch-like signature.

    Args:
        input: numpy array or jt.Var of shape (N, C, H, W) or (C, H, W)
        size: output spatial size (H, W)
        scale_factor: scale factor for H and W
        mode: interpolation mode, e.g. 'nearest' or 'bilinear'
        align_corners: forwarded to jittor.nn.interpolate for 'bilinear'
    """
    numpy_input = isinstance(input, np.ndarray)
    torch_like_input = False
    if numpy_input:
        x = jt.array(input)
    elif hasattr(input, "detach") and hasattr(input, "cpu") and hasattr(input, "numpy"):
        torch_like_input = True
        x = jt.array(input.detach().cpu().numpy())
    else:
        x = input

    out = nn.interpolate(
        x,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )

    if numpy_input:
        return out.numpy()
    if torch_like_input and hasattr(input, "new_tensor"):
        return input.new_tensor(out.numpy())
    return out


class color_sys:
    def __init__(self, num_colors) -> None:
        self.num_colors = num_colors
        colors = []
        for i in np.arange(0.0, 360.0, 360.0 / num_colors):
            hue = i / 360.0
            lightness = (50 + np.random.rand() * 10) / 100.0
            saturation = (90 + np.random.rand() * 10) / 100.0
            colors.append(
                tuple(int(j * 255) for j in colorsys.hls_to_rgb(hue, lightness, saturation))
            )
        self.colors = colors

    def __call__(self, idx):
        return self.colors[idx]


def inverse_sigmoid(x, eps=1e-3):
    """
    Numerically stable inverse of the sigmoid, working on Jittor Var or numpy array.
    """
    if isinstance(x, jt.Var): # always jt.Var
        x = jt.clamp(x, 0.0, 1.0)
        x1 = jt.clamp(x, eps, 1.0)
        x2 = jt.clamp(1.0 - x, eps, 1.0)
        return jt.log(x1 / x2)
    if hasattr(x, "clamp") and hasattr(x, "log"):
        x = x.clamp(min=0.0, max=1.0)
        x1 = x.clamp(min=eps)
        x2 = (1.0 - x).clamp(min=eps)
        return (x1 / x2).log()
    else:
        x_arr = np.asarray(x, dtype=np.float32)
        x_arr = np.clip(x_arr, 0.0, 1.0)
        x1 = np.clip(x_arr, eps, None)
        x2 = np.clip(1.0 - x_arr, eps, None)
        return np.log(x1 / x2)
def clean_state_dict(state_dict):
    import jittor as jt
    import torch
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            k = k[7:]  # remove `module.`
        # Handle backbone naming difference: backbone.0 -> backbone.backbone
        if k.startswith("backbone.0"):
            k = k.replace("backbone.0", "backbone.backbone", 1)
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
