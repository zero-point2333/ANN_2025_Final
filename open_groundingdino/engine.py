# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

import jittor as jt
import jittor.nn as nn

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.cocogrounding_eval import CocoGroundingEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def clip_grad_norm_(parameters, max_norm: float, optimizer=None, eps: float = 1e-6):
    """
    Jittor 版的梯度裁剪。
    """
    if max_norm <= 0:
        return 0.0
    
    # 优先使用 Jittor 内置实现
    if hasattr(nn, "clip_grad_norm_"):
        return nn.clip_grad_norm_(parameters, max_norm)
    
    grads = []
    for p in parameters:
        g = None
        if optimizer is not None and hasattr(p, "opt_grad"):
            try:
                g = p.opt_grad(optimizer)
            except Exception:
                g = None
        if g is None:
            continue
        grads.append(g)
        
    if not grads:
        return 0.0
        
    # 计算总范数
    total_norm_sq = jt.zeros((1,), dtype=grads[0].dtype)
    for g in grads:
        # 这里会产生计算图，但随后的 assign 会切断它对参数的影响，所以通常没事
        total_norm_sq = total_norm_sq + jt.sum(g * g)
    total_norm = jt.sqrt(total_norm_sq)
    
    clip_coef = max_norm / (total_norm + eps)
    clip_coef = jt.minimum(clip_coef, 1.0)
    
    # 原地更新梯度
    for g in grads:
        g.assign(g * clip_coef)
        
    return total_norm


def train_one_epoch(model, criterion,
                    data_loader: Iterable, optimizer: jt.optim.AdamW,
                    epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None):

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    accum_steps = getattr(args, "accumulation_steps", 1)
    if accum_steps < 1:
        accum_steps = 1

    optimizer.zero_grad()
    step = -1

    for step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, logger=logger)
    ):
        # 1. 确保输入数据已经在 Jittor 格式 (在 collate_fn 或这里处理)
        # samples 应该是 NestedTensor (内部是 jt.Var)
        # targets 里的 Tensor 应该是 jt.Var
        print(f"step:{step}")
        captions = [t["caption"] for t in targets]
        cap_list = [t["cap_list"] for t in targets]
        
        # 过滤 targets，确保只有 Var 进入 criterion（防止元数据干扰）
        # 注意：这里假设 loader 已经把 boxes 等转成了 jt.Var
        targets_var = [{k: v for k, v in t.items() if isinstance(v, jt.Var)} for t in targets]

        outputs = model(samples, captions=captions)
        loss_dict = criterion(outputs, targets_var, cap_list, captions)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # 2. 这里的 Reduce 逻辑需要小心
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.backward(losses / accum_steps)
        
        if (step + 1) % accum_steps == 0:
            if max_norm > 0:
                clip_grad_norm_(model.parameters(), max_norm, optimizer=optimizer)
            optimizer.step()
            optimizer.zero_grad()
            if args.onecyclelr:
                lr_scheduler.step()

        # 3. 【关键修改】Metric Logger 更新
        # 必须确保传入 logger 的所有值都是 python float (.item())
        # 否则 logger 会持有一个不断增长的 list of Vars，导致显存泄露
        log_dict = {}
        for k, v in loss_dict_reduced_scaled.items():
            log_dict[k] = v.item() if isinstance(v, jt.Var) else v
            
        for k, v in loss_dict_reduced_unscaled.items():
            log_dict[k] = v.item() if isinstance(v, jt.Var) else v

        metric_logger.update(loss=loss_value, **log_dict)
        
        if 'class_error' in loss_dict_reduced:
            ce = loss_dict_reduced['class_error']
            metric_logger.update(class_error=ce.item() if isinstance(ce, jt.Var) else ce)
            
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # 4. 【关键修改】定期 GC
        # Jittor 的内存回收比较懒惰，手动 GC 可以防止碎片化导致的 OOM
        if step % 50 == 0:
            jt.gc()

        _cnt += 1
        if args.debug and _cnt % 15 == 0:
            print("BREAK!"*5)
            break

    # 处理最后一个可能未完成的 accum step
    if step >= 0 and (step + 1) % accum_steps != 0:
        if max_norm > 0:
            clip_grad_norm_(model.parameters(), max_norm, optimizer=optimizer)
        optimizer.step()
        optimizer.zero_grad()
        if args.onecyclelr:
            lr_scheduler.step()

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    
    # epoch 结束彻底清理一次
    jt.sync_all()
    jt.gc()
    return resstat


@jt.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, output_dir, wo_class_error=False, args=None, logger=None):

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = getattr(args, "useCats", True)
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    
    # 确保 base_ds 是 CPU 数据集
    coco_evaluator = CocoGroundingEvaluator(base_ds, iou_types, useCats=useCats)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    
    if args.use_coco_eval:
        from pycocotools.coco import COCO
        coco = COCO(args.coco_val_path)
        category_dict = coco.loadCats(coco.getCatIds())
        cat_list = [item['name'] for item in category_dict]
    else:
        cat_list = args.label_list
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        # targets 不需要转 Var，因为这里只是为了取 image_id 和 orig_size
        # 但如果 model output 需要 orig_size 计算，则需要转
        
        bs = samples.tensors.shape[0]
        input_captions = [caption] * bs

        outputs = model(samples, captions=input_captions)

        # 获取原始尺寸，转为 Jittor Var 用于 postprocessors 计算
        # 注意：targets 中的数据可能是 numpy，需要转换
        orig_target_sizes = []
        for t in targets:
            # 兼容处理：如果是 Var 直接用，如果是 numpy 转 array
            val = t["orig_size"]
            if not isinstance(val, jt.Var):
                val = jt.array(val)
            orig_target_sizes.append(val)
        orig_target_sizes = jt.stack(orig_target_sizes, dim=0)
        
        # Postprocessing: 返回的结果通常是 {'scores': Var, 'labels': Var, 'boxes': Var}
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        if 'segm' in postprocessors.keys():
            target_sizes = jt.stack([jt.array(t["size"]) for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            
        # 5. 【关键修改】结果数据清洗
        # coco_evaluator.update 会把结果存入列表。
        # 如果这里直接存 Jittor Var，评估过程的显存占用会随图片数量线性增长，必爆显存。
        # 必须把 Tensor 转为 Numpy/List。
        
        res = {}
        for target, output in zip(targets, results):
            # output 是一个 dict，例如 {'scores': Var, ...}
            # 我们需要把它变成纯 CPU 数据
            cpu_output = {}
            for k, v in output.items():
                if isinstance(v, jt.Var):
                    # .numpy() 会触发计算并把数据拉回 CPU，同时切断计算图
                    cpu_output[k] = v.numpy()
                else:
                    cpu_output[k] = v
            
            res[target['image_id'].item()] = cpu_output

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            # 同理处理 Panoptic
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            # 确保 res_pano 里面的数据也是 cpu 友好的
            # (这里省略 Panoptic 的详细转换，原理同上，如有需要请检查 PanopticEvaluator)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name
            panoptic_evaluator.update(res_pano)

        # 6. 定期 GC
        _cnt += 1
        if _cnt % 10 == 0:
            jt.gc()
            
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
        
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    # 评估结束清理
    jt.sync_all()
    jt.gc()
    
    return stats, coco_evaluator