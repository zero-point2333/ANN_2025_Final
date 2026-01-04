import argparse
import json
import time
import os
import copy
import numpy as np
import jittor as jt
import jittor.nn as nn
import torchvision
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.inference import _load_checkpoint_any, load_model
from groundingdino.util.misc import collate_fn
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span

# 引入 LVIS 库
try:
    from lvis import LVIS, LVISEval, LVISResults
except ImportError:
    print("Please install lvis api: pip install lvis")
    exit()

class LVISDetection(torchvision.datasets.VisionDataset):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder)
        self.lvis = LVIS(ann_file)
        self.ids = list(self.lvis.imgs.keys())
        self._transforms = transforms
        self.root = img_folder

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = str(self.lvis.load_imgs([img_id])[0]['id']).zfill(12)
        file_name = f"{img_info}.jpg"
        # LVIS 图片通常在 coco 文件夹结构下，可能需要根据实际路径调整
        # 如果 file_name 包含路径（如 coco/val2017/xxx.jpg），则直接用
        # 否则拼接 root
        img_path = os.path.join(self.root, file_name)
        
        # Jittor/PIL 读取图片
        from PIL import Image
        img = Image.open(img_path).convert("RGB")

        # 加载标注 (Target)
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        anns = self.lvis.load_anns(ann_ids)
        
        target = []
        for ann in anns:
            target.append(ann)

        w, h = img.size
        
        if len(target) > 0:
            boxes = [obj["bbox"] for obj in target]
            boxes = jt.array(boxes, dtype=jt.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
            boxes[:, 0::2].clamp_(min_v=0, max_v=w)
            boxes[:, 1::2].clamp_(min_v=0, max_v=h)
            
            # 过滤无效框
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            # 对应的还需要过滤 classes，但在纯测试流程中，dataset输出的target主要用于transform
            # GroundingDINO 推理不需要 GT 类别，只需要 GT boxes 做 shape 参考（或不需要）
        else:
            boxes = jt.zeros((0, 4), dtype=jt.float32)

        target_new = {}
        target_new["image_id"] = img_id
        target_new["boxes"] = boxes
        target_new["orig_size"] = jt.array([int(h), int(w)])
        
        # 仅仅为了兼容 transform 接口
        if self._transforms is not None:
            img, target_new = self._transforms(img, target_new)

        return img, target_new

    def __len__(self):
        return len(self.ids)

class PostProcessLVIS(nn.Module):
    """ 
    通用后处理模块，支持动态传入 label_map。
    用于处理分块推理时的局部 ID 到全局 LVIS ID 的映射。
    """
    def __init__(self, num_select=300, tokenlizer=None) -> None:
        super().__init__()
        self.num_select = num_select
        self.tokenlizer = tokenlizer

    def build_positive_map(self, cat_list):
        """为当前的类别块构建 positive_map"""
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat.lower()] for cat in cat_list]
        positive_map = create_positive_map_from_span(self.tokenlizer(captions), tokenspanlist)
        return positive_map, captions

    def execute(self, outputs, target_sizes, category_list, global_ids):
        """
        category_list: 当前 Chunk 的类别名列表
        global_ids: 当前 Chunk 对应的 LVIS 全局 Category ID
        """
        # 1. 动态构建 Map
        positive_map, _ = self.build_positive_map(category_list)
        # positive_map shape: (num_classes_in_chunk, 256)
        
        # 2. 计算 Logits
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob_to_token = out_logits.sigmoid()  # bs, 100, 256
        
        # (bs, 100, 256) @ (num_chunk, 256).T -> (bs, 100, num_chunk)
        prob_to_label = prob_to_token @ positive_map.transpose(0, 1)

        # 3. 选 TopK
        # 注意：这里是在当前 Chunk 内选 TopK，外部循环还需要合并所有 Chunk 的结果
        prob = prob_to_label
        # 展平: (bs, 100 * num_chunk)
        topk_values, topk_indexes = jt.topk(
            prob.reshape(out_logits.shape[0], -1), self.num_select, dim=1)
        
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2] # 除以类别数，得到 query index (0-900)
        labels_local = topk_indexes % prob.shape[2] # 得到当前 chunk 内的类别 index
        
        # 4. 解码 Box
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = jt.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = jt.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        for idx in range(scores.shape[0]):
            # 映射回全局 ID
            # labels_local[idx] 是 tensor，需要转为 python list 来索引 global_ids
            local_indices = labels_local[idx].numpy().tolist()
            final_labels = [global_ids[i] for i in local_indices]
            
            results.append(
                {
                    'scores': scores[idx], 
                    'labels': jt.array(final_labels), 
                    'boxes': boxes[idx]
                }
            )
        return results

class LvisGroundingEvaluator:
    def __init__(self, lvis_gt: LVIS):
        self.lvis_gt = lvis_gt
        self.results = []
        self.processed_img_ids = set()

    def update(self, predictions):
        # predictions: dict {img_id: {'boxes':..., 'scores':..., 'labels':...}}
        for img_id, res in predictions.items():
            self.processed_img_ids.add(int(img_id))
            boxes = res['boxes'].numpy()
            scores = res['scores'].numpy()
            labels = res['labels'].numpy()
            
            for i in range(len(boxes)):
                x, y, x2, y2 = boxes[i]
                w = x2 - x
                h = y2 - y
                
                res_item = {
                    "image_id": int(img_id),
                    "category_id": int(labels[i]),
                    "bbox": [float(x), float(y), float(w), float(h)], 
                    "score": float(scores[i])
                }
                self.results.append(res_item)
                
    def summarize_subset(self):
        """
        只评估当前已经处理过的图片子集
        """
        if not self.results:
            return

        print(f"\n[Intermediate Eval] Evaluating on {len(self.processed_img_ids)} images...")
        
        # 1. 加载当前结果
        # LVIS API 加载结果可能会有点慢，随着结果增多
        try:
            lvis_results = LVISResults(self.lvis_gt, self.results)
        except Exception as e:
            print(f"Error loading intermediate results: {e}")
            return

        # 2. 初始化评估器
        lvis_eval = LVISEval(self.lvis_gt, lvis_results, 'bbox')
        
        # 3. 【关键步骤】只评估已处理的 Image IDs
        # 如果不设置这个，分母会是整个数据集，导致 AP 非常低
        lvis_eval.params.img_ids = list(self.processed_img_ids)
        
        # 4. 运行评估
        # 为了防止控制台输出过多刷屏，可以选择性地抑制部分输出，这里保留标准输出
        lvis_eval.run()
        lvis_eval.print_results()
        print("-" * 50)
                
    def save_results(self, output_file):
        """保存原始结果到 JSON 文件"""
        if not self.results:
            print("Warning: No results to save.")
            return
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f"Saving {len(self.results)} results to {output_file} ...")
        with open(output_file, 'w') as f:
            json.dump(self.results, f)
        print("Done.")

    def summarize(self):
        if len(self.results) == 0:
            print("No detections!")
            return
        print(f"Total detections: {len(self.results)}")
        # 创建 LVIS 结果对象
        lvis_results = LVISResults(self.lvis_gt, self.results)
        lvis_eval = LVISEval(self.lvis_gt, lvis_results, 'bbox')
        lvis_eval.run()
        lvis_eval.print_results()

def _iter_batches(dataset, batch_size=1):
    total = len(dataset)
    for start in range(0, total, batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, total))]
        yield collate_fn(batch)

def _load_bert_from_checkpoint(model, checkpoint_path: str) -> None:
    # (保持原样)
    ckpt = _load_checkpoint_any(checkpoint_path)
    if isinstance(ckpt, dict):
        sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        sd = ckpt
    if not isinstance(sd, dict):
        return
    bert_sd = {}
    for k, v in sd.items():
        k_norm = k
        if k_norm.startswith("module."):
            k_norm = k_norm[len("module."):]
        if k_norm.startswith("bert."):
            bert_sd[k_norm[len("bert."):]] = v
    if not bert_sd:
        print("bert load skipped: no bert.* keys in checkpoint", flush=True)
        return
    bert = getattr(model, "bert", None)
    if bert is not None and hasattr(bert, "load_bert_state_dict"):
        try:
            load_res = bert.load_bert_state_dict(bert_sd, strict=False)
        except Exception:
            pass

def chunk_list(lst, n):
    """将列表分成大小为 n 的块"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main(args):
    # config
    cfg = SLConfig.fromfile(args.config_file)

    # build model
    model = load_model(args.config_file, args.checkpoint_path, device=args.device)
    _load_bert_from_checkpoint(model, args.checkpoint_path)
    model.eval()

    # build dataloader
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # 使用 LVIS Dataset
    dataset = LVISDetection(args.image_dir, args.anno_path, transforms=transform)
    
    # build post processor
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessLVIS(num_select=args.num_select, tokenlizer=tokenlizer)
    
    # build evaluator
    evaluator = LvisGroundingEvaluator(dataset.lvis)

    # 准备 LVIS 类别和 ID
    # LVIS API 中 dataset['categories'] 包含了所有类别信息
    all_cats = dataset.lvis.dataset['categories']
    # 按照 ID 排序或者不做排序，但在切分时必须保证 ID 和 Name 对齐
    # 格式: [{'id': 1, 'name': 'aerosol_can'}, ...]
    
    # 配置分块大小 (Chunk Size)
    # LVIS 类别名很多，通常建议 32 或 50 一组，以免超出 Text Encoder 长度
    CHUNK_SIZE = 64
    cat_chunks = list(chunk_list(all_cats, CHUNK_SIZE))
    
    print(f"Total LVIS categories: {len(all_cats)}")
    print(f"Split into {len(cat_chunks)} chunks for inference.")

    # run inference
    start = time.time()
    total = len(dataset)
    print("Dataset length:", total)
    bs = 1

    # 注意：LVIS 评估为了显存安全，建议 Batch Size = 1
    for i, (images, targets) in enumerate(_iter_batches(dataset, batch_size=bs)):
        bs = images.tensors.shape[0]
        orig_target_sizes = jt.stack([t["orig_size"] for t in targets], dim=0)
        
        all_results_for_batch = [[] for _ in range(bs)] # 存储每张图的所有 chunk 结果

        # 对每个 Chunk 进行推理
        for _, cat_chunk in enumerate(cat_chunks):
            chunk_names = [c['name'] for c in cat_chunk]
            chunk_ids = [c['id'] for c in cat_chunk] # 全局 LVIS IDs
            
            # 构建 Prompt
            caption = " . ".join(chunk_names) + ' .'
            input_captions = [caption] * bs
            
            with jt.no_grad():
                # 只有第一次 chunk 需要设置 image tensor (如果模型支持 cache)，
                # 但 GroundingDINO Jittor 版通常每次 forward 都会处理，
                # 为了安全起见，每次 unset
                model.unset_image_tensor() 
                outputs = model(images, captions=input_captions)
                
                # 传入当前 chunk 的 names 和 ids 进行后处理
                chunk_results = postprocessor(outputs, orig_target_sizes, chunk_names, chunk_ids)
                
                # 收集结果
                for b_i in range(bs):
                    all_results_for_batch[b_i].append(chunk_results[b_i])

        # 合并所有 Chunk 的结果并进行 NMS (可选，简单起见这里直接 TopK 或 concat)
        # 此时 all_results_for_batch[0] 是一个 list，包含了 len(chunks) 个 dict
        final_batch_results = []
        for b_i in range(bs):
            img_results = all_results_for_batch[b_i]
            # 拼接所有 chunk 的 boxes, scores, labels
            cat_boxes = jt.concat([r['boxes'] for r in img_results], dim=0)
            cat_scores = jt.concat([r['scores'] for r in img_results], dim=0)
            cat_labels = jt.concat([r['labels'] for r in img_results], dim=0)
            
            # 由于分块推理可能会产生很多低分框，建议再做一次全局过滤/NMS
            # 这里简单做个 TopK 过滤，保留前 300 个
            if cat_scores.shape[0] > args.num_select:
                vals, idxs = jt.topk(cat_scores, args.num_select)
                cat_boxes = cat_boxes[idxs]
                cat_scores = vals
                cat_labels = cat_labels[idxs]

            final_batch_results.append({
                'boxes': cat_boxes,
                'scores': cat_scores,
                'labels': cat_labels
            })

        cocogrounding_res = {
            target["image_id"]: output for target, output in zip(targets, final_batch_results)
        }
        evaluator.update(cocogrounding_res)

        if (i + 1) % 10 == 0:
            used_time = time.time() - start
            eta = (total - i - 1) / (i + 1) * used_time
            print(f"processed {(i + 1) * bs}/{total} images. time: {used_time:.2f}s, ETA: {eta:.2f}s = {eta / 3600:.2f}h")
        
        if (i + 1) % 1000 == 0: # summarize per 1k images; delete when want to test the full dataset
            evaluator.summarize_subset()
            
    if args.output_dir:
        res_file = os.path.join(args.output_dir, "results.json")
        evaluator.save_results(res_file)

    print("all images processed")

    evaluator.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO eval on LVIS", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda", help="running device")
    parser.add_argument("--num_select", type=int, default=300, help="number of topk to select")
    parser.add_argument("--anno_path", type=str, required=True, help="lvis annotation json path")
    parser.add_argument("--image_dir", type=str, required=True, help="image dir path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="dir to save json results")
    
    args = parser.parse_args()
    main(args)
