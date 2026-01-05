import argparse
import json
import time
import os
import copy
from glob import glob
import numpy as np
from PIL import Image

def _get_dist_info():
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("LOCAL_RANK", str(rank))))
    return rank, world_size, local_rank

_rank, _world_size, _local_rank = _get_dist_info()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)

import jittor as jt
import jittor.nn as nn

# 移除 torchvision 依赖，移除 transforms 依赖
# import torchvision
# import groundingdino.datasets.transforms as T

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

# ==========================================
# 1. 自定义且不依赖 Torchvision 的 LVIS Dataset
# ==========================================
class JittorLVISDetection:
    def __init__(self, img_folder, ann_file):
        self.root = img_folder
        print("Loading LVIS annotations...", flush=True)
        self.lvis = LVIS(ann_file)
        self.ids = list(self.lvis.imgs.keys())
        
        # 预定义 Normalize 参数 (ImageNet Mean/Std)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.ids)
    
    def _resize_image(self, img: Image.Image):
        """
        手动实现 GroundingDINO 的 Resize 逻辑:
        Resize 短边到 800，同时保证长边不超过 1333
        """
        w, h = img.size
        min_size = 800
        max_size = 1333
        
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        
        if max_original_size / min_original_size * min_size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
        else:
            size = min_size
            
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
            
        return img.resize((ow, oh), Image.BILINEAR)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        # LVIS 图片路径处理
        # LVIS val2017 通常对应 COCO val2017 图片
        img_meta = self.lvis.load_imgs([img_id])[0]
        file_name = img_meta['file_name'] # e.g., "000000123456.jpg" 或 "val2017/000000123456.jpg"
        
        # 处理路径拼接
        if os.path.exists(os.path.join(self.root, file_name)):
            img_path = os.path.join(self.root, file_name)
        else:
            # 尝试处理可能的子目录情况
            # 如果 root 指向的是 coco 根目录，而 file_name 只是文件名
            # 这里根据你实际的数据集目录结构可能需要微调
            possible_path = os.path.join(self.root, "val2017", os.path.basename(file_name))
            if os.path.exists(possible_path):
                img_path = possible_path
            else:
                # 回退到直接拼接
                img_path = os.path.join(self.root, os.path.basename(file_name))

        img = Image.open(img_path).convert("RGB")
        w, h = img.size # 原始尺寸

        # -------------------------------
        # 手动 Transform: Resize -> ToTensor -> Normalize
        # -------------------------------
        img_resized = self._resize_image(img)
        
        # PIL -> Numpy -> Jittor
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        # HWC -> CHW
        img_np = img_np.transpose((2, 0, 1))
        # Normalize
        img_np = (img_np - self.mean) / self.std
        
        img_jt = jt.array(img_np)

        # -------------------------------
        # 构造 Target (仅用于传递 orig_size)
        # -------------------------------
        target_new = {}
        target_new["image_id"] = img_id
        target_new["orig_size"] = jt.array([int(h), int(w)]) # 注意: orig_size 通常存 (h, w)
        
        # 注意：GroundingDINO Inference 不需要 GT Box，
        # 如果代码其他地方没有用到 target['boxes']，这里可以省略加载 GT 的过程以加速
        # 为了兼容 collate_fn，这里留空即可
        
        return img_jt, target_new

# ==========================================
# Post Processor (保持 Jittor 逻辑)
# ==========================================
class PostProcessLVIS(nn.Module):
    def __init__(self, num_select=300, tokenlizer=None) -> None:
        super().__init__()
        self.num_select = num_select
        self.tokenlizer = tokenlizer

    def build_positive_map(self, cat_list):
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat.lower()] for cat in cat_list]
        positive_map = create_positive_map_from_span(self.tokenlizer(captions), tokenspanlist)
        return positive_map, captions

    def execute(self, outputs, target_sizes, category_list, global_ids):
        # 1. 动态构建 Map
        positive_map, _ = self.build_positive_map(category_list)
        
        # 2. 计算 Logits
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob_to_token = out_logits.sigmoid()
        
        # (bs, 100, 256) @ (num_chunk, 256).T -> (bs, 100, num_chunk)
        prob_to_label = prob_to_token @ positive_map.transpose(0, 1)

        # 3. 选 TopK (Chunk 内部)
        prob = prob_to_label
        topk_values, topk_indexes = jt.topk(
            prob.reshape(out_logits.shape[0], -1), self.num_select, dim=1)
        
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels_local = topk_indexes % prob.shape[2]
        
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
            # 这里的 .numpy().tolist() 会导致同步，但在循环末尾影响不大
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

# ==========================================
# Evaluator
# ==========================================
class LvisGroundingEvaluator:
    def __init__(self, lvis_gt: LVIS):
        self.lvis_gt = lvis_gt
        self.results = []
        self.processed_img_ids = set()

    def update(self, predictions):
        for img_id, res in predictions.items():
            self.processed_img_ids.add(int(img_id))
            
            # 必须转为 numpy 存入 list
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
        if not self.results: return
        print(f"\n[Intermediate Eval] Evaluating on {len(self.processed_img_ids)} images...")
        try:
            # 抑制 LVIS 加载时的打印
            lvis_results = LVISResults(self.lvis_gt, self.results)
            lvis_eval = LVISEval(self.lvis_gt, lvis_results, 'bbox')
            lvis_eval.params.img_ids = list(self.processed_img_ids)
            lvis_eval.run()
            lvis_eval.print_results()
        except Exception as e:
            print(f"Eval warning: {e}")
        print("-" * 50)
                
    def save_results(self, output_file):
        if not self.results: return
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f"Saving {len(self.results)} results to {output_file} ...")
        with open(output_file, 'w') as f:
            json.dump(self.results, f)

    def summarize(self):
        if len(self.results) == 0:
            print("No detections!")
            return
        print(f"Total detections: {len(self.results)}")
        lvis_results = LVISResults(self.lvis_gt, self.results)
        lvis_eval = LVISEval(self.lvis_gt, lvis_results, 'bbox')
        lvis_eval.run()
        lvis_eval.print_results()

# ==========================================
# Utils
# ==========================================
def _is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _collect_images(image_dir: str):
    images = sorted(
        p for p in glob(os.path.join(image_dir, "**", "*"), recursive=True)
        if os.path.isfile(p) and _is_image_file(p)
    )
    return images

def _map_image_paths_to_lvis_ids(lvis_api, image_paths, image_dir: str):
    name_to_id = {}
    for img_id, info in lvis_api.imgs.items():
        fname = info.get("file_name", "")
        if fname:
            name_to_id.setdefault(fname, img_id)
            name_to_id.setdefault(os.path.basename(fname), img_id)

    selected_ids = []
    missing = 0
    for path in image_paths:
        rel = os.path.relpath(path, image_dir)
        img_id = name_to_id.get(rel) or name_to_id.get(os.path.basename(rel))
        if img_id is None:
            missing += 1
            continue
        selected_ids.append(img_id)
    if missing:
        print(f"Warning: {missing} images not found in LVIS annotations.")
    return selected_ids

def _iter_batches(dataset, batch_size=1):
    total = len(dataset)
    for start in range(0, total, batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, total))]
        yield collate_fn(batch)

def _load_bert_from_checkpoint(model, checkpoint_path: str) -> None:
    ckpt = _load_checkpoint_any(checkpoint_path)
    if isinstance(ckpt, dict):
        sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        sd = ckpt
    if not isinstance(sd, dict): return
    bert_sd = {}
    for k, v in sd.items():
        k_norm = k
        if k_norm.startswith("module."): k_norm = k_norm[len("module."):]
        if k_norm.startswith("bert."): bert_sd[k_norm[len("bert."):]] = v
    if not bert_sd:
        print("bert load skipped", flush=True)
        return
    bert = getattr(model, "bert", None)
    if bert is not None and hasattr(bert, "load_bert_state_dict"):
        try:
            bert.load_bert_state_dict(bert_sd, strict=False)
            print(f"bert loaded with keys: {len(bert_sd)}", flush=True)
        except Exception: pass

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# ==========================================
# Main
# ==========================================
def main(args):
    rank, world_size, local_rank = _get_dist_info()

    images = _collect_images(args.image_dir)
    my_images = images[rank::world_size]
    if not my_images:
        print("No images to process for this rank.")
        return

    # config
    cfg = SLConfig.fromfile(args.config_file)

    # build model
    model = load_model(args.config_file, args.checkpoint_path, device=args.device)
    _load_bert_from_checkpoint(model, args.checkpoint_path)
    model.eval()

    # 1. 使用自定义 Dataset (移除了 transforms 参数)
    dataset = JittorLVISDetection(args.image_dir, args.anno_path)
    dataset.ids = _map_image_paths_to_lvis_ids(dataset.lvis, my_images, args.image_dir)
    if not dataset.ids:
        print("No LVIS images matched for this rank.")
        return

    output_dir = os.path.join(args.output_dir, f"rank{rank}")
    
    # build post processor
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessLVIS(num_select=args.num_select, tokenlizer=tokenlizer)
    
    # build evaluator
    evaluator = LvisGroundingEvaluator(dataset.lvis)

    # LVIS Categories Chunking
    all_cats = dataset.lvis.dataset['categories']
    CHUNK_SIZE = 64 # 可以尝试调大到 50 或 64，取决于显存和文本长度
    cat_chunks = list(chunk_list(all_cats, CHUNK_SIZE))
    
    print(f"Total LVIS categories: {len(all_cats)}")
    print(f"Split into {len(cat_chunks)} chunks for inference.")

    # run inference
    start = time.time()
    total = len(dataset)
    print("Dataset length:", total)
    
    # Force batch size = 1 for safety with varying image sizes and huge label set
    bs = 1 

    for i, (images, targets) in enumerate(_iter_batches(dataset, batch_size=bs)):
        bs_actual = images.tensors.shape[0]
        # Jittor Var, orig_size
        orig_target_sizes = jt.stack([t["orig_size"] for t in targets], dim=0)
        
        all_results_for_batch = [[] for _ in range(bs_actual)]

        # --- Chunk Loop ---
        # 针对每一张图，跑遍所有的类别块
        for _, cat_chunk in enumerate(cat_chunks):
            chunk_names = [c['name'] for c in cat_chunk]
            chunk_ids = [c['id'] for c in cat_chunk]
            
            # Prompt Construction
            caption = " . ".join(chunk_names) + ' .'
            input_captions = [caption] * bs_actual
            
            with jt.no_grad():
                model.unset_image_tensor() 
                outputs = model(images, captions=input_captions)
                
                # Post Process for this chunk
                chunk_results = postprocessor(outputs, orig_target_sizes, chunk_names, chunk_ids)
                
                for b_i in range(bs_actual):
                    all_results_for_batch[b_i].append(chunk_results[b_i])

        # --- Merge Chunks ---
        final_batch_results = []
        for b_i in range(bs_actual):
            img_results = all_results_for_batch[b_i]
            
            cat_boxes = jt.concat([r['boxes'] for r in img_results], dim=0)
            cat_scores = jt.concat([r['scores'] for r in img_results], dim=0)
            cat_labels = jt.concat([r['labels'] for r in img_results], dim=0)
            
            # Global TopK Filter
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
            print(f"Processed {i + 1}/{total}. Time: {used_time:.1f}s, ETA: {eta/60:.1f}min")
        
        # Intermediate eval every 2000 images
        if (i + 1) % 2000 == 0: 
            evaluator.summarize_subset()
            
    if args.output_dir:
        res_file = os.path.join(output_dir, "results.json")
        evaluator.save_results(res_file)

    print("All processed. Final evaluation:")
    evaluator.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO eval on LVIS (Jittor)", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_select", type=int, default=300)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    
    args = parser.parse_args()
    main(args)
