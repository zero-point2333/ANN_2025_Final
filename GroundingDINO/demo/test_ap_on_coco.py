# test_ap_on_coco_jittor.py
import argparse
import time
import os
import numpy as np
from PIL import Image

import jittor as jt
import jittor.nn as nn
import jittor.transform as JTransform

# 引入 pycocotools 替代 torchvision.datasets.CocoDetection
from pycocotools.coco import COCO

from groundingdino.datasets.transforms import resize
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.inference import _load_checkpoint_any, load_model
from groundingdino.util.misc import collate_fn
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator

# ==========================================
# 1. 自定义且不依赖 Torchvision 的 COCO Dataset
# ==========================================
class JittorCocoDetection:
    def __init__(self, img_folder, ann_file):
        self.root = img_folder
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # 预定义 Normalize 参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.ids)

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(imgIds=id))

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = self._load_image(img_id)
        target = self._load_target(img_id)

        w, h = img.size
        
        # 处理 Box：从 COCO 格式提取并转换为 Jittor 格式
        boxes = [obj["bbox"] for obj in target]
        # 注意：如果一张图没有 box，需要处理空的情况
        if len(boxes) > 0:
            boxes = jt.array(boxes, dtype=jt.float32).reshape(-1, 4)
            # xywh -> xyxy
            boxes[:, 2:] += boxes[:, :2] 
            boxes[:, 0::2].clamp_(min_v=0, max_v=w)
            boxes[:, 1::2].clamp_(min_v=0, max_v=h)
            
            # 过滤无效框
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
        else:
            boxes = jt.zeros((0, 4), dtype=jt.float32)

        target_new = {}
        target_new["image_id"] = img_id
        target_new["boxes"] = boxes
        target_new["orig_size"] = jt.array([int(h), int(w)])

        img_res, target_res = resize(img, target_new, 800, 1333)
        img_np = np.array(img_res).astype(np.float32) / 255.0
        img_np = img_np.transpose((2, 0, 1))
        img_np = (img_np - self.mean) / self.std
        img_res = jt.array(img_np)
        return img_res, target_res

# ==========================================
# PostProcessor (保持不变，确保 Jittor 语法)
# ==========================================
class PostProcessCocoGrounding(nn.Module):
    def __init__(self, num_select=300, coco_api=None, tokenlizer=None) -> None:
        super().__init__()
        self.num_select = num_select
        assert coco_api is not None
        category_dict = coco_api.dataset['categories']
        cat_list = [item['name'] for item in category_dict]
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = create_positive_map_from_span(
            tokenlizer(captions), tokenspanlist)

        id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                  41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

        new_pos_map = jt.zeros((91, 256), dtype=jt.float32)
        for k, v in id_map.items():
            new_pos_map[v] = positive_map[k]
        self.positive_map = new_pos_map

    def execute(self, outputs, target_sizes, not_to_xyxy=False):
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map
        prob_to_label = prob_to_token @ pos_maps.transpose(0, 1)

        assert out_logits.shape[0] == target_sizes.shape[0]
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = jt.topk(
            prob.reshape(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]

        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = jt.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = jt.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = []
        for idx in range(scores.shape[0]):
            results.append(
                {'scores': scores[idx], 'labels': labels[idx], 'boxes': boxes[idx]}
            )
        return results

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
            print(f"bert loaded with keys: {len(bert_sd)}", flush=True)
        except Exception:
            pass

def main(args):
    # config
    cfg = SLConfig.fromfile(args.config_file)

    # build model
    model = load_model(args.config_file, args.checkpoint_path, device=args.device)
    _load_bert_from_checkpoint(model, args.checkpoint_path)
    model.eval()

    # ==========================================
    # 3. 使用自定义的 Dataset 类 (移除了 Transform 参数，因为逻辑内嵌了)
    # ==========================================
    dataset = JittorCocoDetection(
        args.image_dir, args.anno_path)

    # build post processor
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessCocoGrounding(
        num_select=args.num_select, coco_api=dataset.coco, tokenlizer=tokenlizer)
    
    # build evaluator
    evaluator = CocoGroundingEvaluator(
        dataset.coco, iou_types=("bbox",), useCats=True)

    # build captions
    category_dict = dataset.coco.dataset['categories']
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    # run inference
    start = time.time()
    total = len(dataset)
    print("Dataset length:", total)
    
    # 注意：如果显存不足，减小 batch_size
    for i, (images, targets) in enumerate(_iter_batches(dataset, batch_size=args.batch_size)):
        # get images and captions
        bs = images.tensors.shape[0]
        input_captions = [caption] * bs

        # feed to the model
        with jt.no_grad():
            # Jittor 不需要 explicit 的 to(device)，但为了保险起见
            # images = images.to(args.device) 
            
            # 如果 model 内部有对 image tensor 的特殊处理，可能需要 unset
            # model.unset_image_tensor() # GroundingDINO specific
            
            outputs = model(images, captions=input_captions)
            
            # 获取 targets 中的 orig_size (Jittor Var)
            orig_target_sizes = jt.stack(
                [t["orig_size"] for t in targets], dim=0)
            
            results = postprocessor(outputs, orig_target_sizes)
            
        cocogrounding_res = {
            target["image_id"]: output for target, output in zip(targets, results)}
        evaluator.update(cocogrounding_res)

        if (i + 1) % 10 == 0:
            used_time = time.time() - start
            eta = total / ((i + 1) * args.batch_size) * used_time - used_time
            print(f"processed {(i + 1) * args.batch_size}/{total} images. time: {used_time:.2f}s")
            
    print("all images processed")

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    print("Final results:", evaluator.coco_eval["bbox"].stats.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Grounding DINO eval on COCO (Jittor Version)", add_help=True)
    # load model
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_select", type=int, default=300)

    # coco info
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for inference")
    
    args = parser.parse_args()
    main(args)
