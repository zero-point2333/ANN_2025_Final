import argparse
import time
import jittor as jt
import jittor.nn as nn
import torchvision
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.inference import _load_checkpoint_any, load_model
from groundingdino.util.misc import collate_fn
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)  # target: list

        # import ipdb; ipdb.set_trace()

        w, h = img.size
        boxes = [obj["bbox"] for obj in target]
        boxes = jt.array(boxes, dtype=jt.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        boxes[:, 0::2].clamp_(min_v=0, max_v=w)
        boxes[:, 1::2].clamp_(min_v=0, max_v=h)
        # filt invalid boxes/masks/keypoints
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        target_new = {}
        image_id = self.ids[idx]
        target_new["image_id"] = image_id
        target_new["boxes"] = boxes
        target_new["orig_size"] = jt.array([int(h), int(w)])

        if self._transforms is not None:
            img, target = self._transforms(img, target_new)

        return img, target


class PostProcessCocoGrounding(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=300, coco_api=None, tokenlizer=None) -> None:
        super().__init__()
        self.num_select = num_select

        assert coco_api is not None
        category_dict = coco_api.dataset['categories']
        cat_list = [item['name'] for item in category_dict]
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = create_positive_map_from_span(
            tokenlizer(captions), tokenspanlist)  # 80, 256. normed

        id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                  41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

        # build a mapping from label_id to pos_map
        new_pos_map = jt.zeros((91, 256), dtype=jt.float32)
        for k, v in id_map.items():
            new_pos_map[v] = positive_map[k]
        self.positive_map = new_pos_map
    def execute(self, outputs, target_sizes, not_to_xyxy=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # pos map to logit
        prob_to_token = out_logits.sigmoid()  # bs, 100, 256
        pos_maps = self.positive_map
        # (bs, 100, 256) @ (91, 256).T -> (bs, 100, 91)
        prob_to_label = prob_to_token @ pos_maps.transpose(0, 1)
        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()
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

        # and from relative [0, 1] to absolute [0, height] coordinates
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
            missing = getattr(load_res, "missing_keys", None)
            unexpected = getattr(load_res, "unexpected_keys", None)
            missing_n = len(missing) if missing is not None else 0
            unexpected_n = len(unexpected) if unexpected is not None else 0
            print(
                f"bert load: keys={len(bert_sd)} missing={missing_n} unexpected={unexpected_n}",
                flush=True,
            )
        except Exception:
            pass
def main(args):
    # config
    cfg = SLConfig.fromfile(args.config_file)

    # build model
    model = load_model(args.config_file, args.checkpoint_path, device=args.device)
    _load_bert_from_checkpoint(model, args.checkpoint_path)
    model.eval()
    # build dataloader
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = CocoDetection(
        args.image_dir, args.anno_path, transforms=transform)
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
    for i, (images, targets) in enumerate(_iter_batches(dataset, batch_size=10)):
        # get images and captions
        bs = images.tensors.shape[0]
        input_captions = [caption] * bs

        # feed to the model
        with jt.no_grad():
            model.unset_image_tensor()
            outputs = model(images, captions=input_captions)
            orig_target_sizes = jt.stack(
                [t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)
        cocogrounding_res = {
            target["image_id"]: output for target, output in zip(targets, results)}
        evaluator.update(cocogrounding_res)
        print(f"inferred {i}-th batch")

        if (i+1) % 30 == 0:
            used_time = time.time() - start
            eta = total / (i+1e-5) * used_time - used_time
            print(
                f"processed {i}/{total} images. time: {used_time:.2f}s, ETA: {eta:.2f}s")
    print("all images processed")

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    print("Final results:", evaluator.coco_eval["bbox"].stats.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Grounding DINO eval on COCO", add_help=True)
    # load model
    parser.add_argument("--config_file", "-c", type=str,
                        required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--device", type=str, default="cuda",
                        help="running device (default: cuda)")

    # post processing
    parser.add_argument("--num_select", type=int, default=300,
                        help="number of topk to select")

    # coco info
    parser.add_argument("--anno_path", type=str,
                        required=True, help="coco root")
    parser.add_argument("--image_dir", type=str,
                        required=True, help="coco image dir")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for dataloader")
    args = parser.parse_args()

    main(args)