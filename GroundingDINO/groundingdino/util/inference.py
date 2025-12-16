from typing import Tuple, List, Any

import os
import cv2
import numpy as np
import supervision as sv
from PIL import Image
import bisect
import jittor as jt
import torch

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap


def convert_pytorch_to_jittor(state_dict):
    """
    Recursively convert PyTorch tensors in state_dict to Jittor Vars.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            new_state_dict[k] = jt.array(v.detach().numpy())
        elif isinstance(v, dict):
            new_state_dict[k] = convert_pytorch_to_jittor(v)
        else:
            new_state_dict[k] = v
    return new_state_dict


def _set_jittor_device(device: str) -> None:
    """
    根据 device 字符串切换 Jittor 的 CUDA 开关。
    保留原来 device 参数接口，但内部改用 jt.flags.use_cuda。
    """
    if device is None:
        return
    device = device.lower()
    if device.startswith("cuda"):
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0


def _to_numpy(x: Any) -> np.ndarray:
    """
    统一把 Jittor Var / 列表 / 元组 转成 numpy.ndarray，
    """
    if isinstance(x, np.ndarray):
        return x
    try:
        if isinstance(x, jt.Var):
            return x.numpy()
    except Exception:
        pass
    return np.asarray(x)


def _box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    替代 torchvision.ops.box_convert，支持 cxcywh -> xyxy 的转换。
    boxes: (..., 4) 格式 [cx, cy, w, h]
    返回: (..., 4) 格式 [x0, y0, x1, y1]
    """
    x_c = boxes[..., 0]
    y_c = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]
    x0 = x_c - 0.5 * w
    y0 = y_c - 0.5 * h
    x1 = x_c + 0.5 * w
    y1 = y_c + 0.5 * h
    return np.stack([x0, y0, x1, y1], axis=-1)


# ----------------------------------------------------------------------------------------------------------------------
# OLD API
# ----------------------------------------------------------------------------------------------------------------------


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda", offline_mode=False):
    _set_jittor_device(device)

    args = SLConfig.fromfile(model_config_path)
    args.device = device

    # 新增：设置离线模式
    if offline_mode:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    model = build_model(args)

    if model_checkpoint_path:
        ckpt = jt.load(model_checkpoint_path)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        state_dict = clean_state_dict(state_dict)
        state_dict = convert_pytorch_to_jittor(state_dict)
        
        # Filter out BERT parameters since BERT is wrapped as PyTorch module
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('bert.')}

        if hasattr(model, "load_state_dict"):
            try:
                model.load_state_dict(filtered_state_dict, strict=False)
            except TypeError:
                model.load_state_dict(filtered_state_dict)
        elif hasattr(model, "load_parameters"):
            model.load_parameters(filtered_state_dict)

    if hasattr(model, "eval"):
        model.eval()
    return model


def load_image(image_path: str) -> Tuple[np.ndarray, Any]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def predict(
        model,
        image: Any,  # 原来是 torch.Tensor，这里放宽为 Any / Jittor Var
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False
) -> Tuple[Any, Any, List[str]]:
    """
    Jittor 版本的预测：
    - 不再使用 torch.no_grad / .cpu() / .sigmoid()(.cpu())；
    - 假定 model 和 image 都是 Jittor 风格的对象；
    - 返回 boxes, scores, phrases 的接口保持不变。
    """
    caption = preprocess_caption(caption=caption)
    _set_jittor_device(device)

    # 这里直接使用传入的 model 和 image
    with jt.no_grad():
        # Convert image to Jittor Var
        if not isinstance(image, jt.Var):
            if hasattr(image, 'convert'):  # PIL Image
                import numpy as np
                image = np.array(image)
            image = jt.array(image)
        outputs = model(image[None], captions=[caption])

    # outputs["pred_logits"]: (nq, vocab_dim)
    # outputs["pred_boxes"]:  (nq, 4)
    prediction_logits = outputs["pred_logits"].sigmoid()[0]
    prediction_boxes = outputs["pred_boxes"][0]

    # 按 query 取最大 logit 作为该 query 的 box score
    max_per_query = prediction_logits.max(dim=1)[0]
    mask = max_per_query > box_threshold

    logits = prediction_logits[mask]          # (n, vocab_dim)
    boxes = prediction_boxes[mask]            # (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    if remove_combined:
        sep_idx = [
            i for i in range(len(tokenized['input_ids']))
            if tokenized['input_ids'][i] in [101, 102, 1012]
        ]

        phrases: List[str] = []
        for logit in logits:
            # logit: (vocab_dim,)
            max_idx = int(logit.argmax())
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrase = get_phrases_from_posmap(
                logit > text_threshold,
                tokenized,
                tokenizer,
                left_idx,
                right_idx,
            ).replace(".", "")
            phrases.append(phrase)
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit in logits
        ]

    # 保持原返回结构： boxes, box_scores, phrases
    box_scores = logits.max(dim=1)[0]
    return boxes, box_scores, phrases


def annotate(image_source: np.ndarray, boxes: Any, logits: Any, phrases: List[str]) -> np.ndarray:
    h, w, _ = image_source.shape

    boxes_np = _to_numpy(boxes)
    logits_np = _to_numpy(logits)

    # 从归一化的 cxcywh 转到实际像素坐标 xyxy
    boxes_scaled = boxes_np * np.array([w, h, w, h], dtype=np.float32)
    xyxy = _box_cxcywh_to_xyxy(boxes_scaled)

    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {float(logit):.2f}"
        for phrase, logit in zip(phrases, logits_np)
    ]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


# ----------------------------------------------------------------------------------------------------------------------
# NEW API
# ----------------------------------------------------------------------------------------------------------------------


class Model:

    def __init__(
        self,
        model_config_path: str,
        model_checkpoint_path: str,
        device: str = "cuda"
    ):
        # load_model 已经内部处理了 device，这里不再 .to(device)
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device
        )
        self.device = device

    def predict_with_caption(
        self,
        image: np.ndarray,
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> Tuple[sv.Detections, List[str]]:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections, labels = model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        """
        processed_image = Model.preprocess_image(image_bgr=image)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits,
        )
        return detections, phrases

    def predict_with_classes(
        self,
        image: np.ndarray,
        classes: List[str],
        box_threshold: float,
        text_threshold: float
    ) -> sv.Detections:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections = model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        """
        caption = ". ".join(classes)
        processed_image = Model.preprocess_image(image_bgr=image)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits,
        )
        class_id = Model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        return detections

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> Any:
        """
        预处理保持原有 transforms.T 接口：
        - image_bgr: OpenCV 读进来的 BGR 图像；
        - 返回值交给 Jittor 版 T.ToTensor 决定类型。
        """
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: Any,
            logits: Any
    ) -> sv.Detections:
        """
        把模型输出的 cxcywh 归一化框 + 置信度，转换为 supervision.Detections。
        """
        boxes_np = _to_numpy(boxes)
        logits_np = _to_numpy(logits)

        boxes_scaled = boxes_np * np.array([source_w, source_h, source_w, source_h], dtype=np.float32)
        xyxy = _box_cxcywh_to_xyxy(boxes_scaled)
        confidence = logits_np
        return sv.Detections(xyxy=xyxy, confidence=confidence)

    @staticmethod
    def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            for class_ in classes:
                if class_ in phrase:
                    class_ids.append(classes.index(class_))
                    break
            else:
                class_ids.append(None)
        return np.array(class_ids)
