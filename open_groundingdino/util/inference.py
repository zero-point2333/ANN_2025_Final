from typing import Tuple, List, Any, Dict, Optional
import os
import sys
import bisect
from pathlib import Path


import numpy as np
from PIL import Image

from util.debug_tools import debug_enabled, log_tensor, log_text

def _ensure_jittor_cache_dir() -> None:
    """
    Jittor writes a compiler/cache folder under HOME/.cache/jittor.
    In the grading sandbox /home may be read-only, so redirect HOME/JITTOR_HOME
    to a repo-local folder if we cannot create the default path.
    """
    repo_root = Path(__file__).resolve().parents[3]
    safe_home = repo_root / ".jittor_home"

    orig_home = Path(os.environ.get("HOME", str(Path.home())))
    os.environ.setdefault("GROUNDINGDINO_ORIG_HOME", str(orig_home))

    # Prefer existing HF/ModelScope caches from the original home to avoid downloads.
    orig_modelscope = orig_home / ".cache" / "modelscope"
    modelscope_cache = orig_modelscope if orig_modelscope.exists() else (safe_home / ".cache" / "modelscope")
    orig_hf_cache = orig_home / ".cache" / "huggingface" / "hub"
    hf_cache = orig_hf_cache if orig_hf_cache.exists() else (safe_home / ".cache" / "huggingface" / "hub")
    os.environ.setdefault("MODELSCOPE_CACHE", str(modelscope_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache))
    if str(modelscope_cache).startswith(str(safe_home)):
        modelscope_cache.mkdir(parents=True, exist_ok=True)
    if str(hf_cache).startswith(str(safe_home)):
        Path(hf_cache).mkdir(parents=True, exist_ok=True)

    def _can_write_home(home_path: Path) -> bool:
        try:
            probe = home_path / ".cache" / "jittor" / "_perm_test"
            probe.mkdir(parents=True, exist_ok=True)
            test_file = probe / "touch.txt"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            return True
        except Exception:
            return False

    # Use current HOME if writable; otherwise fall back into the repo.
    current_home = orig_home
    if not _can_write_home(current_home):
        safe_home.mkdir(parents=True, exist_ok=True)
        (safe_home / ".cache" / "jittor").mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(safe_home)
        os.environ["JITTOR_HOME"] = str(safe_home)


def _ensure_python_config() -> None:
    """
    Point jittor to the correct pythonX.Y-config from the current interpreter.
    """
    if "python_config_path" in os.environ:
        return
    exe = Path(sys.executable).resolve()
    candidate = exe.with_name(f"python{sys.version_info.major}.{sys.version_info.minor}-config")
    if candidate.exists():
        os.environ["python_config_path"] = str(candidate)


_ensure_jittor_cache_dir()
_ensure_python_config()
# Prevent Jittor from auto-downloading CUDA when we explicitly run CPU-only.
os.environ.setdefault("nvcc_path", "")
# Avoid multiprocessing pool (semlock permission issues in some sandboxes).
os.environ.setdefault("DISABLE_MULTIPROCESSING", "1")

import jittor as jt

# Optional deps (only used by annotate / Model wrapper)
try:
    import cv2
except Exception:
    cv2 = None

try:
    import supervision as sv
except Exception:
    sv = None

# Torch is used to load .pth (PyTorch zip checkpoint)
try:
    import torch
except Exception:
    torch = None

import datasets.transforms as T
from models import build_model
from util.misc import clean_state_dict
from util.slconfig import SLConfig
from util.utils import get_phrases_from_posmap


def _set_jittor_device(device: str) -> None:
    if device is None:
        return
    d = device.lower()
    if d.startswith("cuda"):
        try:
            jt.flags.use_cuda = 1
        except RuntimeError:
            # CUDA not available, keep CPU mode
            jt.flags.use_cuda = 0
    else:
        jt.flags.use_cuda = 0


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, jt.Var):
        return x.numpy()
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    # boxes: (..., 4) [cx, cy, w, h] -> (..., 4) [x0, y0, x1, y1]
    x_c = boxes[..., 0]
    y_c = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]
    x0 = x_c - 0.5 * w
    y0 = y_c - 0.5 * h
    x1 = x_c + 0.5 * w
    y1 = y_c + 0.5 * h
    return np.stack([x0, y0, x1, y1], axis=-1)


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    return result if result.endswith(".") else (result + ".")
def _load_checkpoint_any(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pth", ".pt"]:
        if torch is None:
            raise RuntimeError(f"torch is required to load {ext} checkpoints: {path}")
        return torch.load(path, map_location="cpu")
    return jt.load(path)
def _extract_state_dict(ckpt: Any) -> Dict[str, Any]:
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    if not isinstance(sd, dict):
        raise TypeError(f"Checkpoint state_dict is not a dict, got: {type(sd)}")
    return clean_state_dict(sd)


def _torch_tensor_to_np(v: Any) -> np.ndarray:
    if torch is not None and isinstance(v, torch.Tensor):
        arr = v.detach().cpu().numpy()
    else:
        arr = np.asarray(v)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def _try_load_bert_weights(model: Any, full_state_dict: Dict[str, Any]) -> None:
    """
    If model.bert is a torch module, load bert.* weights into it (best effort).
    """
    if torch is None:
        return
    bert = getattr(model, "bert", None)
    if bert is None:
        return

    bert_sd = {k[len("bert."):]: v for k, v in full_state_dict.items() if k.startswith("bert.")}
    if not bert_sd:
        return

    try:
        bert_sd_torch = {}
        for k, v in bert_sd.items():
            if torch.is_tensor(v):
                bert_sd_torch[k] = v.detach().cpu()
            elif isinstance(v, jt.Var):
                bert_sd_torch[k] = torch.from_numpy(v.numpy())
            else:
                bert_sd_torch[k] = torch.from_numpy(np.asarray(v))

        if hasattr(bert, "load_bert_state_dict"):
            bert.load_bert_state_dict(bert_sd_torch, strict=False)
        elif hasattr(bert, "load_state_dict"):
            bert.load_state_dict(bert_sd_torch, strict=False)
    except Exception:
        # keep best-effort; do not block the whole model
        pass


def _filter_by_model_shape_best_effort(model: Any, sd_np: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Best-effort: keep only keys that exist in model.state_dict() and match shape.
    If a 2D weight is transposed, auto-transpose.
    Only include parameters, not buffers.
    """
    try:
        msd = model.state_dict()
        if not isinstance(msd, dict):
            return sd_np
    except Exception:
        return sd_np

    # Get parameters only, exclude buffers
    try:
        params = dict(model.named_parameters())
        param_names = set(params.keys())
    except Exception:
        param_names = set(msd.keys())

    out: Dict[str, np.ndarray] = {}
    skipped_keys = []
    transposed_keys = []
    for k, arr in sd_np.items():
        if k not in msd:
            skipped_keys.append(k)
            continue
        if k not in param_names:
            # Skip buffers
            continue
        tgt = msd[k]
        tgt_shape = getattr(tgt, "shape", None)
        if tgt_shape is None:
            tgt_shape = np.asarray(tgt).shape

        if tuple(arr.shape) == tuple(tgt_shape):
            out[k] = arr
            continue

        # common: linear weights transposed
        if arr.ndim == 2 and tuple(arr.T.shape) == tuple(tgt_shape):
            out[k] = np.ascontiguousarray(arr.T)
            transposed_keys.append(k)
            continue

        # otherwise skip
        skipped_keys.append(k)
        continue

    if debug_enabled():
        log_text(f"Skipped params (not in model or shape mismatch): {len(skipped_keys)} keys", force=True)
        if skipped_keys:
            log_text(f"Sample skipped: {skipped_keys[:5]}", force=True)
        log_text(f"Transposed params: {len(transposed_keys)} keys", force=True)
        if transposed_keys:
            log_text(f"Sample transposed: {transposed_keys[:5]}", force=True)
        log_text(f"Matched params: {len(out)} keys", force=True)

    return out


def _filter_by_model_shape_best_effort_jt(model: Any, sd_jt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort: keep only keys that exist in model.state_dict() and match shape.
    If a 2D weight is transposed, auto-transpose.
    For Jittor tensors.
    """
    try:
        msd = model.state_dict()
        if not isinstance(msd, dict):
            return sd_jt
    except Exception:
        return sd_jt

    # Get parameters only, exclude buffers
    try:
        params = dict(model.named_parameters())
        param_names = set(params.keys())
    except Exception:
        param_names = set(msd.keys())

    out: Dict[str, Any] = {}
    skipped_keys = []
    transposed_keys = []
    for k, var in sd_jt.items():
        if k not in msd:
            skipped_keys.append(k)
            continue

        tgt = msd[k]
        tgt_shape = getattr(tgt, "shape", None)
        if tgt_shape is None:
            tgt_shape = np.asarray(tgt).shape

        if tuple(var.shape) == tuple(tgt_shape):
            out[k] = var
            continue

        # common: linear weights transposed
        if var.ndim == 2 and tuple(var.T.shape) == tuple(tgt_shape):
            out[k] = var.T
            transposed_keys.append(k)
            continue

        # otherwise skip
        skipped_keys.append(k)
        continue

    if debug_enabled():
        log_text(f"Skipped params (not in model or shape mismatch): {len(skipped_keys)} keys", force=True)
        if skipped_keys:
            log_text(f"Sample skipped: {skipped_keys[:5]}", force=True)
        log_text(f"Transposed params: {len(transposed_keys)} keys", force=True)
        if transposed_keys:
            log_text(f"Sample transposed: {transposed_keys[:5]}", force=True)
        log_text(f"Matched params: {len(out)} keys", force=True)

    return out


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda", offline_mode: bool = False):
    _set_jittor_device(device)

    args = SLConfig.fromfile(model_config_path)
    args.device = device

    if offline_mode:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    model = build_model(args)

    if model_checkpoint_path:
        ckpt = _load_checkpoint_any(model_checkpoint_path)
        full_sd = _extract_state_dict(ckpt)

        # (Optional) load bert.* into model.bert if it's a torch module
        _try_load_bert_weights(model, full_sd)

        # Convert NON-bert params to Jittor tensors for loading
        sd_jt = {}
        for k, v in full_sd.items():
            if k.startswith("bert."):
                continue
            arr = _torch_tensor_to_np(v)
            sd_jt[k] = jt.array(arr)

        # Reduce mismatches (avoid大量 load failed / 更稳)
        sd_jt = _filter_by_model_shape_best_effort_jt(model, sd_jt)
        if debug_enabled():
            total_keys = len(full_sd)
            matched_keys = len(sd_jt)
            log_text(
                f"load_model: matched params {matched_keys}/{total_keys} "
                f"(bert params skipped: {total_keys - len(full_sd)} not counted)",
                force=True,
            )
        try:
            try:
                load_res = model.load_state_dict(sd_jt, strict=False)
            except TypeError:
                load_res = model.load_state_dict(sd_jt)
            if isinstance(load_res, tuple) and len(load_res) == 2:
                missing, unexpected = load_res
            else:
                missing, unexpected = [], []
            if debug_enabled():
                log_text(f"missing sample: {[k for k in missing if 'bbox_embed' in k][:10]}", force=True)
                log_text(f"unexpected sample: {[k for k in unexpected if 'bbox_embed' in k][:10]}", force=True)
        except Exception as e:
            if debug_enabled():
                log_text(f"load_state_dict failed: {e}", force=True)
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
        image,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False,
        debug: bool = False
):
    """
    Safe predict for Jittor:
    - Always convert torch.Tensor / numpy / PIL to a contiguous float32 numpy copy before jt.array()
    - Guard against empty candidates to avoid reduce/max on empty tensors
    - Do masking / thresholds in numpy to avoid any potential Jittor bool quirks
    """
    caption = preprocess_caption(caption=caption)
    debug_mode = debug_enabled(debug)
    if debug_mode:
        log_text(
            f"predict(): device={device} box_th={box_threshold} text_th={text_threshold} caption='{caption}'",
            force=True,
        )
        log_tensor("predict.input.image_raw", image, force=True)
    _set_jittor_device(device)

    # --- SAFE input conversion: avoid jt.array(torch.Tensor) ---
    if not isinstance(image, jt.Var):
        import numpy as np
        try:
            import torch
            is_torch = isinstance(image, torch.Tensor)
        except Exception:
            is_torch = False

        if is_torch:
            # detach -> cpu -> numpy, then force a real contiguous copy
            image = image.detach().cpu().numpy()
            image = np.array(image, dtype=np.float32, copy=True, order="C")
        elif hasattr(image, "convert"):  # PIL.Image
            image = np.array(image, dtype=np.float32, copy=True, order="C")
        else:
            image = np.array(image, dtype=np.float32, copy=True, order="C")

        image = jt.array(image)
    if debug_mode:
        log_tensor("predict.input.image_jt", image, force=True)

    with jt.no_grad():
        outputs = model(image[None], captions=[caption])
    if debug_mode:
        for k, v in outputs.items():
            log_tensor(f"predict.outputs.{k}", v, force=True)

    # --- Convert outputs to numpy and apply sigmoid/clamp in numpy to avoid surprises ---
    raw_logits = outputs["pred_logits"]
    raw_boxes = outputs["pred_boxes"]

    logits_np = _to_numpy(raw_logits)
    boxes_np = _to_numpy(raw_boxes)

    # squeeze batch dim if present
    if logits_np.ndim == 3 and logits_np.shape[0] == 1:
        logits_np = logits_np[0]
    if boxes_np.ndim == 3 and boxes_np.shape[0] == 1:
        boxes_np = boxes_np[0]

    # numerical-stable sigmoid in numpy
    logits_np = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -50.0, 50.0)))
    logits_np = np.nan_to_num(logits_np, nan=0.0, posinf=1.0, neginf=0.0)
    boxes_np = boxes_np.astype(np.float32, copy=False)
    boxes_np = np.nan_to_num(boxes_np, nan=0.0, posinf=1.0, neginf=0.0)
    boxes_np = np.clip(boxes_np, 0.0, 1.0)
    if debug_mode:
        log_tensor("predict.pred_logits_np", logits_np, force=True)
        log_tensor("predict.pred_boxes_np", boxes_np, force=True)
    prediction_logits: np.ndarray = logits_np
    prediction_boxes: np.ndarray = boxes_np

    # --- Guard: num_queries==0 ---
    if len(prediction_logits.shape) == 0 or prediction_logits.shape[0] == 0:
        empty_boxes = jt.array(np.zeros((0, 4), dtype=np.float32))
        empty_scores = jt.array(np.zeros((0,), dtype=np.float32))
        return empty_boxes, empty_scores, []

    max_per_query = prediction_logits.max(axis=1)
    if debug_mode:
        log_tensor("predict.max_per_query", max_per_query, force=True)
    mask = max_per_query > box_threshold
    if debug_mode:
        try:
            true_count = int(mask.sum())
        except Exception:
            true_count = "?"
        log_text(f"predict.mask shape={mask.shape} true={true_count}", force=True)

    logits = prediction_logits[mask]
    boxes = prediction_boxes[mask]
    boxes = np.ascontiguousarray(boxes, dtype=np.float32)

    # --- Guard: no boxes pass threshold ---
    if logits.shape[0] == 0:
        empty_boxes = jt.array(boxes)
        empty_scores = jt.array(np.zeros((0,), dtype=np.float32))
        if debug_mode:
            log_text("predict: no boxes passed threshold", force=True)
        return empty_boxes, empty_scores, []

    tokenizer = model.tokenizer
    max_text_len = getattr(model, "max_text_len", 256)
    tokenized = tokenizer(
        caption,
        padding="max_length",
        truncation=True,
        max_length=max_text_len,
        return_tensors="np",
    )
    # flatten to 1-D for phrase extraction
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized.get("attention_mask")
    if hasattr(input_ids, "ndim") and input_ids.ndim == 2:
        input_ids = input_ids[0]
    if attention_mask is not None and hasattr(attention_mask, "ndim") and attention_mask.ndim == 2:
        attention_mask = attention_mask[0]
    tokenized_phrase = {"input_ids": input_ids.tolist()}
    if attention_mask is not None:
        tokenized_phrase["attention_mask"] = attention_mask.tolist()
    # ids to suppress (pad/cls/sep/eos/bos)
    forbidden_ids = set(
        i
        for i in [
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
        ]
        if i is not None
    )
    if debug_mode:
        log_text(
            f"tokenized for phrases: len={len(tokenized_phrase['input_ids'])} "
            f"logit_len={logits.shape[1] if logits.ndim==2 else logits.shape}",
            force=True,
        )

    box_scores_np = logits.max(axis=1)
    kept_boxes = []
    kept_scores = []
    kept_phrases = []

    ids_np = np.array(input_ids)
    attn_np = np.array(attention_mask, dtype=bool) if attention_mask is not None else None

    for box, score, logit in zip(boxes, box_scores_np, logits):
        logit = logit[: len(ids_np)]
        text_mask = logit > text_threshold
        if attn_np is not None:
            text_mask = text_mask & (attn_np)
        if forbidden_ids:
            text_mask = text_mask & (~np.isin(ids_np, list(forbidden_ids)))
        if not text_mask.any():
            continue
        phrase = get_phrases_from_posmap(
            jt.array(text_mask),
            tokenized_phrase,
            tokenizer,
        ).replace(".", "").strip()
        if phrase == "":
            continue
        kept_boxes.append(box)
        kept_scores.append(score)
        kept_phrases.append(phrase)

    if not kept_boxes:
        if debug_mode:
            log_text("predict: no phrases with valid tokens after masking", force=True)
        empty_boxes = jt.array(np.zeros((0, 4), dtype=np.float32))
        empty_scores = jt.array(np.zeros((0,), dtype=np.float32))
        return empty_boxes, empty_scores, []

    boxes_var = jt.array(np.ascontiguousarray(kept_boxes, dtype=np.float32))
    box_scores_var = jt.array(np.ascontiguousarray(kept_scores, dtype=np.float32))
    if debug_mode:
        log_tensor("predict.box_scores", np.asarray(kept_scores), force=True)
        log_text(f"predict: kept_phrases={len(kept_phrases)}", force=True)
    return boxes_var, box_scores_var, kept_phrases
def annotate(image_source: np.ndarray, boxes: Any, logits: Any, phrases: List[str]) -> np.ndarray:
    if sv is None or cv2 is None:
        raise ImportError("annotate() requires supervision and opencv-python installed.")

    h, w, _ = image_source.shape
    boxes_np = _to_numpy(boxes)
    logits_np = _to_numpy(logits)

    boxes_scaled = boxes_np * np.array([w, h, w, h], dtype=np.float32)
    xyxy = _box_cxcywh_to_xyxy(boxes_scaled)

    detections = sv.Detections(xyxy=xyxy)
    labels = [f"{phrase} {float(score):.2f}" for phrase, score in zip(phrases, logits_np)]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

    annotated = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated = bbox_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    return annotated


class Model:
    def __init__(self, model_config_path: str, model_checkpoint_path: str, device: str = "cuda", offline_mode: bool = False):
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device,
            offline_mode=offline_mode,
        )
        self.device = device

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> Any:
        if cv2 is None:
            raise ImportError("preprocess_image requires opencv-python installed.")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image_rgb)

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(pil, None)
        return image_transformed

    @staticmethod
    def post_process_result(source_h: int, source_w: int, boxes: Any, logits: Any):
        if sv is None:
            raise ImportError("post_process_result requires supervision installed.")
        boxes_np = _to_numpy(boxes)
        logits_np = _to_numpy(logits)

        boxes_scaled = boxes_np * np.array([source_w, source_h, source_w, source_h], dtype=np.float32)
        xyxy = _box_cxcywh_to_xyxy(boxes_scaled)

        return sv.Detections(
            xyxy=xyxy,
            confidence=logits_np.float(),
        )

    def predict_with_caption(
        self,
        image: np.ndarray,
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        processed = Model.preprocess_image(image_bgr=image)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        h, w, _ = image.shape
        detections = Model.post_process_result(h, w, boxes, logits)
        return detections, phrases

    def predict_with_classes(
        self,
        image: np.ndarray,
        classes: List[str],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        caption = ". ".join(classes) + "."
        processed = Model.preprocess_image(image_bgr=image)
        boxes, logits, _phrases = predict(
            model=self.model,
            image=processed,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        h, w, _ = image.shape
        detections = Model.post_process_result(h, w, boxes, logits)
        return detections