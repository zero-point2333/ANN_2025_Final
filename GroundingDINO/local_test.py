import os, sys, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# =======================
# Device switches
# (MUST be set before importing groundingdino.util.inference which imports jittor)
# =======================
USE_GPU = os.environ.get("JT_USE_CUDA", "1").lower() not in ("0", "false", "no")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if USE_GPU:
    # Avoid early CUDA init during import; let Jittor enable later via device="cuda".
    os.environ.pop("use_cuda", None)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    nvcc = os.environ.get("NVCC_PATH") or os.environ.get("nvcc_path") or shutil.which("nvcc")
    if nvcc:
        os.environ["nvcc_path"] = nvcc
    else:
        os.environ.pop("nvcc_path", None)
else:
    os.environ["use_cuda"] = "0"
    os.environ["nvcc_path"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Avoid multiprocess compiler pool (sandbox blocks semaphores)
os.environ["DISABLE_MULTIPROCESSING"] = "1"
# Point Jittor at the right pythonX.Y-config so it won't try to compile against system python
exe_real = os.path.realpath(sys.executable)
py_config = os.path.join(os.path.dirname(exe_real), f"python{sys.version_info.major}.{sys.version_info.minor}-config")
os.environ.setdefault("python_config_path", py_config)
# Enable verbose NaN/Inf diagnostics
os.environ.setdefault("GROUNDINGDINO_DEBUG_NAN", "1")

def _prefer_cuda_cache_path():
    if not USE_GPU:
        return
    try:
        import sysconfig
        import jittor_utils as jit_utils
        base_cache = jit_utils.find_cache_path()
        ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        # If a CPU core exists in the base cache, drop it from sys.path to avoid importing it.
        if any(name.startswith("jittor_core") and name.endswith(ext) for name in os.listdir(base_cache)):
            sys.path = [p for p in sys.path if p != base_cache]

        cu_dirs = []
        for name in os.listdir(base_cache):
            if name.startswith("cu") and os.path.isdir(os.path.join(base_cache, name)):
                cu_dirs.append(name)
        if not cu_dirs:
            return
        prefer = os.environ.get("cuda_arch")
        if prefer in cu_dirs:
            chosen = prefer
        else:
            def _ver_key(name: str):
                v = name[2:].split("_sm_")[0]
                parts = []
                for chunk in v.replace("_", ".").split("."):
                    if chunk.isdigit():
                        parts.append(int(chunk))
                return parts
            chosen = max(cu_dirs, key=_ver_key)
        cand = os.path.join(base_cache, chosen)
        core = os.path.join(cand, "jittor_core" + ext)
        if os.path.isfile(core) and cand not in sys.path:
            sys.path.insert(0, cand)
    except Exception:
        # best effort; fall back to default import behavior
        pass

_prefer_cuda_cache_path()

from groundingdino.util.inference import load_model, load_image, predict

# Extra safety after import
import jittor as jt
if not USE_GPU:
    jt.flags.use_cuda = 0
print("JT use_cuda =", jt.flags.use_cuda, flush=True)

def _resolve_data_path(rel_dir: str, filename: str, env_key: str) -> str:
    override = os.environ.get(env_key)
    if override:
        return override

    roots = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent / "GroundingDINO",
        Path.cwd(),
        Path.cwd() / "GroundingDINO",
    ]
    tried = []
    for root in roots:
        cand = root / rel_dir / filename
        tried.append(str(cand))
        if cand.is_file():
            return str(cand)

    raise FileNotFoundError(f"{filename} not found; tried: {', '.join(tried)}")

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)

def _box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=-1)

def _annotate_image(image_source, boxes, logits, phrases) -> Image.Image:
    img_np = np.asarray(image_source)
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    h, w = img_np.shape[:2]
    boxes_np = _to_numpy(boxes)
    logits_np = _to_numpy(logits)
    if boxes_np.size == 0:
        return Image.fromarray(img_np)

    boxes_scaled = boxes_np * np.array([w, h, w, h], dtype=np.float32)
    xyxy = _box_cxcywh_to_xyxy(boxes_scaled)
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img)
    colors = [
        (255, 75, 75),
        (75, 200, 255),
        (120, 255, 120),
        (255, 200, 80),
        (180, 140, 255),
    ]
    for idx, (box, score, phrase) in enumerate(zip(xyxy, logits_np, phrases)):
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = max(0.0, min(w - 1.0, x1))
        y1 = max(0.0, min(h - 1.0, y1))
        x2 = max(0.0, min(w - 1.0, x2))
        y2 = max(0.0, min(h - 1.0, y2))
        color = colors[idx % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{phrase} {float(score):.2f}"
        draw.text((x1 + 2, max(0.0, y1 - 12)), label, fill=color)
    return img

device = "cuda" if USE_GPU else "cpu"
checkpoint_path = _resolve_data_path(
    "weights",
    "groundingdino_swint_ogc.pth",
    "GROUNDINGDINO_CHECKPOINT",
)
config_path = _resolve_data_path(
    "groundingdino/config",
    "GroundingDINO_SwinT_OGC.py",
    "GROUNDINGDINO_CONFIG",
)
model = load_model(
    config_path,
    checkpoint_path,
    device=device,
    offline_mode=True
)
print("MODEL LOADED OK", flush=True)

IMAGE_PATH = _resolve_data_path(".asset", "cat_dog.jpeg", "GROUNDINGDINO_IMAGE")
TEXT_PROMPT = os.environ.get("GROUNDINGDINO_TEXT_PROMPT", "chair . person . dog .")
BOX_TRESHOLD = float(os.environ.get("GROUNDINGDINO_BOX_TH", "0.1"))
TEXT_TRESHOLD = float(os.environ.get("GROUNDINGDINO_TEXT_TH", "0.1"))
image_source, image = load_image(IMAGE_PATH)
print("IMAGE LOADED OK", flush=True)
print("image type:", type(image), flush=True)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=device,
    debug=True,
)

print("PREDICT OK",
      "boxes.shape=", getattr(boxes, "shape", None),
      "logits.shape=", getattr(logits, "shape", None),
      "num_phrases=", len(phrases),
      flush=True)

if os.environ.get("GROUNDINGDINO_SAVE_ANNOTATED", "1").lower() not in ("0", "false", "no"):
    try:
        annotated = _annotate_image(image_source, boxes, logits, phrases)
        out_path = os.environ.get(
            "GROUNDINGDINO_OUTPUT",
            str(Path(__file__).resolve().parent / "outputs" / "local_test_out.jpg"),
        )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(out_path)
        print(f"ANNOTATED IMAGE SAVED: {out_path}", flush=True)
    except Exception as exc:
        print(f"ANNOTATE FAILED: {exc}", flush=True)

sys.exit(0)