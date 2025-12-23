import os, sys

# =======================
# CPU-only hard switches
# (MUST be set before importing groundingdino.util.inference which imports jittor)
# =======================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1) Tell Jittor "do NOT use cuda" via flag-style env var
os.environ["use_cuda"] = "0"                      # Jittor flag name
# 2) Prevent CUDA toolchain auto-enable / auto-download
os.environ["nvcc_path"] = ""                      # empty string stops jittor_utils.install_cuda
# 3) Hide GPUs from CUDA runtime (use empty string; avoid -1 which can be quirky)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# 4) Avoid multiprocess compiler pool (sandbox blocks semaphores)
os.environ["DISABLE_MULTIPROCESSING"] = "1"
# 5) Point Jittor at the right pythonX.Y-config so it won't try to compile against system python
exe_real = os.path.realpath(sys.executable)
py_config = os.path.join(os.path.dirname(exe_real), f"python{sys.version_info.major}.{sys.version_info.minor}-config")
os.environ.setdefault("python_config_path", py_config)
# Enable verbose NaN/Inf diagnostics
os.environ.setdefault("GROUNDINGDINO_DEBUG_NAN", "1")

from groundingdino.util.inference import load_model, load_image, predict

# Extra safety after import (should already be CPU-only if the env worked)
import jittor as jt
jt.flags.use_cuda = 0
print("JT use_cuda =", jt.flags.use_cuda, flush=True)

model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth",
    device="cpu",
    offline_mode=True
)
print("MODEL LOADED OK", flush=True)

IMAGE_PATH = ".asset/cat_dog.jpeg"
TEXT_PROMPT = "chair . person . dog ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)
print("IMAGE LOADED OK", flush=True)
print("image type:", type(image), flush=True)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device="cpu",
    debug=True,
)

print("PREDICT OK",
      "boxes.shape=", getattr(boxes, "shape", None),
      "logits.shape=", getattr(logits, "shape", None),
      "num_phrases=", len(phrases),
      flush=True)

sys.exit(0)
