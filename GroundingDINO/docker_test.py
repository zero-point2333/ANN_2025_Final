import os
from pathlib import Path

from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch
import cv2

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

checkpoint_path = _resolve_data_path("weights", "groundingdino_swint_ogc.pth", "GROUNDINGDINO_CHECKPOINT")
config_path = _resolve_data_path("groundingdino/config", "GroundingDINO_SwinT_OGC.py", "GROUNDINGDINO_CONFIG")
model = load_model(config_path, checkpoint_path)
model = model.to('cuda:0')
print(torch.cuda.is_available())
print('DONE!')
