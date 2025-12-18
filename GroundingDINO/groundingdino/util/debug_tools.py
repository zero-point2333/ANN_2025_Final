import os
import sys
from typing import Any

import numpy as np


def _as_numpy(x: Any) -> np.ndarray:
    """
    Convert jittor / torch / numpy / list into a contiguous numpy array for inspection.
    Avoid importing heavy deps here; check sys.modules for jittor to prevent side effects.
    """
    jt = sys.modules.get("jittor", None)
    if jt is not None and isinstance(x, getattr(jt, "Var", ())):
        arr = x.numpy()
    else:
        arr = np.asarray(x)
    if arr.dtype.kind not in ("f", "c"):
        arr = arr.astype(np.float32, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def debug_enabled(explicit: bool = False) -> bool:
    """
    Global debug switch controlled by env GROUNDINGDINO_DEBUG_NAN or an explicit flag.
    """
    if explicit:
        return True
    val = os.environ.get("GROUNDINGDINO_DEBUG_NAN", "")
    return str(val).lower() in ("1", "true", "yes", "y", "on")


def log_text(msg: str, force: bool = False, prefix: str = "[DEBUG]") -> None:
    if not debug_enabled(force):
        return
    print(f"{prefix} {msg}", flush=True)


def log_tensor(name: str, value: Any, force: bool = False, prefix: str = "[DEBUG]") -> None:
    """
    Print lightweight stats (shape, dtype, min/max, NaN/Inf counts) for any tensor-like object.
    """
    if not debug_enabled(force):
        return
    try:
        arr = _as_numpy(value)
        total = int(arr.size)
        num_nan = int(np.isnan(arr).sum()) if total > 0 else 0
        num_inf = int(np.isinf(arr).sum()) if total > 0 else 0
        finite_mask = np.isfinite(arr) if total > 0 else []
        if total == 0 or not np.any(finite_mask):
            finite_min = finite_max = finite_mean = None
        else:
            finite_vals = arr[finite_mask]
            finite_min = float(finite_vals.min())
            finite_max = float(finite_vals.max())
            finite_mean = float(finite_vals.mean())
        print(
            f"{prefix} {name}: shape={arr.shape} dtype={arr.dtype} "
            f"nan={num_nan} inf={num_inf} finite_min={finite_min} "
            f"finite_max={finite_max} finite_mean={finite_mean}",
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - best effort debug aid
        print(f"{prefix} {name}: <failed to log> {exc}", flush=True)
