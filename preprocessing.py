"""
preprocessing.py
----------------
Image pre-processing filters applied BEFORE feature extraction.

Available methods
-----------------
  none       : identity (pass-through)
  clahe      : CLAHE in CIE-Lab L* channel
  ssr        : Single-Scale Retinex (log-domain Gaussian illumination estimate)
  msr        : Multi-Scale Retinex  (average of 3 SSR scales)
  msr_clahe  : MSR followed by CLAHE (often best for dark / hazy images)
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _float32(bgr: np.ndarray) -> np.ndarray:
    return bgr.astype(np.float32) / 255.0


def _uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# CLAHE
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(
    img: Image.Image,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> Image.Image:
    """
    CLAHE applied to the L* channel of CIE-Lab colour space.
    Enhances local contrast while preserving colour balance.
    """
    bgr = _pil_to_bgr(img)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    L_eq  = clahe.apply(L)
    lab_eq = cv2.merge([L_eq, a, b])
    bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)
    return _bgr_to_pil(bgr_eq)


# ─────────────────────────────────────────────────────────────────────────────
# Single-Scale Retinex (SSR)
# ─────────────────────────────────────────────────────────────────────────────

def _ssr_channel(channel: np.ndarray, sigma: float) -> np.ndarray:
    """Retinex on a single float32 channel [0,1]."""
    eps = 1e-6
    log_c = np.log(channel + eps)
    blurred = cv2.GaussianBlur(channel, (0, 0), sigma)
    log_l   = np.log(blurred + eps)
    retinex = log_c - log_l
    # stretch to [0, 1]
    mn, mx = retinex.min(), retinex.max()
    if mx > mn:
        retinex = (retinex - mn) / (mx - mn)
    return retinex.astype(np.float32)


def apply_ssr(
    img: Image.Image,
    sigma: float = 80.0,
) -> Image.Image:
    """Single-Scale Retinex."""
    bgr = _float32(_pil_to_bgr(img))
    out = np.stack([_ssr_channel(bgr[:, :, c], sigma) for c in range(3)], axis=2)
    return _bgr_to_pil(_uint8(out))


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Scale Retinex (MSR)
# ─────────────────────────────────────────────────────────────────────────────

def apply_msr(
    img: Image.Image,
    sigmas: tuple[float, ...] = (15.0, 80.0, 250.0),
    restore_color: bool = True,
) -> Image.Image:
    """
    Multi-Scale Retinex.
    With restore_color=True (MSRCR) a colour-restoration term is applied
    to preserve chromatic information.
    """
    bgr   = _float32(_pil_to_bgr(img))
    eps   = 1e-6
    retinex = np.zeros_like(bgr)

    for sigma in sigmas:
        for c in range(3):
            blurred        = cv2.GaussianBlur(bgr[:, :, c], (0, 0), sigma)
            retinex[:, :, c] += np.log(bgr[:, :, c] + eps) - np.log(blurred + eps)

    retinex /= len(sigmas)

    if restore_color:
        # colour-restoration factor
        intensity = bgr.sum(axis=2, keepdims=True) + eps
        cr = np.log(125.0 * bgr / intensity + 1.0)
        retinex *= cr

    # per-channel stretch
    for c in range(3):
        ch = retinex[:, :, c]
        mn, mx = ch.min(), ch.max()
        retinex[:, :, c] = (ch - mn) / (mx - mn + eps)

    return _bgr_to_pil(_uint8(retinex))


# ─────────────────────────────────────────────────────────────────────────────
# MSR + CLAHE
# ─────────────────────────────────────────────────────────────────────────────

def apply_msr_clahe(
    img: Image.Image,
    sigmas: tuple[float, ...] = (15.0, 80.0, 250.0),
    clip_limit: float = 2.0,
) -> Image.Image:
    return apply_clahe(apply_msr(img, sigmas=sigmas), clip_limit=clip_limit)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

METHODS: dict[str, str] = {
    "none":      "No preprocessing",
    "clahe":     "CLAHE (local contrast)",
    "ssr":       "Single-Scale Retinex",
    "msr":       "Multi-Scale Retinex (MSRCR)",
    "msr_clahe": "MSR + CLAHE",
}


def preprocess(img: Image.Image, method: str, **kwargs) -> Image.Image:
    """Apply the named preprocessing method to a PIL Image."""
    method = method.lower()
    if method in ("none", ""):
        return img
    if method == "clahe":
        return apply_clahe(img, **kwargs)
    if method == "ssr":
        return apply_ssr(img, **kwargs)
    if method == "msr":
        return apply_msr(img, **kwargs)
    if method == "msr_clahe":
        return apply_msr_clahe(img, **kwargs)
    raise ValueError(f"Unknown preprocessing method: {method!r}")
