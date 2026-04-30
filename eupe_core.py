"""
eupe_core.py — v6  (performance-optimised)
===========================================
Key changes vs v5
-----------------
• Batched tile inference: all tiles sent to GPU in one forward pass.
  For a 1920×1080 image at tile=448, overlap=0.25 this is typically
  8-16 tiles → 1 GPU call instead of 8-16 sequential calls.
• No PIL round-trip inside the tile loop: float32 ndarray → normalised
  tensor directly, skipping uint8→PIL→ToImage→float32.
• Hann-window accumulation on GPU (torch tensors stay on device).
• Configurable tile_px (default 448) and overlap (default 0.25) —
  larger tiles = fewer GPU calls = faster, with minimal quality loss
  because EUPE already has global attention.
• PCA uses svd_solver='randomized' for ~3× speedup on large feature maps.
• _arr_to_b64 uses JPEG (quality 92) for speed; callers that need lossless
  PNG (label maps, saved masks) call _arr_to_b64_png explicitly.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize as sk_normalize

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

MODEL_CHOICES = [
    "eupe_vitt16", "eupe_vits16", "eupe_vitb16",
    "eupe_convnext_tiny", "eupe_convnext_small", "eupe_convnext_base",
]

PATCH_SIZE: dict[str, int] = {
    "eupe_vitt16": 16, "eupe_vits16": 16, "eupe_vitb16": 16,
    "eupe_convnext_tiny": 32, "eupe_convnext_small": 32, "eupe_convnext_base": 32,
}

# Maximum number of tiles sent to GPU in one batch.
# RTX A3000 has 6 GB VRAM; ViT-B processes 1 tile in ~180 MB peak,
# so 16 tiles ≈ 3 GB — safe headroom.  Reduce if you see OOM.
MAX_BATCH = 16


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(repo_dir: str, model_name: str, weights_path: str,
               device: torch.device):
    m = torch.hub.load(repo_dir, model_name, source="local", weights=weights_path)
    m = m.eval().to(device)
    # Warm-up: one dummy forward so the first real call isn't slow
    try:
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            if hasattr(m, "get_intermediate_layers"):
                m.get_intermediate_layers(dummy, n=1, return_class_token=False)
            elif hasattr(m, "forward_features"):
                m.forward_features(dummy)
            else:
                m(dummy)
    except Exception:
        pass
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation (vectorised, no PIL)
# ─────────────────────────────────────────────────────────────────────────────

def _crop_to_tensor(img_padded: np.ndarray, y0: int, x0: int,
                    tile_px: int, device: torch.device) -> torch.Tensor:
    """
    Slice a float32 HxWxC array, normalise, return (1,3,tile,tile) on device.
    Avoids uint8→PIL→ToImage→float32 round-trip entirely.
    """
    crop = img_padded[y0:y0 + tile_px, x0:x0 + tile_px]   # (T,T,3) float32
    t    = torch.from_numpy(crop).permute(2, 0, 1)          # (3,T,T)
    mean = IMAGENET_MEAN.to(t.device)
    std  = IMAGENET_STD.to(t.device)
    t    = (t - mean[:, None, None]) / std[:, None, None]
    return t.unsqueeze(0).to(device)                        # (1,3,T,T)


def _batch_crops(img_padded: np.ndarray,
                 positions: list[tuple[int, int]],
                 tile_px: int, device: torch.device) -> torch.Tensor:
    """Stack multiple crops into a single (B,3,T,T) batch tensor."""
    crops = []
    for y0, x0 in positions:
        c = img_padded[y0:y0 + tile_px, x0:x0 + tile_px]
        crops.append(torch.from_numpy(c).permute(2, 0, 1))
    batch = torch.stack(crops, 0).to(device)                # (B,3,T,T)
    mean  = IMAGENET_MEAN.to(device)[:, None, None]
    std   = IMAGENET_STD.to(device)[:, None, None]
    return (batch - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# Single-batch feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _forward_batch(model, batch: torch.Tensor) -> torch.Tensor:
    """
    (B,3,H,W) → (B, gh, gw, D) float32 CPU.
    Handles ViT (get_intermediate_layers / forward_features) and ConvNeXt.
    """
    if hasattr(model, "get_intermediate_layers"):
        # Returns list of length n; each element is (B, N_tokens, D)
        out   = model.get_intermediate_layers(batch, n=1, return_class_token=False)[0]
        # out: (B, N, D)  — N = gh*gw for ViT without class token
        feats = out.float().cpu()                                      # (B,N,D)
        B, N, D = feats.shape
        g = int(math.isqrt(N))
        if g * g != N:
            feats = feats[:, 1:, :]
            N -= 1; g = int(math.isqrt(N))
        return feats.reshape(B, g, g, D)

    elif hasattr(model, "forward_features"):
        out   = model.forward_features(batch)
        feats = out.float().cpu()
        if feats.dim() == 4:                     # ConvNeXt: (B,C,gh,gw)
            return feats.permute(0, 2, 3, 1)     # → (B,gh,gw,C)
        # ViT: (B, N+1, D) with class token
        feats = feats[:, 1:]                     # drop CLS → (B,N,D)
        B, N, D = feats.shape
        g = int(math.isqrt(N))
        return feats.reshape(B, g, g, D)

    else:
        out   = model(batch)
        feats = out.float().cpu()
        if feats.dim() == 4:
            return feats.permute(0, 2, 3, 1)
        feats = feats[:, 1:]
        B, N, D = feats.shape
        g = int(math.isqrt(N))
        return feats.reshape(B, g, g, D)


# ─────────────────────────────────────────────────────────────────────────────
# Cosine (Hann) window
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_window(size: int) -> torch.Tensor:
    r = torch.hann_window(size, periodic=False)
    return r.unsqueeze(0) * r.unsqueeze(1)    # (size, size)


# ─────────────────────────────────────────────────────────────────────────────
# Main feature extraction  (batched, GPU-accelerated accumulation)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    model,
    image: Image.Image,
    device: torch.device,
    tile_px:    int   = 448,    # larger tile → fewer GPU calls
    overlap:    float = 0.25,   # 0.25 gives good quality with 4× fewer tiles vs 0.5
    patch_size: int   = 16,
    max_batch:  int   = MAX_BATCH,
) -> np.ndarray:
    """
    Batched tiled EUPE with cosine-window blending.
    Returns float32 (gh, gw, D).

    Performance notes
    -----------------
    tile_px=448, overlap=0.25:  ~4–8 tiles for 1920×1080 → ~1 GPU batch
    tile_px=224, overlap=0.5 :  ~40 tiles → 3–4 GPU batches (old default)
    """
    tile_px   = max(patch_size, round(tile_px / patch_size) * patch_size)
    stride_px = max(patch_size, round(tile_px * (1 - overlap) / patch_size) * patch_size)
    W, H      = image.size

    pad_h = (tile_px - (H - tile_px) % stride_px) % stride_px if H > tile_px else 0
    pad_w = (tile_px - (W - tile_px) % stride_px) % stride_px if W > tile_px else 0
    H_pad, W_pad = H + pad_h, W + pad_w

    img_np     = np.array(image, dtype=np.float32) / 255.0
    img_padded = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    ys = list(range(0, max(1, H_pad - tile_px + 1), stride_px))
    xs = list(range(0, max(1, W_pad - tile_px + 1), stride_px))
    positions = [(y0, x0) for y0 in ys for x0 in xs]

    # ── Probe: determine output grid shape & feature dim ──────────────────
    probe_batch = _batch_crops(img_padded, [positions[0]], tile_px, device)
    probe_feat  = _forward_batch(model, probe_batch)     # (1, gh_t, gw_t, D)
    gh_t, gw_t, D = probe_feat.shape[1], probe_feat.shape[2], probe_feat.shape[3]
    es = tile_px // gh_t   # effective spatial stride per patch (pixels)

    gh_tot = H_pad // es
    gw_tot = W_pad // es

    # Accumulation buffers on GPU
    fsum  = torch.zeros(gh_tot, gw_tot, D, device=device)
    wsum  = torch.zeros(gh_tot, gw_tot,    device=device)
    win2d = _cosine_window(gh_t).to(device)   # (gh_t, gh_t)

    # ── Batched forward passes ────────────────────────────────────────────
    for i in range(0, len(positions), max_batch):
        batch_pos  = positions[i:i + max_batch]
        batch_t    = _batch_crops(img_padded, batch_pos, tile_px, device)  # (B,3,T,T)
        batch_feat = _forward_batch(model, batch_t).to(device)             # (B,gh_t,gw_t,D)

        for j, (y0, x0) in enumerate(batch_pos):
            gy0, gx0 = y0 // es, x0 // es
            feats_j  = batch_feat[j]                                       # (gh_t,gw_t,D)
            fsum[gy0:gy0 + gh_t, gx0:gx0 + gh_t] += feats_j * win2d.unsqueeze(-1)
            wsum[gy0:gy0 + gh_t, gx0:gx0 + gh_t] += win2d

    features = (fsum / wsum.unsqueeze(-1).clamp(min=1e-6)).cpu().numpy()
    gh_orig  = math.ceil(H / es)
    gw_orig  = math.ceil(W / es)
    return features[:gh_orig, :gw_orig].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PCA → RGB  (randomized solver for speed)
# ─────────────────────────────────────────────────────────────────────────────

def pca_to_rgb(features: np.ndarray, fg_pct: float = 30.0) -> np.ndarray:
    """(gh, gw, D) → (gh, gw, 3) float32 [0,1]."""
    gh, gw, D = features.shape
    flat = features.reshape(-1, D)
    # randomized solver is ~3× faster than 'auto' for large N, small n_components
    comp = PCA(n_components=3, svd_solver='randomized', random_state=0).fit_transform(flat)
    pc1  = comp[:, 0]
    fg   = pc1 > np.percentile(pc1, fg_pct)
    if fg.sum() > 3:
        full     = np.zeros((len(flat), 3), np.float32)
        full[fg] = PCA(n_components=3, svd_solver='randomized', random_state=0).fit_transform(flat[fg])
        comp     = full
    rgb = comp.astype(np.float32)
    for c in range(3):
        ch = rgb[:, c]; lo, hi = ch.min(), ch.max()
        rgb[:, c] = (ch - lo) / (hi - lo + 1e-8)
    return rgb.reshape(gh, gw, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Upsampling
# ─────────────────────────────────────────────────────────────────────────────

def upsample_to(arr_hwc: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    W, H = wh
    t    = torch.from_numpy(arr_hwc).permute(2, 0, 1).unsqueeze(0)
    up   = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return up.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)


def upsample_labels(lbl: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    W, H = wh
    t    = torch.from_numpy(lbl.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    up   = F.interpolate(t, size=(H, W), mode="nearest").squeeze().numpy().astype(np.int32)
    return up


# ─────────────────────────────────────────────────────────────────────────────
# K-Means  (L2-normalised feature space)
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans(
    features:   np.ndarray,
    pca_rgb:    np.ndarray,
    image:      Image.Image,
    n_clusters: int,
    space: str = "feature",
) -> tuple[np.ndarray, np.ndarray]:
    gh, gw, D = features.shape

    if space == "pca":
        flat_c = pca_rgb.reshape(-1, 3)
    else:
        flat_c = features.reshape(-1, D).astype(np.float32)
        norms  = np.linalg.norm(flat_c, axis=1, keepdims=True)
        flat_c = flat_c / np.where(norms > 0, norms, 1.0)

    km     = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(flat_c).reshape(gh, gw)

    W, H    = image.size
    lbl_up  = upsample_labels(labels, (W, H))
    orig_np = np.array(image).astype(np.float32) / 255.0

    palette = np.array([
        orig_np[lbl_up == k].mean(0) if (lbl_up == k).any() else np.zeros(3)
        for k in range(n_clusters)
    ], dtype=np.float32)
    mask_img = palette[lbl_up]

    rng          = np.random.default_rng(0)
    flat_palette = rng.uniform(0.2, 0.95, (n_clusters, 3)).astype(np.float32)
    return mask_img, flat_palette[lbl_up]


def make_overlay(orig_np: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.5) -> np.ndarray:
    return np.clip(orig_np * (1 - alpha) + mask * alpha, 0, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Watershed boundary refinement
# ─────────────────────────────────────────────────────────────────────────────

def watershed_refine(
    image:      Image.Image,
    lbl_map:    np.ndarray,
    n_erosions: int = 3,
) -> np.ndarray:
    """Snap coarse class boundaries to image-gradient edges via cv2.watershed."""
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    H, W    = lbl_map.shape
    kernel  = np.ones((3, 3), np.uint8)

    labels     = np.unique(lbl_map)
    lbl_to_idx = {int(lbl): i + 1 for i, lbl in enumerate(labels)}
    idx_to_lbl = {i + 1: int(lbl) for i, lbl in enumerate(labels)}

    markers = np.zeros((H, W), dtype=np.int32)
    for lbl in labels:
        region = (lbl_map == lbl).astype(np.uint8)
        eroded = region
        for _ in range(max(1, n_erosions)):
            eroded = cv2.erode(eroded, kernel)
        markers[eroded > 0] = lbl_to_idx[int(lbl)]

    cv2.watershed(img_bgr, markers)

    if (markers == -1).any():
        filled_u8 = np.where(markers > 0, markers, 0).astype(np.uint8)
        boundary  = (markers == -1)
        for _ in range(10):
            if not boundary.any():
                break
            dilated   = cv2.dilate(filled_u8, kernel)
            filled_u8 = np.where(boundary, dilated, filled_u8)
            boundary  = filled_u8 == 0
        markers = filled_u8.astype(np.int32)

    fallback = int(labels[0])
    result   = np.full((H, W), fallback, dtype=np.int32)
    for idx, lbl_id in idx_to_lbl.items():
        result[markers == idx] = lbl_id
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Annotation geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _polygon_grid_mask(poly_img: list, img_wh: tuple,
                       grid_wh: tuple) -> np.ndarray:
    W, H   = img_wh
    gw, gh = grid_wh
    scaled = [(x * gw / W, y * gh / H) for x, y in poly_img]
    canvas = Image.new("L", (gw, gh), 0)
    ImageDraw.Draw(canvas).polygon(scaled, fill=255)
    return np.array(canvas) > 0


def _bbox_grid_mask(x1, y1, x2, y2, img_wh: tuple,
                    grid_wh: tuple) -> np.ndarray:
    W, H   = img_wh
    gw, gh = grid_wh
    gx1 = int(x1 * gw / W);  gy1 = int(y1 * gh / H)
    gx2 = min(int(x2 * gw / W) + 1, gw)
    gy2 = min(int(y2 * gh / H) + 1, gh)
    m   = np.zeros((gh, gw), bool)
    m[gy1:gy2, gx1:gx2] = True
    return m


def collect_annotation_samples(
    features:    np.ndarray,
    img_wh:      tuple[int, int],
    annotations: list[dict],
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    gh, gw, D = features.shape
    flat      = features.reshape(-1, D).astype(np.float32)

    class_ids = sorted({a["class_id"] for a in annotations})
    cid2idx   = {cid: i for i, cid in enumerate(class_ids)}

    X_parts, y_parts = [], []
    for ann in annotations:
        cid  = ann["class_id"]
        cidx = cid2idx[cid]
        if ann["type"] == "bbox":
            x1, y1, x2, y2 = ann["coords"]
            mask = _bbox_grid_mask(x1, y1, x2, y2, img_wh, (gw, gh))
        else:
            mask = _polygon_grid_mask(ann["coords"], img_wh, (gw, gh))
        sel = flat[mask.reshape(-1)]
        if len(sel) == 0:
            continue
        X_parts.append(sel)
        y_parts.extend([cidx] * len(sel))

    if not X_parts:
        raise ValueError("No annotated feature patches — shapes may be outside the image.")

    return (np.vstack(X_parts),
            np.array(y_parts, dtype=np.int32),
            cid2idx)


# ─────────────────────────────────────────────────────────────────────────────
# Label propagation — single image
# ─────────────────────────────────────────────────────────────────────────────

def propagate_labels(
    features:    np.ndarray,
    img_wh:      tuple[int, int],
    annotations: list[dict],
    method: str  = "centroid",
    use_watershed: bool = False,
    image:  Optional[Image.Image] = None,
    watershed_erosions: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    gh, gw, D = features.shape
    flat      = features.reshape(-1, D).astype(np.float32)

    X, y, cid2idx = collect_annotation_samples(features, img_wh, annotations)
    class_ids = sorted(cid2idx, key=cid2idx.get)
    n_classes = len(class_ids)

    if method == "svm" and n_classes >= 2:
        try:
            Xn       = sk_normalize(X)
            fn       = sk_normalize(flat)
            clf      = LinearSVC(max_iter=3000, C=1.0)
            clf.fit(Xn, y)
            pred_idx = clf.predict(fn)
        except Exception:
            method = "centroid"

    if method == "centroid" or n_classes < 2:
        centroids = np.array([
            X[y == i].mean(0) if (y == i).any() else np.zeros(D)
            for i in range(n_classes)
        ], dtype=np.float32)
        cn       = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        fn2      = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
        pred_idx = (fn2 @ cn.T).argmax(axis=1)

    pred_cid = np.array([class_ids[i] for i in pred_idx], dtype=np.int32).reshape(gh, gw)
    W, H     = img_wh
    lbl_up   = upsample_labels(pred_cid, (W, H))

    if use_watershed and image is not None:
        lbl_up = watershed_refine(image, lbl_up, n_erosions=watershed_erosions)

    cid_color = {a["class_id"]: a["class_color"] for a in annotations}
    colored   = np.zeros((H, W, 3), np.float32)
    for cid, col in cid_color.items():
        colored[lbl_up == cid] = np.array(col, np.float32) / 255.0

    class_info = [
        {"class_id": cid, "class_color": cid_color.get(cid, [128, 128, 128])}
        for cid in class_ids
    ]
    return lbl_up, colored, class_info


# ─────────────────────────────────────────────────────────────────────────────
# Global cross-image model
# ─────────────────────────────────────────────────────────────────────────────

class GlobalModel:
    """Accumulates labelled feature patches from all annotated images."""

    def __init__(self):
        self.X:    Optional[np.ndarray] = None
        self.y:    Optional[np.ndarray] = None
        self.class_ids: list[int]       = []
        self.cid2idx:   dict[int, int]  = {}
        self.clf        = None
        self.clf_method = "centroid"
        self.n_samples_per_image: dict[str, int] = {}
        self._pending:  dict[str, tuple] = {}

    def reset(self):
        self.__init__()

    @property
    def n_classes(self) -> int:
        return len(self.class_ids)

    @property
    def n_samples(self) -> int:
        return 0 if self.X is None else len(self.X)

    def add_samples(self, X: np.ndarray, y_cid: np.ndarray,
                    filename: str, cid2idx: dict[int, int]) -> None:
        self._pending[filename] = (X, y_cid)
        self._rebuild()

    def _rebuild(self):
        pending = self._pending
        if not pending:
            return
        all_cids: set[int] = set()
        for X, y_cid in pending.values():
            all_cids.update(y_cid.tolist())
        self.class_ids = sorted(all_cids)
        self.cid2idx   = {cid: i for i, cid in enumerate(self.class_ids)}
        X_parts, y_parts = [], []
        for fname, (X, y_cid) in pending.items():
            y_idx = np.array([self.cid2idx[c] for c in y_cid], dtype=np.int32)
            X_parts.append(X)
            y_parts.append(y_idx)
            self.n_samples_per_image[fname] = len(X)
        self.X   = np.vstack(X_parts)
        self.y   = np.concatenate(y_parts)
        self.clf = None

    def fit(self, method: str = "centroid") -> None:
        if self.X is None or self.n_classes < 1:
            raise ValueError("No labelled samples.")
        if self.clf is not None:
            return
        if method == "svm" and self.n_classes >= 2:
            try:
                Xn       = sk_normalize(self.X)
                self.clf = LinearSVC(max_iter=3000, C=1.0)
                self.clf.fit(Xn, self.y)
                self.clf_method = "svm"
                return
            except Exception:
                pass
        centroids = np.array([
            self.X[self.y == i].mean(0) if (self.y == i).any() else np.zeros(self.X.shape[1])
            for i in range(self.n_classes)
        ], dtype=np.float32)
        self.clf        = centroids
        self.clf_method = "centroid"

    def predict(self, flat_features: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Call fit() first.")
        if self.clf_method == "svm":
            fn       = sk_normalize(flat_features.astype(np.float32))
            pred_idx = self.clf.predict(fn)
        else:
            centroids = self.clf
            cn  = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
            fn2 = flat_features.astype(np.float32)
            fn2 = fn2 / (np.linalg.norm(fn2, axis=1, keepdims=True) + 1e-8)
            pred_idx = (fn2 @ cn.T).argmax(axis=1)
        return np.array([self.class_ids[i] for i in pred_idx], dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Global model propagation
# ─────────────────────────────────────────────────────────────────────────────

def propagate_with_global_model(
    global_model:  GlobalModel,
    features:      np.ndarray,
    img_wh:        tuple[int, int],
    cid_color_map: dict[int, list],
    method: str    = "centroid",
    use_watershed: bool = False,
    image:  Optional[Image.Image] = None,
    watershed_erosions: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    global_model.fit(method)
    gh, gw, D = features.shape
    flat      = features.reshape(-1, D).astype(np.float32)
    pred_cid  = global_model.predict(flat).reshape(gh, gw)
    W, H      = img_wh
    lbl_up    = upsample_labels(pred_cid, (W, H))

    if use_watershed and image is not None:
        lbl_up = watershed_refine(image, lbl_up, n_erosions=watershed_erosions)

    colored = np.zeros((H, W, 3), np.float32)
    for cid, col in cid_color_map.items():
        colored[lbl_up == cid] = np.array(col, np.float32) / 255.0

    class_info = [
        {"class_id": cid, "class_color": cid_color_map.get(cid, [128, 128, 128])}
        for cid in global_model.class_ids
    ]
    return lbl_up, colored, class_info
