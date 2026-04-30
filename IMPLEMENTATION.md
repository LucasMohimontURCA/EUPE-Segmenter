# EUPE Visualizer — Implementation Reference

This document explains *how* each component works internally, the mathematical
basis of each algorithm, and the design trade-offs made.  It is not a user
guide.

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [Feature extraction (`eupe_core.py`)](#2-feature-extraction)
3. [PCA → RGB visualisation](#3-pca--rgb-visualisation)
4. [Image preprocessing (`preprocessing.py`)](#4-image-preprocessing)
5. [K-means clustering](#5-k-means-clustering)
6. [Annotation geometry](#6-annotation-geometry)
7. [Label propagation — centroid classifier](#7-label-propagation--centroid-classifier)
8. [Label propagation — SVM classifier](#8-label-propagation--svm-classifier)
9. [Centroid vs SVM — when to use which](#9-centroid-vs-svm--when-to-use-which)
10. [Watershed boundary refinement](#10-watershed-boundary-refinement)
11. [Global cross-image model (`GlobalModel`)](#11-global-cross-image-model)
12. [Session persistence (`session.py`)](#12-session-persistence)
13. [Reblend without re-inference](#13-reblend-without-re-inference)
14. [Threading model](#14-threading-model)
15. [Coordinate systems](#15-coordinate-systems)
16. [Output files](#16-output-files)

---

## 1. Architecture overview

```
┌──────────────────────────────────────────────────────────┐
│  Browser (index.html)                                    │
│  ┌──────────┐  ┌──────────────────────────────────────┐  │
│  │ Visualize│  │ Annotate tab                         │  │
│  │ tab      │  │  canvas + class manager + prop panel │  │
│  └────┬─────┘  └──────────────┬───────────────────────┘  │
└───────┼─────────────────────── ┼────────────────────────┘
        │ REST/JSON              │
┌───────▼────────────────────────▼──────────────────────────┐
│  app.py  (Flask, thin routes)                             │
│  STATE dict  (model, image list, cache, session, gm)      │
├────────────┬──────────────────┬───────────────────────────┤
│ eupe_core  │ preprocessing.py │  session.py               │
│  ML logic  │  CLAHE/Retinex   │  JSON persistence         │
└────────────┴──────────────────┴───────────────────────────┘
```

The server is intentionally single-process (`threaded=True` means Flask spawns
threads per request but all share a single Python process and STATE dict).
The `threading.Lock` in STATE guards all writes to the cache and global model.

---

## 2. Feature extraction

### 2.1 EUPE backbone

EUPE (End-to-end Unified Patch Embeddings) is a Vision Transformer trained for
dense feature representation.  For each forward pass on a 224×224 input, a
ViT-B/16 produces **196 patch tokens** (14×14 grid), each a 768-dimensional
vector.  These vectors encode both local texture and global semantic context
because attention layers allow every patch to attend to every other patch.

### 2.2 Tiled inference with Hann-window blending

**Problem:** A raw image may be much larger than 224×224.  Resizing to 224
discards spatial detail; a single forward pass at full resolution is
memory-prohibitive.

**Solution:** Tile the image with overlapping windows, run the model on each,
and blend with a cosine (Hann) weight:

```
w[i,j] = hann(i) * hann(j)      (outer product of 1-D Hann windows)
```

For each tile at offset (y0, x0):

```python
gy0, gx0 = y0 // effective_stride, x0 // effective_stride
fsum[gy0:gy0+g, gx0:gx0+g] += feats * win2d.unsqueeze(-1)
wsum[gy0:gy0+g, gx0:gx0+g] += win2d
```

Final feature map: `features = fsum / wsum`.

The Hann window tapers to zero at tile edges, so patch vectors near a tile
boundary have lower weight than those near the tile centre.  Overlapping tiles
(default overlap=0.5) ensure every patch is covered by at least two tiles; the
weighted average suppresses seam artefacts.

**Effective stride (`es`):** `tile_px // patches_per_tile`.  For ViT-B/16 with
tile_px=224: `es = 224 // 14 = 16 px/patch`.  The feature grid therefore has
spatial resolution `image_size / 16` — identical to running ViT on the full
image if it fitted in memory.

### 2.3 Feature cache

Features are cached keyed by `(path_str, preproc_method)`.  Changing
preprocessing or model invalidates the cache; changing k or alpha does not.

---

## 3. PCA → RGB visualisation

### 3.1 Purpose

The 768-D feature space cannot be displayed directly.  PCA projects it to 3
principal components, which are mapped to R, G, B.  This is a *visualisation
only*; it is never used as input to the classifier or k-means by default.

### 3.2 Two-stage PCA with foreground masking

**Why two stages?**  The first principal component (PC1) of EUPE features
strongly correlates with background vs. foreground.  Background patches cluster
at one extreme of PC1.  Running PCA on all patches means PC1 "wastes" its
variance budget on the background/foreground split, leaving only PC2 and PC3 to
capture within-foreground variation — giving a dull, low-contrast visualisation
for the objects of interest.

**Stage 1:** Fit PCA on all patches, take PC1.  Threshold at the `fg_pct`
percentile (default 30%) to produce a foreground mask.

**Stage 2:** Re-fit PCA on *foreground patches only*.  Project all patches
through this foreground PCA (background patches remain zero).

This gives high-contrast, semantically meaningful colour variations within the
foreground region.

**Per-channel normalisation:**

```python
rgb[:, c] = (ch - ch.min()) / (ch.max() - ch.min())
```

Maps each component independently to [0, 1], preventing any single component
from dominating.

---

## 4. Image preprocessing

Applied *before* feature extraction.  The preprocessed image is fed to EUPE,
not the raw image (the raw image is still used for overlay colouring and
watershed).

### 4.1 CLAHE

**Contrast-Limited Adaptive Histogram Equalisation** (Zuiderveld 1994).

1. Convert RGB → CIE-Lab (perceptually uniform colour space).
2. Apply CLAHE to the L* (lightness) channel only.
3. Convert back to RGB.

Why L* only: equalising in RGB would shift hue.  L* is achromatic.

CLAHE divides the L* channel into a grid of tiles (default 8×8) and performs
histogram equalisation independently in each tile, then bilinearly interpolates
between tile transforms.  The *clip limit* caps the histogram at each tile to
prevent over-amplification of noise — a contrast enhancement is redistributed
to neighbouring bins rather than amplified indefinitely.

**When to use:** Low-contrast scenes, overcast lighting, medical images, remote
sensing with haze.

### 4.2 Single-Scale Retinex (SSR)

Based on Land's Retinex theory: perceived colour is independent of illumination.
The Retinex estimate of reflectance R is:

```
R(x,y) = log(I(x,y)) - log(L(x,y))
```

where L is the estimated illumination, approximated by a Gaussian blur of I:

```
L(x,y) = G_σ * I(x,y)
```

In log domain this is a difference of log-images:

```
R = log(I + ε) - log(G_σ * I + ε)
```

Each channel is processed independently.  σ (default 80px) controls the scale
of illumination variations removed.  Small σ removes fine-grained shading;
large σ removes global gradients.

Output is stretched to [0, 1] per channel.

**When to use:** Uneven lighting (shadows, spotlight), dark images.

### 4.3 Multi-Scale Retinex with Colour Restoration (MSRCR)

Average SSR over multiple σ values (default: 15, 80, 250 px):

```
R_MSR = (1/K) Σ_k  [log(I) - log(G_{σ_k} * I)]
```

The *colour restoration* factor:

```
CR(x,y) = log(125 · I(x,y) / (I_R + I_G + I_B + ε) + 1)
R_MSRCR = R_MSR * CR
```

This prevents colour desaturation that SSR can cause by multiplying the
Retinex output by a term that is large when the channel is a large fraction of
total intensity (i.e., when the colour is saturated).

**When to use:** Best general-purpose enhancement, especially for images with
both uneven illumination *and* important colour variation.  Computationally 3×
SSR.

### 4.4 MSR + CLAHE

MSR followed by CLAHE.  MSR handles large-scale illumination; CLAHE sharpens
local micro-contrast.  Best for extreme scenes (tunnels, night images, medical
scans with low local contrast).

---

## 5. K-means clustering

### 5.1 Feature space (default)

```python
flat_c = features.reshape(-1, D)           # (gh*gw, 768)
flat_c /= ||flat_c||_2                      # L2-normalise rows
labels = MiniBatchKMeans(k).fit_predict(flat_c)
```

**Why L2-normalise?**  In high-dimensional spaces (D=768), Euclidean distance
is dominated by vector magnitude.  L2-normalisation makes Euclidean distance
equivalent to cosine distance:

```
||u/||u|| - v/||v||||² = 2(1 - cosine_similarity(u,v))
```

EUPE features are directional (semantic content is encoded in the direction of
the vector, not its magnitude), so cosine distance is the natural metric.

**Why MiniBatchKMeans?**  Standard KMeans performs full matrix multiplications
each iteration — O(N·k·D) per step.  For a 1000×1000 image with D=768, N≈4000
patches and k=10 clusters, each KMeans step touches 30M floats.  MiniBatch
processes random subsets, reducing memory and wall-clock time by ~10×.

### 5.2 PCA-RGB space (optional comparison)

Clusters the 3-D PCA-RGB values.  Much faster (3D instead of 768D) but
discards 99%+ of the feature variance.  Useful for visual inspection but
inferior for semantic segmentation.

### 5.3 Recolouring

Each cluster k is assigned the **mean original-image RGB** of all pixels
belonging to it.  This preserves the photographic appearance while revealing
the cluster structure — a form of bilateral-filter-like smoothing where the
smoothing kernel is defined in feature space rather than spatial/colour space.

The flat-colour **label map** assigns a deterministic random colour per cluster
(seeded at 0) for unambiguous boundary inspection.

---

## 6. Annotation geometry

Annotations are stored in **image pixel coordinates** (origin top-left,
x right, y down) at the full original resolution.

When used for feature extraction, they must be converted to **feature grid
coordinates**:

```python
gx = int(ix * gw / W)     # scale from image px → grid px
gy = int(iy * gh / H)
```

**Bounding box** → `mask[gy1:gy2, gx1:gx2] = True`

**Polygon** → rasterised onto a (gh, gw) PIL image using `ImageDraw.polygon`,
which implements a standard scanline fill algorithm.

The number of feature patches inside a region depends on both the shape size
and the patch stride.  For ViT-B/16, a 224×224 region on a 1000×1000 image
yields approximately `(224/1000 * 56)² ≈ 157` patches.  Very small regions
(< ~3 patches on a side) may produce zero feature vectors — the code raises a
`ValueError` in this case.

---

## 7. Label propagation — centroid classifier

### 7.1 Centroid computation

For each class c, collect all feature vectors from annotated regions:

```
X_c = {f_i : patch i falls inside any annotation of class c}
centroid_c = mean(X_c)
```

### 7.2 Classification

Every patch in the image is assigned to the class whose centroid is closest
in **cosine similarity**:

```
pred(f) = argmax_c  cosine_sim(f, centroid_c)
        = argmax_c  (f / ||f||) · (centroid_c / ||centroid_c||)
```

This is implemented as a matrix multiply:

```python
fn  = flat / ||flat||        # (N, D) L2-normalised query
cn  = centroids / ||centroids||   # (k, D)
sims = fn @ cn.T             # (N, k)  all cosine similarities in one BLAS call
pred = sims.argmax(axis=1)
```

### 7.3 Complexity

O(N·k·D) for the matrix multiply.  For N=4000, k=5, D=768: 15M operations —
negligible.

### 7.4 Properties

- **Inductive:** any unseen patch is classifiable.
- **No hyperparameters** beyond the annotations themselves.
- **Sensitive to class imbalance in annotation area:** if one class has 10×
  more annotated patches, its centroid is computed from 10× more samples but
  this does not bias the classifier (the centroid is normalised before
  computing similarity).
- **Assumes convex class clusters in feature space.**  If a class is
  multi-modal (e.g., "vegetation" spanning grass and tree canopy), the centroid
  lies in a low-density region and performance degrades.  Use SVM or draw more
  diverse annotations in that case.

---

## 8. Label propagation — SVM classifier

### 8.1 LinearSVC formulation

A one-vs-rest (OvR) LinearSVC trains k binary linear classifiers.  For class c
vs. rest:

```
min_{w_c, b_c}  (1/2)||w_c||² + C Σ_i max(0, 1 - y_ci(w_c·x_i + b_c))
```

where y_ci ∈ {+1, -1}.  Default C=1.0.

### 8.2 Feature normalisation

All feature vectors are L2-normalised before fitting and prediction.  This is
critical for two reasons:

1. **Scale invariance:** without normalisation, patches with large-magnitude
   features dominate the margin calculation.
2. **Equivalence to cosine SVM:** normalised linear SVM computes the linear
   decision boundary in angular (cosine) space, which matches the geometry of
   EUPE features.

### 8.3 Prediction

```python
pred = argmax_c  (w_c · x_norm + b_c)
```

The predicted class is the one with the highest decision-function score.

### 8.4 Complexity

Training: O(N·k·D·iter) where iter ≈ 100–1000 (liblinear default).  For
N=4000, k=5, D=768, iter=100: ~1.5B operations.  This is the bottleneck and
explains why SVM is slower than centroid.

Prediction: O(N·k·D) — same as centroid.

### 8.5 max_iter=3000

LinearSVC uses coordinate descent (liblinear).  The default convergence
tolerance requires more iterations for high-dimensional, nearly-linearly-
separable problems.  3000 is sufficient for typical annotation sizes; a
`ConvergenceWarning` from sklearn indicates the solution is still valid but may
not have converged fully.

---

## 9. Centroid vs SVM — when to use which

| Property | Centroid | SVM |
|---|---|---|
| Speed | Instant (matrix multiply) | 1–5 s for typical annotation sizes |
| Boundary quality | Smooth, sometimes over-generalised | Sharp, margin-maximising |
| Multi-modal classes | Poor (single centroid) | Better (hyperplane can separate non-spherical clusters) |
| Convergence guarantee | Always converges (closed form) | May not converge in max_iter |
| Few annotations (< 20 per class) | Stable | May overfit or fail |
| Many annotations (> 200 per class) | Good | Better |
| Recommendation | First pass, fast iteration | Final refinement |

**Key insight:** the SVM maximises the *margin* between class clusters.  In
feature space, EUPE clusters tend to be well-separated for distinct semantic
categories but may overlap for visually similar categories (e.g., bare soil vs.
dry grass).  In such cases the SVM finds a hyperplane that separates the
annotated examples, while the centroid classifier blurs the boundary.

**When SVM falls back to centroid:** if there is only one class, OvR SVM is
undefined; if `fit()` raises any exception (e.g., `ConvergenceWarning` becomes
an error), the code silently falls back to centroid and continues.

---

## 10. Watershed boundary refinement

### 10.1 Motivation

The feature grid has stride 16px (ViT-B/16).  Labels are upsampled from this
coarse grid using nearest-neighbour interpolation, which creates blocky 16px
boundaries.  Watershed snaps these boundaries to gradient edges in the original
image at full pixel resolution.

### 10.2 Algorithm (cv2.watershed)

OpenCV's watershed requires integer *markers*: pixels labelled with a class
(> 0), uncertain (= 0), or boundary (= -1, output only).

**Step 1: Erosion to find sure-foreground**

For each class region, erode by a disk of radius `n_erosions` (default 3px).
The eroded region is "certainly class k" — far enough from the boundary that
it will not be mislabelled.  These become the markers.

```python
for lbl in unique_labels:
    region = (lbl_map == lbl).astype(uint8)
    eroded = erode(region, n_erosions_times)
    markers[eroded > 0] = lbl + 1    # +1 because 0 = unknown
```

Pixels not covered by any eroded region → marker = 0 (unknown).

**Step 2: Watershed fill**

`cv2.watershed(image_bgr, markers)` implements the Vincent-Soille algorithm
(1991).  It treats pixel intensities as a topographic surface and "floods"
from the marker seeds, expanding each label until flood boundaries meet.  Where
two labels meet, the pixel is marked -1 (watershed boundary).

The algorithm is exact: boundaries will always fall on local gradient maxima
(image edges).

**Step 3: Boundary assignment**

Watershed boundary pixels (marker = -1) are assigned by iterative dilation of
the non-boundary labels:

```python
for _ in range(5):
    dilated = dilate(non_boundary_labels)
    filled  = where(marker == -1, dilated, marker)
```

This grows each label one pixel at a time into the boundary band, equivalent to
nearest-label assignment.

### 10.3 Trade-offs

| n_erosions | Effect |
|---|---|
| 1 | Very thin sure-foreground band; most pixels are "unknown". Watershed dominates — can over-segment if image has many edges. |
| 3 (default) | Balanced. Sure-foreground covers ~80% of each region. |
| 8+ | Very large sure-foreground, almost no unknown pixels. Watershed barely changes the coarse label map. |

**Known limitation:** watershed treats every image edge as a potential
boundary.  If a class boundary coincides with a texture gradient within an
object (e.g., a stripe in a garment), the watershed may fragment it.  In such
cases, reduce `n_erosions` or disable watershed.

---

## 11. Global cross-image model

### 11.1 Motivation

Annotating every image individually is labour-intensive.  If features are
consistent across images (same scene type, same acquisition conditions), a few
representative annotations should generalise to the rest of the dataset.

### 11.2 `GlobalModel` data structure

```python
class GlobalModel:
    X : ndarray (N, D)    # all labelled feature vectors from all images
    y : ndarray (N,)      # class indices (contiguous 0..k-1)
    class_ids : list[int] # sorted class IDs
    cid2idx   : dict      # class_id → class_index
    clf       : object    # fitted classifier (centroid array or LinearSVC)
    n_samples_per_image: {filename: n_patches}
```

### 11.3 Update strategy

When annotations are submitted for image `fname`:

1. Extract feature vectors for all annotated regions → `(X_new, y_new)`.
2. `_pending[fname] = (X_new, y_new_cid)` — overwrites any previous entry for
   this filename (supports redo/correction of annotations).
3. `_rebuild()`: merge all `_pending` entries, recompute the global `(X, y)`,
   reset `clf = None`.

**No incremental learning:** the full pool is re-merged from scratch each time.
This is correct (not approximate) and fast for ≤ 100 images with ≤ 10k
patches/image.  For larger datasets, replace with `SGDClassifier.partial_fit`.

### 11.4 Snowball effect

| Images annotated | Effect |
|---|---|
| 1 | Equivalent to local propagation |
| 2–5 | Centroids stabilise; inter-class boundaries tighten |
| 10+ | Class distributions well-sampled; SVM finds a robust hyperplane |
| 20+ | Results often good enough to propagate unlabelled images without any new annotations |

Each propagation call (local *or* global scope) adds the current image's
features to the pool.  This means even a "local" propagation passively
contributes to the global model for future images.

### 11.5 Class colour consistency

The `cid_color_map` is rebuilt from `STATE["session"]["classes"]` on every
propagation call, so colours are always consistent with the class manager even
if classes are added between sessions.

---

## 12. Session persistence

### 12.1 File format

```
<image_dir>/.eupe_session/session.json
```

```json
{
  "version": 1,
  "classes": [
    {"id": 1, "name": "Road", "color": [255, 0, 0], "hex": "#ff0000"}
  ],
  "class_id_seq": 1,
  "shapes": {
    "image001.jpg": [
      {"type": "bbox", "class_id": 1, "coords": [120, 80, 340, 220]},
      {"type": "polygon", "class_id": 2, "coords": [[10,10],[50,10],[50,50]]}
    ]
  }
}
```

### 12.2 Atomic write

```python
tmp = path.with_suffix(".tmp")
json.dump(session, open(tmp, "w"))
tmp.replace(path)    # atomic rename on POSIX systems
```

`os.replace` is atomic on the same filesystem: the previous version is never
partially visible.

### 12.3 What is NOT persisted

- Feature cache (recomputed on next access — fast with preprocessing cache).
- GlobalModel (recomputed from session shapes on next propagation — deliberate:
  avoids stale feature data after model reload).
- K-means results (ephemeral, user-driven).

---

## 13. Reblend without re-inference

After propagation, the server caches:

```python
STATE["last_prop"][path_str] = {
    "lbl_up":  ndarray,   # (H, W) int32  pixel-level class IDs
    "colored": ndarray,   # (H, W, 3) float32  class colours
    "orig_np": ndarray,   # (H, W, 3) float32  original image
}
```

`/api/reblend` takes `alpha` and returns:

```python
overlay = orig_np * (1 - alpha) + colored * alpha
```

This is an O(H·W·3) = ~3M operations for a 1000×1000 image — returns in < 10 ms.
No model inference, no feature lookup.

The frontend debounces slider events (80 ms) to avoid flooding the server
during fast slider movement.

---

## 14. Threading model

Flask is run with `threaded=True`, meaning each HTTP request gets its own
thread.  All writes to `STATE` go through `STATE["lock"]` (a
`threading.Lock`).

**Reads** are generally not locked: Python's GIL prevents torn reads of
individual object references.  The feature cache may return a stale `None` on
the first access from a second thread, but this only causes a redundant
recompute — the result is idempotent.

**Writes** are locked:
- Cache insertion
- `last_prop` update
- `GlobalModel.add_samples` / `_rebuild`
- Session modification

The UI is single-user (one browser tab) in normal use, so lock contention is
negligible.

---

## 15. Coordinate systems

Three coordinate systems exist, each with a distinct conversion:

| System | Origin | Unit | Used in |
|---|---|---|---|
| Image pixels | Top-left | 1 px | Annotation storage, overlay, watershed |
| Feature grid | Top-left | 1 patch = 16px | Feature extraction, PCA, classifier |
| Canvas pixels | Top-left | variable (zoom-to-fit) | Browser drawing |

**Image → Feature grid:**
```python
gx, gy = int(ix * gw / W), int(iy * gh / H)
```

**Canvas → Image:**
```python
ix, iy = cx * imgNatW / canvas.width, cy * imgNatH / canvas.height
```

**Image → Canvas** (for redrawing saved annotations):
```python
cx, cy = ix * canvas.width / imgNatW, iy * canvas.height / imgNatH
```

Annotations are always serialised in image pixel coordinates, so they remain
valid across browser window resizes.

---

## 16. Output files

All outputs go to `<image_dir>/Original_dir_save/<stem>_<suffix>.png`.

| Suffix | Type | Content |
|---|---|---|
| `original` | PNG | Thumbnail of input image (max 1024px) |
| `pca_rgb` | PNG | PCA visualisation, bilinearly upsampled to original size |
| `mask_orig_color` | PNG | K-means clusters coloured by mean original-image pixel |
| `mask_labels` | PNG | K-means flat-colour label map |
| `overlay` | PNG | K-means mask blended with original |
| `seg_mask` | PNG | Propagation segmentation, class colours from session |
| `seg_overlay` | PNG | Propagation mask blended with original |
| `labelmap` | `.npy` | `(H, W) int32` array of class IDs (pixel-level) |

The `.npy` label map is the primary artifact for downstream training pipelines.
Load with `np.load("..._labelmap.npy")`.  Class IDs are the integer IDs from
the session (stable across sessions).  To convert to a one-hot mask:

```python
lbl = np.load("image_labelmap.npy")          # (H, W)
n_classes = lbl.max() + 1
one_hot = (lbl[..., None] == np.arange(n_classes)).astype(np.uint8)  # (H, W, C)
```
