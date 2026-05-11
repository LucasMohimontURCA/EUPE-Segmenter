"""
app.py — EUPE Visualizer v6
============================
Changes from v5
---------------
• Config loaded from config.json (repo_dir, weights_dir, defaults).
• Save directory renamed "Masks"; mask file = <stem>.png (no suffix),
  overlay = <stem>_overlay.png, labelmap = <stem>_labelmap.npy.
• /api/image returns existing mask/overlay b64 if they exist on disk.
• /api/watershed — apply watershed to the last cached propagation result
  without re-running the classifier (separate button in UI).
• CLI no longer requires --weights_dir (falls back to config.json).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import traceback
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
import torch

import eupe_core as core
import preprocessing as pp
import session as sess

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

_CFG_PATH = Path(__file__).parent / "config.json"


def _load_config() -> dict:
    defaults = {
        "repo_dir":        "EUPE",
        "weights_dir":     "EUPE/weights",
        "default_model":   "eupe_vitb16",
        "default_weights": "EUPE-ViT-B.pt",
        "host":            "127.0.0.1",
        "port":            5000,
    }
    if _CFG_PATH.exists():
        try:
            with open(_CFG_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            defaults.update(data)
        except Exception as e:
            print(f"[config] Could not read config.json: {e}  — using defaults")
    else:
        print(f"[config] config.json not found at {_CFG_PATH}, using defaults")
    return defaults


CFG = _load_config()

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# ─────────────────────────────────────────────────────────────────────────────
# Application state
# ─────────────────────────────────────────────────────────────────────────────

STATE: dict = {
    "repo_dir":    CFG["repo_dir"],
    "weights_dir": CFG["weights_dir"],
    "device":      torch.device("cpu"),
    "model_name":  None,
    "model":       None,
    "image_dir":   None,
    "image_list":  [],
    "session":     sess._empty_session(),
    # (path_str, preproc) -> {"features": ndarray, "size": (W,H)}
    "cache":       {},
    # path_str -> {"lbl_up", "colored", "orig_np"}  for reblend / watershed
    "last_prop":   {},
    "global_model": core.GlobalModel(),
    "lock":         threading.Lock(),
}

# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _masks_dir(image_path: Path) -> Path:
    """<image_dir>/Masks/"""
    return image_path.parent / "Masks"


def _mask_path(image_path: Path) -> Path:
    """Mask PNG: Masks/<stem>.png  (exact same stem as original)."""
    return _masks_dir(image_path) / image_path.with_suffix(".png").name


def _overlay_path(image_path: Path) -> Path:
    """Overlay PNG: Masks/<stem>_overlay.png"""
    return _masks_dir(image_path) / (image_path.stem + "_overlay.png")


def _labelmap_path(image_path: Path) -> Path:
    return _masks_dir(image_path) / (image_path.stem + "_labelmap.npy")

# ─────────────────────────────────────────────────────────────────────────────
# General helpers
# ─────────────────────────────────────────────────────────────────────────────

def _list_weight_files() -> list[str]:
    wd = Path(STATE["weights_dir"])
    return (sorted(p.name for p in wd.iterdir()
                   if p.suffix in {".pt", ".pth", ".bin", ".ckpt"})
            if wd.is_dir() else [])


def _collect_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _get_features(image_path: Path, preproc: str) -> dict:
    key = (str(image_path), preproc)
    with STATE["lock"]:
        entry = STATE["cache"].get(key)
    if entry is not None:
        return entry
    image = pp.preprocess(Image.open(image_path).convert("RGB"), preproc)
    ps    = core.PATCH_SIZE.get(STATE["model_name"], 16)
    feats = core.extract_features(
        STATE["model"], image, STATE["device"],
        tile_px   = CFG.get("tile_px",   448),
        overlap   = CFG.get("overlap",   0.25),
        patch_size= ps,
        max_batch = CFG.get("max_batch", core.MAX_BATCH),
    )
    entry = {"features": feats, "size": image.size}
    with STATE["lock"]:
        STATE["cache"][key] = entry
    return entry


def _evict_cache(max_entries: int = 20) -> None:
    """
    Keep STATE["cache"] under max_entries to bound RAM usage.
    Removes oldest-inserted entries (dict preserves insertion order, Python 3.7+).
    Default: 20 entries × ~25 MB ≈ 500 MB headroom on 32 GB RAM.
    Override via config.json "cache_max_entries".
    """
    with STATE["lock"]:
        while len(STATE["cache"]) > max_entries:
            del STATE["cache"][next(iter(STATE["cache"]))]


def _extract_features_nocache(image_path: Path, preproc: str) -> np.ndarray:
    """
    Extract features WITHOUT caching — used by batch so 100+ images
    don't accumulate in RAM. Returns the raw (gh, gw, D) array only;
    caller is responsible for discarding it after use.
    """
    image = pp.preprocess(Image.open(image_path).convert("RGB"), preproc)
    ps    = core.PATCH_SIZE.get(STATE["model_name"], 16)
    return core.extract_features(
        STATE["model"], image, STATE["device"],
        tile_px   = CFG.get("tile_px",   448),
        overlap   = CFG.get("overlap",   0.5),
        patch_size= ps,
        max_batch = CFG.get("max_batch", core.MAX_BATCH),
    ), image.size


def _prefetch(image_path: Path, preproc: str) -> None:
    """Fire-and-forget background feature extraction for the next image."""
    try:
        _get_features(image_path, preproc)
        _evict_cache(CFG.get("cache_max_entries", 20))
    except Exception:
        pass


def _arr_to_b64(arr: np.ndarray, quality: int = 92) -> str:
    """Fast JPEG encoding for display images."""
    img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False)
    return base64.b64encode(buf.getvalue()).decode()


def _arr_to_b64_png(arr: np.ndarray) -> str:
    """Lossless PNG — pixel-exact values."""
    img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode()


def _png_to_b64(path: Path) -> str | None:
    """Read an existing image from disk and return as fast JPEG b64, or None."""
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92, optimize=False)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _auto_save_mask(path: Path, lbl_up: np.ndarray,
                    colored: np.ndarray, overlay: np.ndarray) -> list[str]:
    """Write mask, overlay, and labelmap — replacing any existing files."""
    out = _masks_dir(path)
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    def _w(arr: np.ndarray, dst: Path):
        Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).save(dst)
        saved.append(str(dst))

    _w(colored, _mask_path(path))
    _w(overlay,  _overlay_path(path))
    np.save(str(_labelmap_path(path)), lbl_up)
    saved.append(str(_labelmap_path(path)))
    return saved

# ─────────────────────────────────────────────────────────────────────────────
# Global model rebuild helper
# ─────────────────────────────────────────────────────────────────────────────

def _trigger_background_rebuild() -> None:
    """
    Manually-triggered rebuild of the GlobalModel from session shapes.
    Called ONLY from:
      - /api/rebuild_global  (explicit UI button)
      - /api/import_session  (after merging foreign shapes)
    NOT called automatically on startup or navigation to avoid loading
    every annotated image into memory silently.
    """
    session = STATE["session"]
    has_shapes = any(v for v in session["shapes"].values())
    if not STATE["model"] or not has_shapes:
        return
    preproc = CFG.get("default_preproc", "none")
    def _run():
        try:
            STATE["global_model"].reset()
            n = _rebuild_global_model(preproc)
            gm = STATE["global_model"]
            print(f"[global_model] Auto-rebuilt: {n} images, "
                  f"{gm.n_samples} samples, {gm.n_classes} classes")
        except Exception:
            import traceback as _tb
            _tb.print_exc()
    threading.Thread(target=_run, daemon=True).start()


def _rebuild_global_model(preproc: str = "none") -> int:
    """
    Walk every annotated image in the session, extract features (from cache or
    freshly), and feed samples into the GlobalModel.  Returns images processed.

    Called automatically when global propagation is requested but the model is
    empty (covers server-restart with an existing session).
    """
    session    = STATE["session"]
    img_lookup = {p.name: p for p in STATE["image_list"]}
    gm         = STATE["global_model"]
    processed  = 0

    for fname, shapes in session["shapes"].items():
        if not shapes:
            continue
        path = img_lookup.get(fname)
        if path is None:
            continue
        cid_color = {c["id"]: c.get("color", [128, 128, 128]) for c in session["classes"]}
        anns = [
            {"class_id": s["class_id"],
             "class_color": cid_color.get(s["class_id"], [128, 128, 128]),
             "type": s["type"], "coords": s["coords"]}
            for s in shapes
        ]
        try:
            feats, img_size = _extract_features_nocache(path, preproc)
            X, y_idx, cid2idx = core.collect_annotation_samples(
                feats, img_size, anns)
            del feats   # free immediately — never accumulate in RAM
            idx_to_cid = {v: k for k, v in cid2idx.items()}
            y_cid = np.array([idx_to_cid[i] for i in y_idx], dtype=np.int32)
            gm.add_samples(X, y_cid, fname, cid2idx)
            processed += 1
        except Exception as ex:
            import traceback as _tb
            print(f"[rebuild_global] Skipping {fname}: {ex}")
            _tb.print_exc()

    return processed


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")


# ── Config endpoint ───────────────────────────────────────────────────────────

@app.route("/api/config")
def api_config():
    gm = STATE["global_model"]
    return jsonify({
        "models":           core.MODEL_CHOICES,
        "weight_files":     _list_weight_files(),
        "preproc_methods":  pp.METHODS,
        "image_dir":        str(STATE["image_dir"]) if STATE["image_dir"] else "",
        "model_name":       STATE["model_name"] or "",
        "n_images":         len(STATE["image_list"]),
        # Defaults from config.json sent to the UI so selects pre-populate
        "default_model":    CFG.get("default_model", ""),
        "default_weights":  CFG.get("default_weights", ""),
        "global_model": {
            "n_samples":          gm.n_samples,
            "n_classes":          gm.n_classes,
            "n_images_labelled":  len(gm.n_samples_per_image),
        },
    })


# ── Directory / model ─────────────────────────────────────────────────────────

@app.route("/api/set_image_dir", methods=["POST"])
def api_set_image_dir():
    raw = (request.json.get("dir") or "").strip()
    d   = Path(raw)
    if not d.is_dir():
        # Try relative to cwd (covers short names like "34" typed from the app dir)
        candidate = Path.cwd() / raw
        if candidate.is_dir():
            d = candidate
    if not d.is_dir():
        return jsonify({"error": f"Directory not found: '{raw}'"}), 400
    d = d.resolve()
    imgs = _collect_images(d)
    if not imgs:
        return jsonify({"error": "No images found"}), 400
    loaded = sess.load(d)
    with STATE["lock"]:
        STATE["image_dir"]  = d
        STATE["image_list"] = imgs
        STATE["cache"]      = {}
        STATE["last_prop"]  = {}
        STATE["session"]    = loaded
        STATE["global_model"].reset()
    return jsonify({"n_images": len(imgs), "dir": str(d),
                    "session": sess.export_summary(loaded)})


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    mn = request.json.get("model")
    wf = request.json.get("weight_file")
    if mn not in core.MODEL_CHOICES:
        return jsonify({"error": "Unknown model"}), 400
    wp = Path(STATE["weights_dir"]) / wf
    if not wp.exists():
        return jsonify({"error": f"Not found: {wp}"}), 400
    try:
        m = core.load_model(STATE["repo_dir"], mn, str(wp), STATE["device"])
        with STATE["lock"]:
            STATE["model_name"] = mn
            STATE["model"]      = m
            STATE["cache"]      = {}
            STATE["global_model"].reset()
        gm = STATE["global_model"]
        return jsonify({
            "ok":    True,
            "model": mn,
            "global_model": {
                "n_samples":         gm.n_samples,
                "n_classes":         gm.n_classes,
                "n_images_labelled": len(gm.n_samples_per_image),
            },
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Image browsing ────────────────────────────────────────────────────────────

@app.route("/api/image")
def api_image():
    imgs = STATE["image_list"]
    if not imgs:
        return jsonify({"error": "No images loaded"}), 404
    idx   = int(request.args.get("idx", 0)) % len(imgs)
    path  = imgs[idx]
    image = Image.open(path).convert("RGB")
    thumb = image.copy()
    thumb.thumbnail((1024, 1024), Image.LANCZOS)
    shapes = sess.get_shapes(STATE["session"], path.name)

    # Load existing mask / overlay from disk if they exist
    existing_mask    = _png_to_b64(_mask_path(path))
    existing_overlay = _png_to_b64(_overlay_path(path))

    # If mask exists on disk but not in last_prop cache, reload colored array
    # so that /api/reblend and /api/watershed work immediately
    if existing_mask and str(path) not in STATE["last_prop"]:
        try:
            mask_img = np.array(Image.open(_mask_path(path))).astype(np.float32) / 255.0
            orig_np  = np.array(image).astype(np.float32) / 255.0
            lbl_up   = (np.load(str(_labelmap_path(path)))
                        if _labelmap_path(path).exists() else None)
            with STATE["lock"]:
                STATE["last_prop"][str(path)] = {
                    "lbl_up":  lbl_up,
                    "colored": mask_img,
                    "orig_np": orig_np,
                }
        except Exception:
            pass   # non-fatal

    return jsonify({
        "idx":            idx,
        "total":          len(imgs),
        "name":           path.name,
        "width":          image.width,
        "height":         image.height,
        "thumb":          _arr_to_b64(np.array(thumb).astype(np.float32) / 255.0),
        "shapes":         shapes,
        "session":        sess.export_summary(STATE["session"]),
        "existing_mask":    existing_mask,
        "existing_overlay": existing_overlay,
    })


# ── Preprocessing preview ─────────────────────────────────────────────────────

@app.route("/api/preprocess_preview")
def api_preprocess_preview():
    """Return the preprocessed image as b64 PNG — no feature extraction."""
    imgs = STATE["image_list"]
    if not imgs:
        return jsonify({"error": "No images loaded"}), 404
    idx     = int(request.args.get("idx", 0)) % len(imgs)
    preproc = request.args.get("preproc", "none")
    path    = imgs[idx]
    try:
        image = Image.open(path).convert("RGB")
        processed = pp.preprocess(image, preproc)
        # Thumbnail to keep response size small
        processed.thumbnail((1024, 1024), Image.LANCZOS)
        return jsonify({
            "preview": _arr_to_b64(np.array(processed).astype(np.float32) / 255.0)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Visualisation ─────────────────────────────────────────────────────────────

@app.route("/api/pca")
def api_pca():
    imgs = STATE["image_list"]
    if not imgs:   return jsonify({"error": "No images loaded"}), 404
    if not STATE["model"]: return jsonify({"error": "No model loaded"}), 400
    idx  = int(request.args.get("idx", 0)) % len(imgs)
    path = imgs[idx]
    try:
        e   = _get_features(path, request.args.get("preproc", "none"))
        rgb = core.pca_to_rgb(e["features"], float(request.args.get("fg_pct", 30.0)))
        img = Image.open(path).convert("RGB")
        return jsonify({"pca_rgb": _arr_to_b64(core.upsample_to(rgb, img.size))})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/kmeans")
def api_kmeans():
    imgs = STATE["image_list"]
    if not imgs:   return jsonify({"error": "No images loaded"}), 404
    if not STATE["model"]: return jsonify({"error": "No model loaded"}), 400
    idx     = int(request.args.get("idx", 0)) % len(imgs)
    k       = max(2, min(32, int(request.args.get("k", 5))))
    fg_pct  = float(request.args.get("fg_pct", 30.0))
    space   = request.args.get("space", "feature")
    alpha   = float(request.args.get("alpha", 0.5))
    preproc = request.args.get("preproc", "none")
    path    = imgs[idx]
    try:
        e       = _get_features(path, preproc)
        feats   = e["features"]
        pca_rgb = core.pca_to_rgb(feats, fg_pct)
        image   = Image.open(path).convert("RGB")
        orig_np = np.array(image).astype(np.float32) / 255.0
        mask_img, label_img = core.run_kmeans(feats, pca_rgb, image, k, space)
        return jsonify({
            "pca_rgb":   _arr_to_b64(core.upsample_to(pca_rgb, image.size)),
            "mask_img":  _arr_to_b64(mask_img),
            "label_img": _arr_to_b64(label_img),
            "overlay":   _arr_to_b64(core.make_overlay(orig_np, mask_img, alpha)),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Annotation session ────────────────────────────────────────────────────────

@app.route("/api/session")
def api_session_get():
    return jsonify(sess.export_summary(STATE["session"]))


@app.route("/api/session/class", methods=["POST"])
def api_upsert_class():
    idir = STATE["image_dir"]
    if idir is None: return jsonify({"error": "No directory set"}), 400
    cls = sess.upsert_class(STATE["session"], request.json)
    sess.save(idir, STATE["session"])
    return jsonify({"class": cls, "session": sess.export_summary(STATE["session"])})


@app.route("/api/session/class/<int:class_id>", methods=["DELETE"])
def api_delete_class(class_id: int):
    idir = STATE["image_dir"]
    if idir is None: return jsonify({"error": "No directory set"}), 400
    sess.delete_class(STATE["session"], class_id)
    sess.save(idir, STATE["session"])
    STATE["global_model"].reset()
    return jsonify({"session": sess.export_summary(STATE["session"])})


@app.route("/api/session/shapes", methods=["POST"])
def api_save_shapes():
    body = request.json or {}
    idir = STATE["image_dir"]
    if idir is None: return jsonify({"error": "No directory set"}), 400
    sess.set_shapes(STATE["session"], body.get("filename", ""), body.get("shapes", []))
    sess.save(idir, STATE["session"])
    return jsonify({"ok": True})


# ── Session import from external directory ────────────────────────────────────

@app.route("/api/import_session", methods=["POST"])
def api_import_session():
    """
    Merge a session from another directory into the current session.
    Works for both same-dataset (shapes added to canvas) and cross-dataset
    (features extracted from source images and fed to global model).
    Returns immediately; feature extraction runs in a background thread.
    """
    idir = STATE["image_dir"]
    if idir is None:
        return jsonify({"error": "Set an image directory first."}), 400
    if not STATE["model"]:
        return jsonify({"error": "Load a model before importing a session."}), 400

    body    = request.json or {}
    src_str = body.get("source_dir", "").strip()
    if not src_str:
        return jsonify({"error": "No source directory provided."}), 400

    # Resolve path: absolute → relative to image dir → relative to cwd
    src = Path(src_str)
    if not src.is_dir():
        c = Path(idir) / src_str
        if c.is_dir(): src = c
    if not src.is_dir():
        c = Path.cwd() / src_str
        if c.is_dir(): src = c
    if not src.is_dir():
        return jsonify({"error": f"Directory not found: '{src_str}'"}), 400
    src = src.resolve()

    # Load foreign session
    foreign = sess.load(src)
    print(f"[import] Loaded session from {src}")
    print(f"[import] Foreign classes: {[c['name'] for c in foreign['classes']]}")
    print(f"[import] Foreign shape keys: {list(foreign['shapes'].keys())[:5]}")
    print(f"[import] Foreign shape counts: { {k: len(v) for k,v in foreign['shapes'].items()} }")

    if not foreign["classes"] and not any(foreign["shapes"].values()):
        return jsonify({"error": f"Source session is empty: {src / '.eupe_session' / 'session.json'}"}), 400

    current = STATE["session"]

    # ── Merge classes by name ─────────────────────────────────────────────────
    cur_name_to_id = {c["name"]: c["id"] for c in current["classes"]}
    id_remap: dict[int, int] = {}
    for fc in foreign["classes"]:
        if fc["name"] in cur_name_to_id:
            id_remap[fc["id"]] = cur_name_to_id[fc["name"]]
        else:
            current["class_id_seq"] += 1
            new_id = current["class_id_seq"]
            id_remap[fc["id"]] = new_id
            cur_name_to_id[fc["name"]] = new_id
            current["classes"].append({
                "id":    new_id,
                "name":  fc["name"],
                "color": fc.get("color", [128, 128, 128]),
                "hex":   fc.get("hex",   "#808080"),
            })
    print(f"[import] id_remap: {id_remap}")

    # ── Build image lookups (case-insensitive for Windows compatibility) ───────
    # Current directory: name.lower() → Path
    cur_img_lower = {p.name.lower(): p for p in STATE["image_list"]}
    # Source directory: name.lower() → Path
    src_img_lower: dict[str, Path] = {}
    for p in src.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            src_img_lower[p.name.lower()] = p

    print(f"[import] Source images found: {len(src_img_lower)}")
    print(f"[import] Source images (first 5): {list(src_img_lower.keys())[:5]}")

    # ── Remap shapes and figure out which images are in which dataset ─────────
    shapes_in_current  = 0   # shapes added to current session JSON (same filename)
    shapes_in_source   = 0   # shapes to be extracted from source images
    source_img_anns: dict[str, tuple[Path, list]] = {}   # fname_lower → (src_path, anns)

    cid_color_map = {c["id"]: c.get("color", [128, 128, 128]) for c in current["classes"]}

    for fname, raw_shapes in foreign["shapes"].items():
        if not raw_shapes:
            continue
        fname_lower = fname.lower()
        remapped = [
            {**s, "class_id": id_remap.get(s["class_id"], s["class_id"])}
            for s in raw_shapes
        ]
        anns = [
            {"class_id":    r["class_id"],
             "class_color": cid_color_map.get(r["class_id"], [128, 128, 128]),
             "type":        r["type"],
             "coords":      r["coords"]}
            for r in remapped
        ]

        cur_path = cur_img_lower.get(fname_lower)
        if cur_path is not None:
            # Same filename exists in current dir → add to current session JSON
            canonical = cur_path.name   # use current dir's actual capitalisation
            existing     = current["shapes"].get(canonical, [])
            existing_set = {str(e) for e in existing}
            added = [r for r in remapped if str(r) not in existing_set]
            current["shapes"][canonical] = existing + added
            shapes_in_current += len(added)
            print(f"[import] Added {len(added)} shapes for {canonical} to session JSON")

        src_path = src_img_lower.get(fname_lower)
        if src_path is not None:
            # Image exists in source dir → extract features and feed global model
            source_img_anns[fname_lower] = (src_path, anns)
            shapes_in_source += len(anns)

    sess.save(idir, current)
    print(f"[import] Session saved. shapes_in_current={shapes_in_current}, "
          f"shapes_in_source={shapes_in_source}, "
          f"source_imgs_to_process={len(source_img_anns)}")

    # ── Background feature extraction and global model feeding ────────────────
    _snap_source_img_anns = dict(source_img_anns)
    _snap_preproc = body.get("preproc", CFG.get("default_preproc", "none"))
    _snap_ps      = core.PATCH_SIZE.get(STATE["model_name"], 16)

    def _feed_global():
        fed, total_samples = 0, 0
        for fname_lower, (src_path, anns) in _snap_source_img_anns.items():
            try:
                cache_key = (str(src_path), _snap_preproc)
                with STATE["lock"]:
                    cached = STATE["cache"].get(cache_key)
                if cached is None:
                    img_pil = pp.preprocess(
                        Image.open(src_path).convert("RGB"), _snap_preproc)
                    feats = core.extract_features(
                        STATE["model"], img_pil, STATE["device"],
                        tile_px   = CFG.get("tile_px",   448),
                        overlap   = CFG.get("overlap",   0.25),
                        patch_size= _snap_ps,
                        max_batch = CFG.get("max_batch", core.MAX_BATCH),
                    )
                    cached = {"features": feats, "size": img_pil.size}
                    with STATE["lock"]:
                        STATE["cache"][cache_key] = cached

                img_pil2 = Image.open(src_path).convert("RGB")
                X, y_idx, cid2idx = core.collect_annotation_samples(
                    cached["features"], img_pil2.size, anns)
                idx_to_cid = {v: k for k, v in cid2idx.items()}
                y_cid = np.array([idx_to_cid[i] for i in y_idx], dtype=np.int32)
                STATE["global_model"].add_samples(X, y_cid, fname_lower, cid2idx)
                fed += 1
                total_samples += len(X)
                print(f"[import] Fed {src_path.name}: {len(X)} samples "
                      f"(global total: {STATE['global_model'].n_samples})")
            except Exception:
                import traceback as _tb
                print(f"[import] Error feeding {src_path.name}:")
                _tb.print_exc()
        print(f"[import] Background done: {fed} images, {total_samples} samples")

    if _snap_source_img_anns:
        threading.Thread(target=_feed_global, daemon=True).start()

    gm = STATE["global_model"]
    return jsonify({
        "ok":                     True,
        "imported_files":         shapes_in_current // max(1, 1),  # shapes in JSON
        "imported_shapes":        shapes_in_current,
        "foreign_images_queued":  len(_snap_source_img_anns),
        "resolved_path":          str(src),
        "session":                sess.export_summary(current),
        "global_model": {
            "n_samples":         gm.n_samples,
            "n_classes":         gm.n_classes,
            "n_images_labelled": len(gm.n_samples_per_image),
        },
    })


# ── Global model status (for frontend polling) ────────────────────────────────

@app.route("/api/global_model_status")
def api_global_model_status():
    """Lightweight endpoint — returns current GlobalModel stats for badge updates."""
    gm = STATE["global_model"]
    return jsonify({
        "n_samples":         gm.n_samples,
        "n_classes":         gm.n_classes,
        "n_images_labelled": len(gm.n_samples_per_image),
    })


# ── Rebuild global model from session shapes ──────────────────────────────────

@app.route("/api/rebuild_global", methods=["POST"])
def api_rebuild_global():
    """
    (Re-)build the GlobalModel from ALL annotated images in the current session.
    Useful after: importing a session, restarting the server, or changing model.
    Body: {"preproc": str}
    """
    if not STATE["model"]:
        return jsonify({"error": "No model loaded"}), 400
    if not STATE["image_list"]:
        return jsonify({"error": "No image directory set"}), 400
    body    = request.json or {}
    preproc = body.get("preproc", CFG.get("default_preproc", "none"))
    try:
        STATE["global_model"].reset()
        n = _rebuild_global_model(preproc)
        gm = STATE["global_model"]
        return jsonify({
            "ok":               True,
            "images_processed": n,
            "global_model": {
                "n_samples":         gm.n_samples,
                "n_classes":         gm.n_classes,
                "n_images_labelled": len(gm.n_samples_per_image),
            },
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Propagation ───────────────────────────────────────────────────────────────

@app.route("/api/propagate", methods=["POST"])
def api_propagate():
    imgs = STATE["image_list"]
    if not imgs:   return jsonify({"error": "No images loaded"}), 404
    if not STATE["model"]: return jsonify({"error": "No model loaded"}), 400

    body    = request.json or {}
    idx     = int(body.get("idx", 0)) % len(imgs)
    anns    = body.get("annotations", [])
    method  = "centroid"   # SVM removed — centroid only
    scope   = body.get("scope", "local")
    use_ws  = bool(body.get("watershed", False))
    ws_er   = int(body.get("ws_erosions", 3))
    alpha   = float(body.get("alpha", 0.5))
    preproc = body.get("preproc", "none")
    path    = imgs[idx]

    # Local propagation requires annotations on this image.
    # Global propagation can work on any image as long as the model has samples
    # from *any* previously annotated image — current-image annotations optional.
    if not anns and scope == "local":
        return jsonify({"error": "No annotations provided"}), 400

    try:
        e       = _get_features(path, preproc)
        feats   = e["features"]
        image   = Image.open(path).convert("RGB")
        orig_np = np.array(image).astype(np.float32) / 255.0

        if scope == "global":
            gm = STATE["global_model"]

            # If global model is empty, try to rebuild it from session shapes
            # (covers the case where the server was restarted but session exists)
            if gm.n_samples == 0:
                _rebuild_global_model(preproc)
                gm = STATE["global_model"]

            # Add current image annotations if provided (may be empty)
            if anns:
                try:
                    X, y_idx, cid2idx = core.collect_annotation_samples(feats, image.size, anns)
                    idx_to_cid = {v: k for k, v in cid2idx.items()}
                    y_cid = np.array([idx_to_cid[i] for i in y_idx], dtype=np.int32)
                    gm.add_samples(X, y_cid, path.name, cid2idx)
                except Exception as ex:
                    print(f"[propagate] Could not add current-image samples: {ex}")

            if gm.n_samples == 0:
                return jsonify({"error": "Global model has no samples yet — annotate at least one image first."}), 400

            cid_color = {a["class_id"]: a["class_color"] for a in anns}
            for cls in STATE["session"]["classes"]:
                cid_color.setdefault(cls["id"], cls["color"])

            lbl_up, colored, class_info = core.propagate_with_global_model(
                gm, feats, image.size, cid_color,
                method=method,
                use_watershed=use_ws, image=image if use_ws else None,
                watershed_erosions=ws_er,
            )
        else:
            lbl_up, colored, class_info = core.propagate_labels(
                feats, image.size, anns,
                method=method,
                use_watershed=use_ws, image=image if use_ws else None,
                watershed_erosions=ws_er,
            )
            if anns:
                try:
                    X, y_idx, cid2idx = core.collect_annotation_samples(feats, image.size, anns)
                    idx_to_cid = {v: k for k, v in cid2idx.items()}
                    y_cid = np.array([idx_to_cid[i] for i in y_idx], dtype=np.int32)
                    STATE["global_model"].add_samples(X, y_cid, path.name, cid2idx)
                except Exception:
                    pass

        overlay = core.make_overlay(orig_np, colored, alpha)

        with STATE["lock"]:
            STATE["last_prop"][str(path)] = {
                "lbl_up":  lbl_up,
                "colored": colored,
                "orig_np": orig_np,
            }

        saved = _auto_save_mask(path, lbl_up, colored, overlay)
        gm    = STATE["global_model"]
        return jsonify({
            "seg_mask":   _arr_to_b64(colored),
            "overlay":    _arr_to_b64(overlay),
            "class_info": class_info,
            "saved":      saved,
            "global_model": {
                "n_samples":         gm.n_samples,
                "n_classes":         gm.n_classes,
                "n_images_labelled": len(gm.n_samples_per_image),
            },
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Watershed on demand ───────────────────────────────────────────────────────

@app.route("/api/watershed", methods=["POST"])
def api_watershed():
    """
    Apply watershed to the cached propagation result for the current image.
    Does NOT re-run the classifier — uses STATE["last_prop"].
    Body: {"idx": int, "ws_erosions": int, "alpha": float}
    """
    imgs = STATE["image_list"]
    if not imgs: return jsonify({"error": "No images"}), 404
    body   = request.json or {}
    idx    = int(body.get("idx", 0)) % len(imgs)
    ws_er  = int(body.get("ws_erosions", 3))
    alpha  = float(body.get("alpha", 0.5))
    path   = imgs[idx]
    prop   = STATE["last_prop"].get(str(path))

    if prop is None or prop.get("lbl_up") is None:
        return jsonify({"error": "No propagation result cached — run Propagate first."}), 404
    try:
        image   = Image.open(path).convert("RGB")
        orig_np = prop["orig_np"]

        # Refine the cached label map
        refined = core.watershed_refine(image, prop["lbl_up"], n_erosions=ws_er)

        # Re-colour using the same colour per label ID that's already in colored
        # Build cid→color from the existing colored array + refined labels
        old_colored = prop["colored"]
        colored = np.zeros_like(old_colored)
        for lbl in np.unique(refined):
            # Sample the colour from the *previous* colored mask (majority vote)
            prev_pixels = old_colored[prop["lbl_up"] == lbl]
            if len(prev_pixels):
                color = prev_pixels.mean(0)
            else:
                prev_pixels2 = old_colored[refined == lbl]
                color = prev_pixels2.mean(0) if len(prev_pixels2) else np.array([0.5, 0.5, 0.5])
            colored[refined == lbl] = color

        overlay = core.make_overlay(orig_np, colored, alpha)

        # Update cache and save
        with STATE["lock"]:
            STATE["last_prop"][str(path)]["lbl_up"]  = refined
            STATE["last_prop"][str(path)]["colored"] = colored

        saved = _auto_save_mask(path, refined, colored, overlay)
        return jsonify({
            "seg_mask": _arr_to_b64(colored),
            "overlay":  _arr_to_b64(overlay),
            "saved":    saved,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Reblend ───────────────────────────────────────────────────────────────────

@app.route("/api/reblend", methods=["POST"])
def api_reblend():
    imgs  = STATE["image_list"]
    if not imgs: return jsonify({"error": "No images"}), 404
    body  = request.json or {}
    idx   = int(body.get("idx", 0)) % len(imgs)
    alpha = float(body.get("alpha", 0.5))
    path  = imgs[idx]
    prop  = STATE["last_prop"].get(str(path))
    if prop is None:
        return jsonify({"error": "No propagation cached for this image"}), 404
    overlay = core.make_overlay(prop["orig_np"], prop["colored"], alpha)
    # Also update the saved overlay on disk
    _overlay_path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((overlay * 255).clip(0, 255).astype(np.uint8)).save(_overlay_path(path))
    return jsonify({"overlay": _arr_to_b64(overlay)})


# ── Visualisation save (PCA-RGB tab) ─────────────────────────────────────────

@app.route("/api/save", methods=["POST"])
def api_save():
    imgs = STATE["image_list"]
    if not imgs: return jsonify({"error": "No images loaded"}), 404
    body    = request.json or {}
    idx     = int(body.get("idx", 0)) % len(imgs)
    path    = imgs[idx]
    out_dir = _masks_dir(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved   = []

    def _s(key: str, dst: Path):
        b64 = body.get(key)
        if not b64: return
        Image.open(io.BytesIO(base64.b64decode(b64))).save(dst)
        saved.append(str(dst))

    _s("thumb",      out_dir / (path.stem + "_original.png"))
    _s("pca_rgb",    out_dir / (path.stem + "_pca_rgb.png"))
    _s("mask_img",   out_dir / (path.stem + "_kmeans_color.png"))
    _s("label_img",  out_dir / (path.stem + "_kmeans_labels.png"))
    _s("overlay",    out_dir / (path.stem + "_kmeans_overlay.png"))
    return jsonify({"saved": saved, "dir": str(out_dir)})



# ── Batch propagation (whole directory) ──────────────────────────────────────

# Shared state for the running batch job
_BATCH = {
    "running":   False,
    "cancel":    False,
    "total":     0,
    "done":      0,
    "errors":    0,
    "current":   "",
    "log":       [],     # list of str, capped at 200 lines
    "lock":      threading.Lock(),
}

def _batch_log(msg: str) -> None:
    with _BATCH["lock"]:
        _BATCH["log"].append(msg)
        if len(_BATCH["log"]) > 200:
            _BATCH["log"] = _BATCH["log"][-200:]
    print(f"[batch] {msg}")


def _run_batch(imgs, preproc, use_ws, ws_er, alpha, cid_color):
    """
    Worker thread — classify every image with the global model.

    Memory strategy
    ---------------
    • Uses _extract_features_nocache() — features are NEVER stored in
      STATE["cache"], so RAM stays flat regardless of dataset size.
    • classify_pixels_streaming() inside propagate_with_global_model()
      processes patch-strip by patch-strip (default 8 patch rows ≈ 128
      pixel rows), so the full (H, W, D) tensor is never in memory.
    • Peak RAM per image ≈ 1 patch-strip ≈ 750 MB (GPU) + overlay (50 MB).
      After each image the Python GC reclaims everything.
    • Only the last-processed image is kept in STATE["last_prop"] for the
      reblend / watershed endpoints (the previous entry is evicted).
    """
    gm = STATE["global_model"]
    _BATCH["total"]   = len(imgs)
    _BATCH["done"]    = 0
    _BATCH["errors"]  = 0
    _BATCH["log"]     = []
    _batch_log(f"Starting batch: {len(imgs)} images, preproc={preproc}, "
               f"global model: {gm.n_samples} samples / {gm.n_classes} classes")

    for path in imgs:
        if _BATCH["cancel"]:
            _batch_log("Cancelled by user.")
            break
        _BATCH["current"] = path.name
        try:
            # Extract without caching — discard after this iteration
            feats, img_size = _extract_features_nocache(path, preproc)
            image   = Image.open(path).convert("RGB")
            orig_np = np.array(image).astype(np.float32) / 255.0

            lbl_up, colored, _ = core.propagate_with_global_model(
                gm, feats, image.size, cid_color,
                method="centroid",
                use_watershed=use_ws,
                image=image if use_ws else None,
                watershed_erosions=ws_er,
            )
            overlay = core.make_overlay(orig_np, colored, alpha)

            # Keep only the current image in last_prop (evict the previous one)
            with STATE["lock"]:
                STATE["last_prop"].clear()
                STATE["last_prop"][str(path)] = {
                    "lbl_up":  lbl_up,
                    "colored": colored,
                    "orig_np": orig_np,
                }

            saved = _auto_save_mask(path, lbl_up, colored, overlay)
            _BATCH["done"] += 1
            _batch_log(f"[{_BATCH['done']}/{_BATCH['total']}] {path.name} "
                       f"→ {len(saved)} files saved")

            # Explicitly release large arrays before next iteration
            del feats, lbl_up, colored, overlay, orig_np

        except Exception as ex:
            _BATCH["errors"] += 1
            import traceback as _tb
            _batch_log(f"ERROR {path.name}: {ex}")
            _tb.print_exc()

    _BATCH["running"] = False
    _BATCH["current"] = ""
    if _BATCH["cancel"]:
        _batch_log(f"Stopped. {_BATCH['done']} done, {_BATCH['errors']} errors.")
    else:
        _batch_log(f"Done. {_BATCH['done']}/{_BATCH['total']} processed, "
                   f"{_BATCH['errors']} errors.")


@app.route("/api/batch_propagate", methods=["POST"])
def api_batch_propagate():
    """
    Build the global model from all session shapes, then propagate to every
    image in the directory, overwriting existing masks.

    Body:
    {
      "preproc":     str,
      "watershed":   bool,
      "ws_erosions": int,
      "alpha":       float
    }

    Returns immediately with {"started": true}.
    Poll /api/batch_status for progress.
    POST /api/batch_cancel to abort.
    """
    if _BATCH["running"]:
        return jsonify({"error": "A batch job is already running. Cancel it first."}), 409
    if not STATE["model"]:
        return jsonify({"error": "No model loaded."}), 400
    if not STATE["image_list"]:
        return jsonify({"error": "No image directory set."}), 400

    # Ensure global model is up-to-date
    STATE["global_model"].reset()
    n_rebuilt = _rebuild_global_model(CFG.get("default_preproc", "none"))
    gm = STATE["global_model"]
    if gm.n_samples == 0:
        return jsonify({"error": "Global model is empty — annotate at least one image first."}), 400

    body   = request.json or {}
    preproc = body.get("preproc", CFG.get("default_preproc", "none"))
    use_ws  = bool(body.get("watershed", False))
    ws_er   = int(body.get("ws_erosions", 3))
    alpha   = float(body.get("alpha", 0.5))

    # Build colour map from session classes
    cid_color = {c["id"]: c.get("color", [128, 128, 128])
                 for c in STATE["session"]["classes"]}

    imgs = list(STATE["image_list"])

    with _BATCH["lock"]:
        _BATCH["running"] = True
        _BATCH["cancel"]  = False

    threading.Thread(
        target=_run_batch,
        args=(imgs, preproc, use_ws, ws_er, alpha, cid_color),
        daemon=True,
    ).start()

    return jsonify({
        "started":         True,
        "total":           len(imgs),
        "global_samples":  gm.n_samples,
        "global_classes":  gm.n_classes,
        "rebuilt_from":    n_rebuilt,
    })


@app.route("/api/batch_status")
def api_batch_status():
    """Poll this to get live progress."""
    with _BATCH["lock"]:
        return jsonify({
            "running": _BATCH["running"],
            "cancel":  _BATCH["cancel"],
            "total":   _BATCH["total"],
            "done":    _BATCH["done"],
            "errors":  _BATCH["errors"],
            "current": _BATCH["current"],
            "log":     _BATCH["log"][-30:],   # last 30 lines for the UI
        })


@app.route("/api/batch_cancel", methods=["POST"])
def api_batch_cancel():
    with _BATCH["lock"]:
        _BATCH["cancel"] = True
    return jsonify({"ok": True})

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("EUPE Visualizer v6",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--repo_dir",    default=CFG["repo_dir"])
    p.add_argument("--weights_dir", default=CFG["weights_dir"])
    p.add_argument("--cpu",   action="store_true")
    p.add_argument("--host",  default=CFG["host"])
    p.add_argument("--port",  type=int, default=CFG["port"])
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device      : {device}")
    print(f"Repo dir    : {args.repo_dir}")
    print(f"Weights dir : {args.weights_dir}")
    STATE["repo_dir"]    = args.repo_dir
    STATE["weights_dir"] = args.weights_dir
    STATE["device"]      = device
    print(f"Open  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
