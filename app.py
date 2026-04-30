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


def _prefetch(image_path: Path, preproc: str) -> None:
    """Fire-and-forget background feature extraction for the next image."""
    try:
        _get_features(image_path, preproc)
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
    Start a background thread to rebuild the GlobalModel from session shapes.
    Only fires if both model and session shapes are present.
    Silently skips if nothing to do.
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
            e   = _get_features(path, preproc)
            img = Image.open(path).convert("RGB")
            X, y_idx, cid2idx = core.collect_annotation_samples(
                e["features"], img.size, anns)
            idx_to_cid = {v: k for k, v in cid2idx.items()}
            y_cid = np.array([idx_to_cid[i] for i in y_idx], dtype=np.int32)
            gm.add_samples(X, y_cid, fname, cid2idx)
            gm.clf = None
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
    # If model already loaded, rebuild global model from newly-loaded session
    _trigger_background_rebuild()
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
        # Rebuild global model in background if session already has annotations
        _trigger_background_rebuild()
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

    # Prefetch features for adjacent images in background
    preproc = request.args.get("preproc", CFG.get("default_preproc", "none"))
    for delta in (1, -1):
        nxt = (idx + delta) % len(imgs)
        if nxt != idx:
            threading.Thread(
                target=_prefetch, args=(imgs[nxt], preproc), daemon=True
            ).start()

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
    Merge a session.json from another directory into the current session.
    Body: {"source_dir": str}

    Behaviour
    ---------
    • Classes are merged by name: if a class with the same name already exists
      in the current session its id is reused; otherwise a new id is assigned.
      This avoids id collisions when two directories used independently.
    • Shapes are imported per filename, skipping filenames that are not present
      in the current image list (different dataset).
    • class_id_seq is bumped if new IDs are minted.
    • The global model is reset — it will be rebuilt on the next propagation
      from the combined set of annotated images.
    """
    idir = STATE["image_dir"]
    if idir is None:
        return jsonify({"error": "Set an image directory first."}), 400
    body     = request.json or {}
    src_str  = body.get("source_dir", "").strip()
    if not src_str:
        return jsonify({"error": "No source directory provided."}), 400

    # Resolution order:
    #   1. As-is (absolute path)
    #   2. Relative to the current image directory  (e.g. "../34" or just "34")
    #   3. Relative to the app.py working directory
    src = Path(src_str)
    if not src.is_dir() and idir is not None:
        candidate = Path(idir) / src_str
        if candidate.is_dir():
            src = candidate
    if not src.is_dir():
        candidate = Path.cwd() / src_str
        if candidate.is_dir():
            src = candidate
    if not src.is_dir():
        return jsonify({"error": f"Directory not found: '{src_str}' (tried absolute, relative to image dir, and relative to working dir)"}), 400
    src = src.resolve()

    foreign = sess.load(src)
    if not foreign["classes"] and not any(foreign["shapes"].values()):
        return jsonify({"error": "Source session is empty."}), 400

    current   = STATE["session"]
    img_names = {p.name for p in STATE["image_list"]}

    # Build name→id map for current classes
    cur_name_to_id = {c["name"]: c["id"] for c in current["classes"]}

    # id remapping: foreign class id → current class id
    id_remap: dict[int, int] = {}
    for fc in foreign["classes"]:
        if fc["name"] in cur_name_to_id:
            id_remap[fc["id"]] = cur_name_to_id[fc["name"]]
        else:
            # Mint a new id
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

    # Import shapes for images that exist in this directory
    imported_files, imported_shapes = 0, 0
    for fname, shapes in foreign["shapes"].items():
        if fname not in img_names:
            continue   # skip images not in this dataset
        remapped = []
        for s in shapes:
            remapped.append({**s, "class_id": id_remap.get(s["class_id"], s["class_id"])})
        # Merge: keep existing shapes, add imported ones (deduplicate by content)
        existing = current["shapes"].get(fname, [])
        existing_set = {str(e) for e in existing}
        added = [r for r in remapped if str(r) not in existing_set]
        current["shapes"][fname] = existing + added
        if added:
            imported_files += 1
            imported_shapes += len(added)

    sess.save(idir, current)
    # Rebuild global model to include newly imported shapes
    _trigger_background_rebuild()

    return jsonify({
        "ok":              True,
        "imported_files":  imported_files,
        "imported_shapes": imported_shapes,
        "resolved_path":   str(src),
        "session":         sess.export_summary(current),
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
    method  = body.get("method", "centroid")
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
                    gm.clf = None
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
                    STATE["global_model"].clf = None
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
