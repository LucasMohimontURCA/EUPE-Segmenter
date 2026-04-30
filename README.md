# EUPE Segmenter

An interactive web-based tool for **dense segmentation** of images using [EUPE](https://github.com/your-org/EUPE) (End-to-end Unified Patch Embeddings) features.

Draw a handful of bounding boxes or polygons on a few representative images, then let the tool propagate semantic labels to the rest of your dataset automatically.

> **Screenshots** — *(add your own here)*

---

## Features

- **Annotate** — draw bounding boxes and polygons per class directly on images
- **Label propagation** — nearest-centroid or linear SVM in EUPE feature space
- **Global model** — annotations accumulate across images; the classifier improves with each new labelled image
- **Session persistence** — classes and shapes auto-saved per directory; survive server restarts
- **Session import** — merge labels from another directory (different dataset, same classes)
- **Watershed refinement** — snap coarse boundaries to image edges via OpenCV watershed
- **Image preprocessing** — CLAHE, Single-Scale Retinex, Multi-Scale Retinex (MSRCR), MSR+CLAHE
- **PCA-RGB visualisation** — inspect the EUPE feature space visually
- **K-Means clustering** — unsupervised segmentation directly in feature space
- **Fast GPU inference** — batched tiled inference on any image size
- **Auto-save** — masks written to `Masks/` on every propagation

---

## Requirements

| Component | Version |
|-----------|---------|
| Python | 3.10 or 3.11 |
| CUDA | 11.8 or 12.x (optional — CPU also works) |
| RAM | 8 GB minimum, 16 GB recommended |
| GPU VRAM | 4 GB minimum for ViT-B (6 GB recommended) |

---

## Installation

### 1. Clone the EUPE repository

```bash
git clone https://github.com/your-org/EUPE.git
cd EUPE
```

### 2. Clone this tool into the same parent directory

```bash
cd ..
git clone https://github.com/your-org/eupe-visualizer.git
cd eupe-visualizer
```

Your directory structure should look like:

```
parent/
  EUPE/                   ← EUPE source code
    weights/
      EUPE-ViT-B.pt       ← downloaded separately (see below)
  eupe-visualizer/        ← this repository
    app.py
    config.json
    ...
```

### 3. Install Python dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Install PyTorch first (match your CUDA version)
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install flask pillow scikit-learn opencv-python-headless numpy
```

### 4. Download model weights

Place the weight files in `EUPE/weights/`:

| File | Description |
|------|-------------|
| `EUPE-ViT-B.pt` | ViT-B/16 backbone (recommended) |
| `EUPE-ViT-S.pt` | ViT-S/16 — faster, slightly lower quality |
| `EUPE-ViT-Ti.pt` | ViT-Ti/16 — fastest |

> Download links are provided in the [EUPE repository](https://github.com/your-org/EUPE).

### 5. Edit `config.json`

```json
{
  "repo_dir":        "EUPE",
  "weights_dir":     "EUPE/weights",
  "default_model":   "eupe_vitb16",
  "default_weights": "EUPE-ViT-B.pt",
  "host":            "127.0.0.1",
  "port":            5000,
  "tile_px":         448,
  "overlap":         0.25,
  "max_batch":       16
}
```

> `repo_dir` and `weights_dir` can be absolute paths (`C:/path/to/EUPE`) if the directories are not relative to `app.py`.

---

## Running

```bash
cd eupe-visualizer
python app.py
```

Then open **http://localhost:5000** in your browser.

### First launch

The **Config** modal opens automatically. You will be prompted to:

1. Set the **image directory** (the folder containing your images to segment)
2. Select the **model architecture** (default: `eupe_vitb16`)
3. Select the **weights file** (default: `EUPE-ViT-B.pt`)
4. Optionally **import a session** from another directory (see [Session Import](#session-import))
5. Click **Load Model & Start**

---

## Workflow

### Annotating images

1. Switch to the **Annotate** tab (default)
2. Create **classes** in the right panel (name + colour)
3. Select a class, then draw on the image:
   - **Box tool** (`B`): click and drag
   - **Polygon tool** (`P`): click to add vertices, click the first point (or double-click) to close
4. Annotations are **saved automatically** — no manual save needed

### Propagating labels

1. After annotating, click **▶ Propagate Labels**
2. The result appears in the strip below the canvas
3. Use the **overlay blend** slider to mix the mask with the original
4. Click **Apply Watershed** to snap boundaries to image edges

### Navigating between images

- Use the **← →** arrow buttons or keyboard arrow keys
- Existing masks load automatically when you navigate to a previously processed image

### Global propagation (recommended workflow)

1. Annotate **3–10 representative images** covering all classes
2. Switch **Scope → Global** in the propagation panel
3. For any subsequent image (even one with no annotations), click **▶ Propagate Labels**
   — the classifier trained on all previous images is applied
4. Use **⟳ Rebuild Global Model** if you restart the server or import a new session

### Session Import

To reuse labels from a different dataset:

1. Open **⚙ Config**
2. Set your current **image directory**
3. Enter the **source directory** in the *Import session* field
4. Click **Import** — classes and shapes are merged into the current session
5. Click **⟳ Rebuild Global Model** in the Annotate sidebar to feed all imported labels into the classifier

---

## Output files

All outputs are saved to `<image_dir>/Masks/`:

| File | Description |
|------|-------------|
| `<stem>.png` | Segmentation mask (class colours) |
| `<stem>_overlay.png` | Mask blended with original image |
| `<stem>_labelmap.npy` | Integer array `(H, W)` of class IDs — use this for training |
| `<stem>_pca_rgb.png` | PCA-RGB visualisation (PCA-RGB tab) |
| `<stem>_kmeans_color.png` | K-Means mask with mean original colours |
| `<stem>_kmeans_labels.png` | K-Means flat-colour label map |

### Using the label map in PyTorch

```python
import numpy as np

lbl = np.load("Masks/image001_labelmap.npy")   # (H, W) int32
# class IDs are the integer IDs from the session
print(np.unique(lbl))    # e.g. [1, 3, 5]
```

---

## Performance tuning (`config.json`)

| Key | Default | Effect |
|-----|---------|--------|
| `tile_px` | `448` | Larger = fewer GPU calls = faster; reduce if you see tiling artefacts |
| `overlap` | `0.25` | Lower = faster; increase to `0.5` for highest boundary quality |
| `max_batch` | `16` | Reduce to `8` if you get GPU out-of-memory errors |

---

## Building a Windows EXE

```bat
cd eupe-visualizer
build_exe.bat
```

The distributable folder is `dist/EUPE_Visualizer/`. Copy your `EUPE/` directory and weights alongside it before distributing.

See [`build_exe.bat`](build_exe.bat) and [`eupe_visualizer.spec`](eupe_visualizer.spec) for build details.

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` | Previous / next image |
| `B` | Bounding-box tool |
| `P` | Polygon tool |
| `Enter` | Close polygon |
| `Del` / `Backspace` | Remove last shape (or last polygon point mid-draw) |
| `Esc` | Cancel in-progress shape / close lightbox |
| Click any result card | Open full-screen lightbox |

---

## Project structure

```
eupe-visualizer/
├── app.py                 Flask server — routes only
├── eupe_core.py           ML: feature extraction, PCA, k-means, propagation
├── preprocessing.py       CLAHE / Retinex image enhancement
├── session.py             JSON-based annotation persistence
├── config.json            Default paths and performance settings
├── templates/
│   └── index.html         Single-page web frontend
├── eupe_visualizer.spec   PyInstaller build spec
├── build_exe.bat          Windows EXE build script
└── IMPLEMENTATION.md      Detailed implementation reference
```

---

## Troubleshooting

**`torch.hub.load` fails / model not found**
- Verify `repo_dir` in `config.json` points to the EUPE clone
- Try an absolute path: `"repo_dir": "C:/Users/you/EUPE"`

**CUDA out of memory**
- Reduce `max_batch` in `config.json` to `4` or `8`
- Switch to a smaller model (`eupe_vits16` or `eupe_vitt16`)

**Propagation gives wrong colours on masked images**
- Make sure you are on the correct class before drawing
- Use **⟳ Rebuild Global Model** after importing sessions or restarting

**Slow first inference**
- Expected — PyTorch JIT-compiles CUDA kernels on the first call. Subsequent calls are fast.

**`cv2.error` on Watershed**
- Make sure OpenCV was installed: `pip install opencv-python-headless`
- Watershed requires ≥ 2 classes in the propagation result

---

## Citation

If you use this tool in your research, please cite the EUPE paper:

```bibtex
@article{eupe2024,
  title   = {EUPE: End-to-end Unified Patch Embeddings for Dense Prediction},
  author  = {…},
  journal = {…},
  year    = {2024}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
