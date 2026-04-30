"""
session.py
----------
Persistent annotation session storage.

Layout on disk
--------------
<image_dir>/
  .eupe_session/
    session.json          ← classes + per-image shape lists + metadata

session.json schema
-------------------
{
  "version": 1,
  "classes": [
    {"id": 1, "name": "Road", "color": [255, 0, 0], "hex": "#ff0000"}
  ],
  "class_id_seq": 1,          // highest id ever used
  "shapes": {
    "image001.jpg": [
      {
        "type": "bbox",          // or "polygon"
        "class_id": 1,
        "coords": [x1,y1,x2,y2] // image pixel space
                                 // polygon: [[x,y], …]
      }
    ]
  }
}
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_SESSION_DIR  = ".eupe_session"
_SESSION_FILE = "session.json"
_VERSION      = 1


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _session_path(image_dir: Path) -> Path:
    return image_dir / _SESSION_DIR / _SESSION_FILE


def _empty_session() -> dict:
    return {
        "version":      _VERSION,
        "classes":      [],
        "class_id_seq": 0,
        "shapes":       {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load(image_dir: Path) -> dict:
    """Load (or create) the session for *image_dir*."""
    path = _session_path(image_dir)
    with _LOCK:
        if not path.exists():
            return _empty_session()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # forward-compat: fill in missing keys
            base = _empty_session()
            base.update(data)
            return base
        except Exception:
            return _empty_session()


def save(image_dir: Path, session: dict) -> None:
    """Atomically write the session to disk."""
    path = _session_path(image_dir)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(session, fh, indent=2, ensure_ascii=False)
        tmp.replace(path)


def get_shapes(session: dict, filename: str) -> list[dict]:
    return session["shapes"].get(filename, [])


def set_shapes(session: dict, filename: str, shapes: list[dict]) -> None:
    session["shapes"][filename] = shapes


def upsert_class(session: dict, cls: dict) -> dict:
    """
    Add or update a class entry.
    If cls has no "id", assign one and return the updated cls dict.
    """
    if not cls.get("id"):
        session["class_id_seq"] += 1
        cls = dict(cls, id=session["class_id_seq"])
    # replace existing or append
    existing = [c for c in session["classes"] if c["id"] != cls["id"]]
    session["classes"] = existing + [cls]
    return cls


def delete_class(session: dict, class_id: int) -> None:
    session["classes"] = [c for c in session["classes"] if c["id"] != class_id]
    # remove shapes that reference this class
    for fname in list(session["shapes"].keys()):
        session["shapes"][fname] = [
            s for s in session["shapes"][fname] if s["class_id"] != class_id
        ]


def export_summary(session: dict) -> dict[str, Any]:
    """Return a lightweight summary safe to send to the browser."""
    return {
        "classes":      session["classes"],
        "class_id_seq": session["class_id_seq"],
        "shape_counts": {k: len(v) for k, v in session["shapes"].items()},
    }
