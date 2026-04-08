# video/inference.py
# ---------------------------------------------------------------
# Per-face scoring and frame annotation.
#
# This file only knows about a single frame at a time.
# It does not know about videos, files, or pipelines.
# ---------------------------------------------------------------
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from config import DEVICE, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from face import align_crop


# ── Inference transform ──────────────────────────────────────────
# Same as val_transform in data/dataset.py but defined here
# independently so the video module has no dependency on data/.

_infer_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Visual constants ─────────────────────────────────────────────
# One distinct border colour per face slot so you can track
# individuals across frames.  BGR format (OpenCV convention).
FACE_COLOURS = [
    (255, 200,   0),   # Face 1 — gold
    (  0, 200, 255),   # Face 2 — cyan
    (200,   0, 255),   # Face 3 — violet
    (  0, 255, 128),   # Face 4 — mint
    (128,   0, 255),   # Face 5 — purple
]


# ── Scoring helpers ──────────────────────────────────────────────

def score_colour(score: float) -> tuple:
    """Return a BGR colour that grades from red (low) to green (high).

    Maps the [1, 5] score range to a red→green gradient so the
    on-screen label gives an instant visual cue.
    """
    t = (score - 1.0) / 4.0
    return (0, int(255 * t), int(255 * (1 - t)))


def score_crop(model: torch.nn.Module, crop_bgr: np.ndarray):
    """Score a single aligned face crop and return a float in [1, 5].

    Parameters
    ----------
    model    : trained nn.Module in eval mode
    crop_bgr : BGR NumPy array of any size (will be resized)

    Returns
    -------
    float score in [1, 5], or None if the crop is empty/invalid
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    t   = _infer_transform(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        s = model(t).item()

    return float(np.clip(s, 1.0, 5.0))


def score_all_faces(model: torch.nn.Module, frame: np.ndarray,
                    face_landmarks_list: list) -> list:
    """Score every detected face in *frame*.

    Parameters
    ----------
    model               : trained nn.Module in eval mode
    frame               : full BGR frame
    face_landmarks_list : list of MediaPipe face landmark sets
                          (one entry per detected face)

    Returns
    -------
    List of dicts, one per face:
        { "score": float | None, "bbox": (x1, y1, x2, y2) }
    """
    from face import crop_face   # local import to avoid circular at module level

    results = []
    for landmarks in face_landmarks_list:
        _, bbox    = crop_face(frame, landmarks, pad=0.25)
        aligned    = align_crop(frame, landmarks, pad=0.25)
        results.append({
            "score": score_crop(model, aligned),
            "bbox":  bbox,
        })
    return results


# ── Frame annotation ─────────────────────────────────────────────

def annotate_frame(frame: np.ndarray, face_results: list) -> np.ndarray:
    """Draw bounding boxes, score labels, and a summary panel.

    Does not modify *frame* in place — works on a copy.

    Parameters
    ----------
    frame        : original BGR frame
    face_results : output of score_all_faces()

    Returns
    -------
    Annotated BGR frame (same size as input)
    """
    ann = frame.copy()

    for i, face in enumerate(face_results):
        s               = face["score"]
        x1, y1, x2, y2 = face["bbox"]
        border_col      = FACE_COLOURS[i % len(FACE_COLOURS)]
        text_col        = score_colour(s) if s is not None else (160, 160, 160)
        label           = f"F{i+1}: {s:.2f}" if s is not None else f"F{i+1}: ?"

        # Bounding box
        cv2.rectangle(ann, (x1, y1), (x2, y2), border_col, 2)

        # Score label with dark background for readability
        lbl_y            = max(y1 - 8, 20)
        (tw, th), _      = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(ann, (x1, lbl_y - th - 4), (x1 + tw + 4, lbl_y + 4), (0, 0, 0), -1)
        cv2.putText(ann, label, (x1 + 2, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_col, 2, cv2.LINE_AA)

    # Summary panel — top-left corner
    panel_h = 30 + 28 * max(len(face_results), 1)
    cv2.rectangle(ann, (6, 6), (220, panel_h), (0, 0, 0), -1)
    cv2.putText(ann, f"Faces: {len(face_results)}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    for i, face in enumerate(face_results):
        s    = face["score"]
        col  = score_colour(s) if s is not None else (160, 160, 160)
        text = f"  F{i+1}: {s:.2f}" if s is not None else f"  F{i+1}: n/a"
        cv2.putText(ann, text, (12, 26 + 28 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 1, cv2.LINE_AA)

    return ann