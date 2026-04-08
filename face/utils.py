# face/utils.py
# ---------------------------------------------------------------
# All face geometry helpers: alignment, cropping, landmark
# remapping.  No model inference lives here — pure CV utilities.
# ---------------------------------------------------------------
import math
import cv2
import mediapipe as mp
import numpy as np

from config import MAX_FACES, FACE_MODEL_PATH


# ── MediaPipe option objects ─────────────────────────────────────
# Built once at import time.  Two separate option sets are needed
# because MediaPipe requires you to declare the running mode up
# front: IMAGE mode for still photos (training data prep) and
# VIDEO mode for frame-by-frame video processing.

BaseOptions           = mp.tasks.BaseOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options_image = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=MAX_FACES,
)

options_video = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=MAX_FACES,
)

# A single shared IMAGE-mode landmarker (used during dataset prep).
# VIDEO-mode landmarkers must be created fresh per video inside a
# `with FaceLandmarker.create_from_options(options_video) as lm:`
# block because they are stateful (they track timestamps).
landmarker_image = FaceLandmarker.create_from_options(options_image)


# ── Internal helpers ─────────────────────────────────────────────

def _rotate_to_landmarks(img_bgr: np.ndarray, landmarks: list) -> np.ndarray:
    """Rotate *img_bgr* so that the eyes are level.

    Uses landmark indices 33 (left eye outer corner) and 263
    (right eye outer corner) — stable across MediaPipe versions.
    """
    h, w = img_bgr.shape[:2]
    left_eye  = landmarks[33]
    right_eye = landmarks[263]
    cx1, cy1  = int(left_eye.x  * w), int(left_eye.y  * h)
    cx2, cy2  = int(right_eye.x * w), int(right_eye.y * h)

    angle = math.degrees(math.atan2(cy2 - cy1, cx2 - cx1))
    mid   = ((cx1 + cx2) / 2, (cy1 + cy2) / 2)
    M     = cv2.getRotationMatrix2D(mid, angle, 1.0)
    return cv2.warpAffine(
        img_bgr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )


class _FakeLandmark:
    """Lightweight stand-in for a MediaPipe NormalizedLandmark."""
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def _remap_landmarks(landmarks: list, frame: np.ndarray, bbox: tuple) -> list:
    """Re-express frame-level landmark coordinates as relative coords
    within a crop bounding box.

    MediaPipe landmarks are normalised to the *full frame*.  After we
    crop a face region we need the same landmarks expressed relative
    to the crop so that ``_rotate_to_landmarks`` works correctly on
    the cropped image.

    Parameters
    ----------
    landmarks : list of MediaPipe NormalizedLandmark
    frame     : the original full-resolution BGR frame
    bbox      : (x1, y1, x2, y2) pixel coords of the crop within *frame*
    """
    h, w          = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    cw = max(x2 - x1, 1)
    ch = max(y2 - y1, 1)

    return [
        _FakeLandmark(
            x=(lm.x * w - x1) / cw,
            y=(lm.y * h - y1) / ch,
        )
        for lm in landmarks
    ]


# ── Public API ───────────────────────────────────────────────────

def align_face(img_bgr: np.ndarray) -> np.ndarray:
    """Detect and eye-align the first face found in *img_bgr*.

    Used during **training data preparation** on still images.
    Returns the original image unchanged if no face is detected.

    Parameters
    ----------
    img_bgr : BGR image as a NumPy array (as returned by cv2.imread)
    """
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
    )
    results = landmarker_image.detect(mp_image)
    if not results.face_landmarks:
        return img_bgr
    return _rotate_to_landmarks(img_bgr, results.face_landmarks[0])


def crop_face(img_bgr: np.ndarray, landmarks: list, pad: float = 0.25):
    """Crop a padded bounding box around one face.

    Used during **video inference** where we already have landmarks
    from a full-frame detection pass.

    Parameters
    ----------
    img_bgr   : full BGR frame
    landmarks : MediaPipe face landmarks for this face
    pad       : fractional padding added around the tight bbox
                (0.25 = 25 % of the face width/height on each side)

    Returns
    -------
    crop : BGR crop of the face region
    bbox : (x1, y1, x2, y2) pixel coordinates within *img_bgr*
    """
    h, w = img_bgr.shape[:2]
    xs   = [int(lm.x * w) for lm in landmarks]
    ys   = [int(lm.y * h) for lm in landmarks]
    x1, x2 = max(0, min(xs)), min(w, max(xs))
    y1, y2 = max(0, min(ys)), min(h, max(ys))

    px = int((x2 - x1) * pad)
    py = int((y2 - y1) * pad)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(w, x2 + px);  y2 = min(h, y2 + py)

    return img_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


def align_crop(frame: np.ndarray, landmarks: list, pad: float = 0.25) -> np.ndarray:
    """Crop **and** eye-align a face from a full video frame in one call.

    This is the function used by the video inference pipeline.
    Combines ``crop_face`` + ``_remap_landmarks`` + ``_rotate_to_landmarks``.

    Parameters
    ----------
    frame     : full BGR frame
    landmarks : MediaPipe face landmarks for this face
    pad       : passed through to ``crop_face``

    Returns
    -------
    Aligned BGR crop, or an empty array if the crop has zero size.
    """
    crop, bbox = crop_face(frame, landmarks, pad=pad)
    if crop.size == 0:
        return crop
    remapped = _remap_landmarks(landmarks, frame, bbox)
    return _rotate_to_landmarks(crop, remapped)