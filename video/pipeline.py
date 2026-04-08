# video/pipeline.py
# ---------------------------------------------------------------
# Full video processing pipeline.
#
# Reads an input video frame by frame, detects and scores all
# faces, and writes two output videos:
#   - annotated  : original frames with bounding boxes + scores
#   - cleaned    : low-scoring faces replaced with background
# ---------------------------------------------------------------
import cv2
import numpy as np
import mediapipe as mp

import torch

from config import SCORE_THRESHOLD, FRAME_SKIP, EXPORT_DIR
from face import FaceLandmarker, options_video
from video.inference import score_all_faces, annotate_frame


# ── Background capture ───────────────────────────────────────────

def capture_background_from_webcam() -> np.ndarray | None:
    """Open the webcam, wait for the user to press Enter, then
    capture a single frame to use as the clean background.

    Returns the BGR frame, or None if the webcam failed.
    Only useful in environments with a physical camera attached
    (local machine).  On Colab, use 'first_frame' mode instead.
    """
    print("Opening webcam — position camera on empty scene, then press Enter.")
    cap = cv2.VideoCapture(0)
    input("Press Enter to capture background...")
    ret, frame = cap.read()
    cap.release()

    if ret:
        print("Background captured from webcam.")
        return frame

    print("Webcam capture failed — falling back to first_frame mode.")
    return None


# ── Main pipeline ────────────────────────────────────────────────

def process_video(
    model: torch.nn.Module,
    input_path: str,
    output_annotated: str | None = None,
    output_cleaned:   str | None = None,
    threshold:        float = SCORE_THRESHOLD,
    frame_skip:       int   = FRAME_SKIP,
    filter_policy:    str   = "any",
    bg_mode:          str   = "first_frame",
    background_frame: np.ndarray | None = None,
) -> tuple[list, float]:
    """Process a video file and optionally write annotated/cleaned outputs.

    Parameters
    ----------
    model            : trained nn.Module in eval mode
    input_path       : path to the source video file
    output_annotated : path for the annotated output video, or None to skip
    output_cleaned   : path for the cleaned output video, or None to skip
    threshold        : faces scoring below this are hidden in the cleaned video
    frame_skip       : run detection every N frames (1 = every frame)
    filter_policy    : how to decide which faces to hide —
                         'any'  hide a face if that face < threshold
                         'all'  hide all faces only if every face < threshold
                         'mean' hide all faces if the mean score < threshold
    bg_mode          : where to get the background for cleaning —
                         'first_frame'  use the first frame of the video
                         'webcam'       capture from webcam (local only)
                         'provided'     use the *background_frame* argument
    background_frame : a pre-captured BGR background frame (used when
                       bg_mode='provided')

    Returns
    -------
    frame_scores : list of (frame_index, [score, ...]) tuples
    fps          : frames-per-second of the source video
    """
    model.eval()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer_ann     = cv2.VideoWriter(output_annotated, fourcc, fps, (W, H)) if output_annotated else None
    writer_cleaned = cv2.VideoWriter(output_cleaned,   fourcc, fps, (W, H)) if output_cleaned   else None

    # Resolve background source
    if bg_mode == "webcam":
        background_frame = capture_background_from_webcam()
        if background_frame is None:
            bg_mode = "first_frame"   # fall back gracefully

    frame_scores      = []
    last_face_results = []
    frame_idx         = 0

    with FaceLandmarker.create_from_options(options_video) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Capture background from the first frame if requested
            if frame_idx == 0 and bg_mode == "first_frame":
                background_frame = frame.copy()
                print("Background set from first frame.")

            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # Run detection + scoring every frame_skip frames;
            # reuse last results on skipped frames
            if frame_idx % frame_skip == 0:
                mp_img  = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                results = landmarker.detect_for_video(mp_img, ts_ms)
                last_face_results = (
                    score_all_faces(model, frame, results.face_landmarks)
                    if results.face_landmarks else []
                )

            face_results = last_face_results
            scores       = [f["score"] for f in face_results if f["score"] is not None]
            frame_scores.append((frame_idx, scores))

            # Write annotated frame
            if writer_ann:
                writer_ann.write(annotate_frame(frame, face_results))

            # Write cleaned frame (replace low-score faces with background)
            if writer_cleaned and background_frame is not None:
                cleaned = frame.copy()
                bg      = cv2.resize(background_frame, (W, H))

                for face in face_results:
                    s = face["score"]
                    if s is None:
                        continue

                    hide = _should_hide(s, scores, face_results, threshold, filter_policy)
                    if hide:
                        x1, y1, x2, y2 = face["bbox"]
                        cleaned[y1:y2, x1:x2] = bg[y1:y2, x1:x2]

                writer_cleaned.write(cleaned)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx}/{total}  "
                      f"faces={len(face_results)}  scores={scores}")

    cap.release()
    if writer_ann:
        writer_ann.release()
        print(f"Annotated video → {output_annotated}")
    if writer_cleaned:
        writer_cleaned.release()
        print(f"Cleaned video   → {output_cleaned}")

    return frame_scores, fps


# ── Filter policy helper ─────────────────────────────────────────

def _should_hide(
    score: float,
    all_scores: list,
    face_results: list,
    threshold: float,
    policy: str,
) -> bool:
    """Return True if this face should be replaced with background."""
    if policy == "any":
        return score < threshold
    if policy == "all":
        return all(
            f["score"] is not None and f["score"] < threshold
            for f in face_results
        )
    if policy == "mean":
        return (np.mean(all_scores) < threshold) if all_scores else False
    # Default fallback
    return score < threshold