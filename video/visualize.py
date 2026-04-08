# video/visualize.py
# ---------------------------------------------------------------
# Post-processing visualisations for video pipeline output.
#
# Both functions take the frame_scores list returned by
# process_video() so they can be called immediately after
# processing or re-run later on saved data.
# ---------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt

from config import SCORE_THRESHOLD, EXPORT_DIR
from video.inference import FACE_COLOURS


def plot_score_timeline(
    frame_scores: list,
    fps: float,
    threshold: float = SCORE_THRESHOLD,
) -> None:
    """Plot per-face beauty scores over time as a line chart.

    One line per face slot, a dashed threshold line, saved to
    EXPORT_DIR/score_timeline.png and shown inline.

    Parameters
    ----------
    frame_scores : list of (frame_index, [score, ...]) from process_video()
    fps          : frames per second of the source video
    threshold    : drawn as a horizontal reference line
    """
    max_faces = max((len(s) for _, s in frame_scores), default=0)
    if max_faces == 0:
        print("No scored frames — nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    for fi in range(max_faces):
        times, vals = [], []
        for idx, s_list in frame_scores:
            if fi < len(s_list) and s_list[fi] is not None:
                times.append(idx / fps)
                vals.append(s_list[fi])

        if times:
            # Convert BGR tuple → normalised RGB for matplotlib
            bgr      = FACE_COLOURS[fi % len(FACE_COLOURS)]
            rgb_norm = tuple(c / 255 for c in reversed(bgr))
            ax.plot(times, vals, lw=1, label=f"Face {fi + 1}", color=rgb_norm)

    ax.axhline(threshold, color="tomato", ls="--", lw=1.5,
               label=f"Threshold ({threshold})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score (1–5)")
    ax.set_title("Per-Frame Beauty Scores — All Faces")
    ax.legend()
    fig.tight_layout()

    out = os.path.join(EXPORT_DIR, "score_timeline.png")
    fig.savefig(out, dpi=150)
    plt.show()
    print(f"Timeline saved → {out}")


def print_summary(
    frame_scores: list,
    fps: float,
    threshold: float = SCORE_THRESHOLD,
) -> None:
    """Print aggregate statistics for a processed video.

    Parameters
    ----------
    frame_scores : list of (frame_index, [score, ...]) from process_video()
    fps          : frames per second (used to report duration)
    threshold    : used to compute kept/filtered counts
    """
    all_scores = [
        s for _, s_list in frame_scores for s in s_list if s is not None
    ]

    if not all_scores:
        print("No face detections in this video.")
        return

    kept    = sum(1 for s in all_scores if s >= threshold)
    total   = len(all_scores)
    duration = len(frame_scores) / fps if fps > 0 else 0

    print("\n" + "=" * 46)
    print("VIDEO SUMMARY")
    print("=" * 46)
    print(f"  Duration              : {duration:.1f}s  ({len(frame_scores)} frames)")
    print(f"  Total face-frames     : {total}")
    print(f"  Score  mean / std     : {np.mean(all_scores):.3f} / {np.std(all_scores):.3f}")
    print(f"  Score  min  / max     : {min(all_scores):.3f} / {max(all_scores):.3f}")
    print(f"  Above threshold ({threshold}) : {kept}/{total}  ({kept/total*100:.1f}%)")
    print("=" * 46 + "\n")