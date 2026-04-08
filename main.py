# main.py
# ---------------------------------------------------------------
# CLI entry point for running the full pipeline outside Colab.
# Works on a local machine or any Linux server with a GPU.
#
# Usage examples:
#
#   # Full pipeline: prepare → train → evaluate → export
#   python main.py
#
#   # Skip training (use existing checkpoint) and just run video
#   python main.py --skip-train --video my_video.mp4
#
#   # Only export to TFLite (model already trained)
#   python main.py --only-export
#
#   # Only process a video (model already trained)
#   python main.py --only-video --video my_video.mp4
# ---------------------------------------------------------------
import argparse
import os
from pathlib import Path

from config import print_env_info, EXPORT_DIR, make_output_dirs
from data    import prepare_data, make_dataloaders
from model   import load_best_model, build_model
from training import train, evaluate
from export  import export_pipeline
from video   import process_video, plot_score_timeline, print_summary


def parse_args():
    p = argparse.ArgumentParser(description="Video Beauty Pipeline")
    p.add_argument("--skip-train",  action="store_true",
                   help="Skip training and load existing best_model.pt")
    p.add_argument("--only-export", action="store_true",
                   help="Only run TFLite export (model must already be trained)")
    p.add_argument("--only-video",  action="store_true",
                   help="Only run video processing (model must already be trained)")
    p.add_argument("--video",       type=str, default=None,
                   help="Path to input video file")
    p.add_argument("--threshold",   type=float, default=None,
                   help="Score threshold for face filtering (overrides config)")
    p.add_argument("--frame-skip",  type=int,   default=None,
                   help="Process every N frames (overrides config)")
    p.add_argument("--filter-policy", type=str, default="any",
                   choices=["any", "all", "mean"],
                   help="Face hide policy: any | all | mean")
    return p.parse_args()


def run_video(model, video_path: str, threshold, frame_skip, filter_policy):
    """Helper that processes a video and prints results."""
    from config import SCORE_THRESHOLD, FRAME_SKIP
    threshold  = threshold  or SCORE_THRESHOLD
    frame_skip = frame_skip or FRAME_SKIP

    stem   = Path(video_path).stem
    ann    = os.path.join(EXPORT_DIR, f"{stem}_annotated.mp4")
    clean  = os.path.join(EXPORT_DIR, f"{stem}_cleaned.mp4")

    frame_scores, fps = process_video(
        model            = model,
        input_path       = video_path,
        output_annotated = ann,
        output_cleaned   = clean,
        threshold        = threshold,
        frame_skip       = frame_skip,
        filter_policy    = filter_policy,
        bg_mode          = "first_frame",
    )
    print_summary(frame_scores, fps, threshold)
    plot_score_timeline(frame_scores, fps, threshold)


def main():
    args = parse_args()
    make_output_dirs()

    print("=" * 50)
    print_env_info()
    print("=" * 50 + "\n")

    # ── Only export ──────────────────────────────────────────────
    if args.only_export:
        print("→ Export only mode")
        _, val_df, _ = prepare_data()
        _, val_loader, _ = make_dataloaders(None, val_df, None)  # only val needed
        model = load_best_model(build_model())
        export_pipeline(model, val_loader)
        return

    # ── Only video ───────────────────────────────────────────────
    if args.only_video:
        if not args.video:
            raise ValueError("--video PATH is required with --only-video")
        print("→ Video only mode")
        model = load_best_model(build_model())
        run_video(model, args.video, args.threshold, args.frame_skip, args.filter_policy)
        return

    # ── Full pipeline ────────────────────────────────────────────
    print("Step 1/4 — Preparing data")
    train_df, val_df, test_df = prepare_data()
    train_loader, val_loader, test_loader = make_dataloaders(train_df, val_df, test_df)

    if args.skip_train:
        print("\nStep 2/4 — Skipping training, loading existing checkpoint")
        model = load_best_model(build_model())
    else:
        print("\nStep 2/4 — Training")
        model = train(train_loader, val_loader)

    print("\nStep 3/4 — Evaluating on test set")
    evaluate(model, test_loader)

    print("\nStep 4/4 — Exporting to TFLite")
    export_pipeline(model, val_loader)

    if args.video:
        print("\nBonus — Processing video")
        run_video(model, args.video, args.threshold, args.frame_skip, args.filter_policy)


if __name__ == "__main__":
    main()