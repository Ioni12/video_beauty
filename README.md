# Video Beauty Pipeline

An end-to-end Machine Learning pipeline for training a face beauty scoring model and applying it to video streams. The application uses MediaPipe to detect faces, evaluates them using a trained PyTorch computer vision model, and optionally filters or hides faces in the video based on configurable score thresholds.

## Features

- **End-to-End Pipeline**: Handles data preparation, model training, evaluation, and inference.
- **Model Export**: Converts trained PyTorch models to ONNX and TensorFlow Lite (TFLite) for efficient, cross-platform deployment.
- **Video Processing**: Evaluates faces in videos frame-by-frame and applies filtering policies based on customizable thresholds.
- **Visualizations**: Automatically generates score timelines and processing summaries.

## Requirements

The project uses PyTorch, MediaPipe, OpenCV, and optional ONNX/TensorFlow dependencies for model export.

Install all dependencies via `pip`:

```bash
pip install -r requirements.txt
```

> **Note**: TensorFlow and ONNX components in `requirements.txt` are only required if you plan to export the model to TFLite. If you only intend to train the model and run PyTorch inference, these can be safely ignored.

## Project Structure

- `main.py` - The CLI entry point for executing the training and inference pipeline.
- `config/` - Configuration files defining hyperparameters, score thresholds, and environments.
- `data/` - Modules for data preparation and PyTorch dataloaders.
- `model/` - Model architecture definitions and loading utilities.
- `training/` - Training loop and model evaluation logic.
- `face/` - Core face detection functionality (likely utilizing MediaPipe).
- `video/` - Video parsing, face scoring per frame, and output rendering.
- `export/` - Utilities for exporting the best PyTorch model into ONNX and TFLite formats.
- `notebooks/` - Jupyter notebooks for experimentation and analysis.

## Usage Overview

The application supports different execution modes depending on the flags passed to `main.py`.

### 1. Full Pipeline (Data Prep -> Train -> Evaluate -> Export)
Run the complete pipeline from scratch:
```bash
python main.py
```

### 2. Skip Training and Process a Video
If you already have a trained `best_model.pt`, you can skip training and immediately process a video:
```bash
python main.py --skip-train --video path/to/your/video.mp4
```

### 3. Video Processing Only (Inference Mode)
Process a video using an already trained model without running through the pipeline setup routines:
```bash
python main.py --only-video --video path/to/your/video.mp4 \
  --threshold 3.0 \
  --frame-skip 2 \
  --filter-policy any
```

### 4. Export Model Only
If you've already trained a model and just want to generate the TFLite assets:
```bash
python main.py --only-export
```

### Configuration Overrides (Video Mode)

When processing a video, you can override default options found in `config/base.py`:
- `--video` : Path to the input `.mp4` video.
- `--threshold` : Score threshold below which a face gets filtered (default corresponds to `config.SCORE_THRESHOLD`).
- `--frame-skip` : Number of frames to skip to accelerate inference (e.g., `2` processes every 2nd frame).
- `--filter-policy` : Defines how to handle faces in the frame. Options: `any` (if any face fails, filter whole frame), `all` (filter only if all fail), `mean` (filter based on the average score in the frame).

## Output Artifacts

- **Model Checkpoints**: Located in the designated export/training directories (e.g. `best_model.pt`).
- **Video Generation**:
  - `*_annotated.mp4` - Video output with plotted bounding boxes and beauty scores overlayed.
  - `*_cleaned.mp4` - The resulting video where faces below the selected threshold are blurred or hidden according to the filter policy.
- **Reports**: Score timelines and summaries printed to the console or saved to disk.
