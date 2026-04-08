# export/tflite.py
# ---------------------------------------------------------------
# Exports the trained PyTorch model to TFLite for Android deployment.
#
# Pipeline:
#   PyTorch (.pt)
#     → ONNX          (opset 18, fixed batch=1)
#     → ONNX simplified (onnxsim, optional but recommended)
#     → TF SavedModel  (onnx2tf)
#     → TFLite INT8    (post-training quantisation, falls back to FP16)
#
# After export, prints the full Android integration spec so you
# know exactly how to pre-process frames on the device.
# ---------------------------------------------------------------
import os
import subprocess

import numpy as np
import onnx
import torch

from config import DEVICE, EXPORT_DIR, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, make_output_dirs


# Fixed paths — all outputs land in EXPORT_DIR
_ONNX_PATH      = lambda: os.path.join(EXPORT_DIR, "mobilenetv3_fbp.onnx")
_ONNX_SIM_PATH  = lambda: os.path.join(EXPORT_DIR, "mobilenetv3_fbp_sim.onnx")
_TF_MODEL_DIR   = lambda: os.path.join(EXPORT_DIR, "tf_savedmodel")
_TFLITE_INT8    = lambda: os.path.join(EXPORT_DIR, "attractiveness_int8.tflite")
_TFLITE_FP16    = lambda: os.path.join(EXPORT_DIR, "attractiveness_fp16.tflite")


# ── Step 1: PyTorch → ONNX ──────────────────────────────────────

def export_onnx(model: torch.nn.Module) -> str:
    """Export *model* to ONNX (opset 18, batch size = 1).

    Returns
    -------
    Path to the saved .onnx file.
    """
    make_output_dirs()
    os.makedirs(_TF_MODEL_DIR(), exist_ok=True)

    path  = _ONNX_PATH()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

    torch.onnx.export(
        model, dummy, path,
        opset_version=18,
        input_names=["input"],
        output_names=["score"],
    )
    onnx.checker.check_model(onnx.load(path))
    print(f"ONNX saved       → {path}")
    return path


# ── Step 2: Simplify ONNX (optional) ────────────────────────────

def simplify_onnx(onnx_path: str) -> str:
    """Run onnxsim to simplify the graph.

    Simplification fuses operations and removes redundant nodes,
    which makes the onnx2tf conversion more reliable.

    Returns
    -------
    Path to the simplified file, or *onnx_path* if onnxsim failed.
    """
    sim_path = _ONNX_SIM_PATH()
    try:
        res = subprocess.run(
            ["onnxsim", onnx_path, sim_path],
            capture_output=True, text=True, timeout=120,
        )
        if res.returncode == 0:
            print(f"ONNX simplified  → {sim_path}")
            return sim_path
        print(f"onnxsim failed (rc={res.returncode}) — using raw ONNX.")
    except Exception as e:
        print(f"onnxsim error ({e}) — using raw ONNX.")
    return onnx_path


# ── Step 3: ONNX → TF SavedModel ────────────────────────────────

def export_tf_savedmodel(onnx_path: str) -> str:
    """Convert ONNX to TensorFlow SavedModel via onnx2tf.

    The  -k input  flag tells onnx2tf to keep the input tensor
    in NCHW layout and transpose internally to NHWC, which is
    what Android/TFLite expects.

    Returns
    -------
    Path to the TF SavedModel directory.
    """
    tf_dir = _TF_MODEL_DIR()
    result = subprocess.run(
        ["onnx2tf", "-i", onnx_path, "-o", tf_dir, "--non_verbose", "-k", "input"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise RuntimeError("onnx2tf conversion failed — see output above.")
    print(f"TF SavedModel    → {tf_dir}")
    return tf_dir


# ── Step 4: TF SavedModel → TFLite ──────────────────────────────

def _representative_dataset(val_loader):
    """Yield calibration batches for INT8 post-training quantisation.

    Images are un-normalised back to [0,1] pixel values and
    converted to NHWC float32, matching what Android will send.
    """
    mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std_t  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    for imgs, _ in val_loader:
        for img_t in imgs[:8]:
            # Undo torchvision normalisation → [0,1], then NHWC
            img_np = (
                (img_t * std_t + mean_t)
                .permute(1, 2, 0)
                .numpy()
                .astype(np.float32)
            )
            yield [np.expand_dims(img_np, 0)]   # shape (1, 224, 224, 3)


def export_tflite(tf_dir: str, val_loader) -> str:
    """Convert TF SavedModel to TFLite.

    Tries INT8 quantisation first (smallest + fastest on Android).
    Falls back to FP16 if INT8 conversion fails.

    Parameters
    ----------
    tf_dir     : path returned by export_tf_savedmodel()
    val_loader : DataLoader used to generate calibration data

    Returns
    -------
    Path to the final .tflite file.
    """
    import tensorflow as tf

    # ── INT8 attempt ─────────────────────────────────────────────
    tflite_path = _TFLITE_INT8()
    try:
        conv = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
        conv.optimizations              = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset     = lambda: _representative_dataset(val_loader)
        conv.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type       = tf.float32   # keep float I/O for easier Android code
        conv.inference_output_type      = tf.float32
        with open(tflite_path, "wb") as f:
            f.write(conv.convert())
        size_kb = os.path.getsize(tflite_path) / 1024
        print(f"INT8 TFLite      → {tflite_path}  ({size_kb:.1f} KB)")
        return tflite_path

    except Exception as e:
        print(f"INT8 failed ({e}) — falling back to FP16.")

    # ── FP16 fallback ─────────────────────────────────────────────
    tflite_path = _TFLITE_FP16()
    conv2 = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
    conv2.optimizations = [tf.lite.Optimize.DEFAULT]
    conv2.target_spec.supported_types = [tf.float16]
    with open(tflite_path, "wb") as f:
        f.write(conv2.convert())
    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"FP16 TFLite      → {tflite_path}  ({size_kb:.1f} KB)")
    return tflite_path


# ── Step 5: Verify + print Android spec ─────────────────────────

def verify_and_print_spec(tflite_path: str) -> None:
    """Run a dummy inference to verify the TFLite model, then print
    the Android integration specification.

    The spec tells you exactly how to pre-process frames on the
    Android side before calling the model.
    """
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    dummy = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    interp.set_tensor(inp_d["index"], dummy)
    interp.invoke()
    print(f"TFLite verified.  Dummy output: {interp.get_tensor(out_d['index'])}")

    print("\n" + "=" * 50)
    print("ANDROID INTEGRATION SPEC")
    print("=" * 50)
    print(f"File      : {os.path.basename(tflite_path)}")
    print(f"Input     : shape={inp_d['shape']}  dtype=float32")
    print(f"Output    : shape={out_d['shape']}  dtype=float32  range=[1,5]")
    print()
    print("Pre-processing (replicate in Android before inference):")
    print(f"  1. Resize camera frame to {IMG_SIZE}x{IMG_SIZE}")
    print( "  2. Convert to RGB float32, divide each pixel by 255.0")
    print(f"  3. Subtract mean {IMAGENET_MEAN} per channel")
    print(f"  4. Divide by std  {IMAGENET_STD}  per channel")
    print( "  5. Layout: NHWC  →  shape (1, 224, 224, 3)")
    print("=" * 50)


# ── Public entry point ───────────────────────────────────────────

def export_pipeline(model: torch.nn.Module, val_loader) -> str:
    """Run the full export pipeline and return the TFLite path.

    Parameters
    ----------
    model      : trained nn.Module with best weights loaded
    val_loader : DataLoader for INT8 calibration data

    Returns
    -------
    Path to the final .tflite file.
    """
    onnx_path  = export_onnx(model)
    onnx_path  = simplify_onnx(onnx_path)
    tf_dir     = export_tf_savedmodel(onnx_path)
    tflite_path = export_tflite(tf_dir, val_loader)
    verify_and_print_spec(tflite_path)
    return tflite_path