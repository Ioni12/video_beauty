# config/env.py
# ---------------------------------------------------------------
# Detects which environment the code is running in and sets all
# file paths accordingly.  Every other module imports paths from
# here — no hardcoded paths anywhere else in the project.
# ---------------------------------------------------------------
import os


def detect_env() -> str:
    """Return 'kaggle', 'colab', or 'local'."""
    if os.path.exists("/kaggle/input"):
        return "kaggle"
    if os.path.exists("/content"):
        return "colab"
    return "local"


ENV = detect_env()

# ── Dataset source paths (read-only, provided by the environment) ──
if ENV == "kaggle":
    DATASET_IMG_DIR = "/kaggle/input/scut-fbp5500-v2-facial-beauty-scores/Images/Images"
    DATASET_LABELS  = "/kaggle/input/scut-fbp5500-v2-facial-beauty-scores/labels.txt"
    _WORK_DIR       = "/kaggle/working"

elif ENV == "colab":
    # On Colab you download the dataset yourself (see notebooks/colab_demo.ipynb).
    # By default we expect it unzipped at /content/data/.
    DATASET_IMG_DIR = "/content/data/Images/Images"
    DATASET_LABELS  = "/content/data/labels.txt"
    _WORK_DIR       = "/content"

else:  # local machine / any other environment
    # Place the dataset in a local `data/` folder next to the project,
    # or override these with environment variables:
    #   export VB_DATASET_IMG_DIR=/path/to/images
    #   export VB_DATASET_LABELS=/path/to/labels.txt
    #   export VB_WORK_DIR=/path/to/output
    DATASET_IMG_DIR = os.environ.get("VB_DATASET_IMG_DIR", "data/Images/Images")
    DATASET_LABELS  = os.environ.get("VB_DATASET_LABELS",  "data/labels.txt")
    _WORK_DIR       = os.environ.get("VB_WORK_DIR",        "output")

# ── Output / working paths (written by the project) ─────────────
ALIGNED_DIR     = os.path.join(_WORK_DIR, "aligned")
CHECKPOINT_DIR  = os.path.join(_WORK_DIR, "checkpoints")
EXPORT_DIR      = os.path.join(_WORK_DIR, "exports")
FACE_MODEL_PATH = "face_landmarker.task"   # downloaded separately; same location everywhere


def make_output_dirs() -> None:
    """Create all output directories if they don't exist yet."""
    for d in (ALIGNED_DIR, CHECKPOINT_DIR, EXPORT_DIR):
        os.makedirs(d, exist_ok=True)


def print_env_info() -> None:
    """Print a summary of the detected environment and resolved paths."""
    print(f"Environment  : {ENV}")
    print(f"Dataset imgs : {DATASET_IMG_DIR}")
    print(f"Labels file  : {DATASET_LABELS}")
    print(f"Aligned dir  : {ALIGNED_DIR}")
    print(f"Checkpoints  : {CHECKPOINT_DIR}")
    print(f"Exports      : {EXPORT_DIR}")