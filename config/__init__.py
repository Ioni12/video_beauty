# config/__init__.py
# ---------------------------------------------------------------
# Single import point for the entire config layer.
# Any module in the project just does:
#   from config import IMG_SIZE, DEVICE, CHECKPOINT_DIR, ...
# ---------------------------------------------------------------
from config.base import (
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    EPOCHS,
    PATIENCE,
    SCORE_THRESHOLD,
    FRAME_SKIP,
    MAX_FACES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEVICE,
)

from config.env import (
    ENV,
    DATASET_IMG_DIR,
    DATASET_LABELS,
    ALIGNED_DIR,
    CHECKPOINT_DIR,
    EXPORT_DIR,
    FACE_MODEL_PATH,
    make_output_dirs,
    print_env_info,
)