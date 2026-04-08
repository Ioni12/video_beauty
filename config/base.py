# config/base.py
# ---------------------------------------------------------------
# All model hyperparameters and constants.
# Nothing here should ever need to change between environments.
# ---------------------------------------------------------------
import torch

# ── Image / training ────────────────────────────────────────────
IMG_SIZE        = 224
BATCH_SIZE      = 32
NUM_WORKERS     = 2
SEED            = 42
EPOCHS          = 50
PATIENCE        = 10

# ── Video inference ─────────────────────────────────────────────
SCORE_THRESHOLD = 3.0   # faces below this score get filtered out
FRAME_SKIP      = 1     # process every N frames (1 = every frame)
MAX_FACES       = 5

# ── Normalisation ───────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Device ──────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")