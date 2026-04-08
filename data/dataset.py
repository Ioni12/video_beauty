# data/dataset.py
# ---------------------------------------------------------------
# PyTorch Dataset + transforms + DataLoader factory.
#
# Kept separate from prepare.py because these are used at
# training time (every epoch), while prepare.py is only used
# once before training starts.
# ---------------------------------------------------------------
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from config import (
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ALIGNED_DIR,
)


# ── Transforms ──────────────────────────────────────────────────
# train_transform applies augmentation to make the model more
# robust.  val_transform is deterministic — no random ops.

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Dataset ─────────────────────────────────────────────────────

class FBPDataset(Dataset):
    """Facial Beauty Prediction dataset.

    Parameters
    ----------
    df        : DataFrame with columns ['filename', 'score'].
                Filenames are basenames only (e.g. 'CF001.jpg');
                the full path is built from *img_dir*.
    img_dir   : Directory containing the (aligned) images.
                Defaults to ALIGNED_DIR from config.
    transform : torchvision transform to apply to each image.
                Pass ``train_transform`` for training,
                ``val_transform`` for val/test/inference.
    """

    def __init__(self, df, img_dir: str = ALIGNED_DIR, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        row   = self.df.iloc[i]
        score = torch.tensor(float(row["score"]), dtype=torch.float32)

        path = os.path.join(self.img_dir, row["filename"])
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # If an image fails to load return a grey placeholder so
            # training doesn't crash on a single bad file.
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, score


# ── DataLoader factory ───────────────────────────────────────────

def make_dataloaders(train_df, val_df, test_df):
    """Build and return (train_loader, val_loader, test_loader).

    Uses BATCH_SIZE and NUM_WORKERS from config automatically.

    Parameters
    ----------
    train_df, val_df, test_df : DataFrames from data.prepare.make_splits()

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        FBPDataset(train_df, transform=train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        FBPDataset(val_df, transform=val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        FBPDataset(test_df, transform=val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"DataLoaders ready  →  "
        f"train: {len(train_loader)} batches  |  "
        f"val: {len(val_loader)} batches  |  "
        f"test: {len(test_loader)} batches"
    )
    return train_loader, val_loader, test_loader