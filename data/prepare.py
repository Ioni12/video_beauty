# data/prepare.py
# ---------------------------------------------------------------
# Loads the SCUT-FBP5500 labels file, aligns faces, and produces
# train / val / test DataFrames ready for the Dataset class.
#
# This is the slow one-time step you run before training.
# Aligned images are cached to disk so you never re-run it unless
# you delete the aligned/ folder.
# ---------------------------------------------------------------
import os
import cv2
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from config import (
    DATASET_IMG_DIR,
    DATASET_LABELS,
    ALIGNED_DIR,
    SEED,
    make_output_dirs,
)
from face import align_face


def load_labels() -> pd.DataFrame:
    """Read the labels file and return the full DataFrame.

    The labels file is a space-separated file with two columns:
        filename   score
    e.g.:  CF001.jpg  3.42

    Returns
    -------
    pd.DataFrame with columns ['filename', 'score']
    """
    df = pd.read_csv(
        DATASET_LABELS,
        sep=" ",
        header=None,
        names=["filename", "score"],
    )
    print(f"Total labels loaded: {len(df)}")
    return df


def filter_caucasian(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Caucasian-female (CF) and Caucasian-male (CM) images.

    The SCUT-FBP5500 dataset uses filename prefixes to identify
    demographic groups.  Training on a single group avoids the
    model learning demographic features instead of beauty features.
    """
    mask = df["filename"].str.startswith(("CF", "CM"))
    result = df[mask].reset_index(drop=True)
    print(f"Caucasian images (CF + CM): {len(result)}")
    return result


def align_and_cache(df: pd.DataFrame) -> pd.DataFrame:
    """Align every face image and save it to ALIGNED_DIR.

    Skips images that are already aligned (checks if the output
    file exists) so re-running is fast.

    Parameters
    ----------
    df : DataFrame with a 'filename' column (just the basename,
         e.g. 'CF001.jpg')

    Returns
    -------
    A cleaned DataFrame containing only rows whose source image
    was found and successfully aligned.
    """
    make_output_dirs()

    valid_rows = []
    skipped    = 0

    for _, row in df.iterrows():
        src = str(Path(DATASET_IMG_DIR) / row["filename"])
        dst = os.path.join(ALIGNED_DIR, row["filename"])

        # Source image missing — skip silently and count
        if not os.path.exists(src):
            skipped += 1
            continue

        # Already aligned — no work needed
        if not os.path.exists(dst):
            img = cv2.imread(src)
            if img is None:
                skipped += 1
                continue
            cv2.imwrite(dst, align_face(img))

        valid_rows.append(row)

    result = pd.DataFrame(valid_rows).reset_index(drop=True)
    print(f"Aligned: {len(result)} images  |  Skipped: {skipped}")
    return result


def make_splits(df: pd.DataFrame):
    """Stratified train / val / test split (80 / 10 / 10).

    Stratification is done on score bins so each split has a
    similar score distribution — important for a regression task
    where a naive random split can skew the score range.

    Parameters
    ----------
    df : cleaned, aligned DataFrame with 'filename' and 'score'

    Returns
    -------
    train_df, val_df, test_df
    """
    df = df.copy()
    df["score_bin"] = pd.cut(df["score"], bins=5, labels=False)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df["score_bin"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["score_bin"],
    )

    print(f"Split  →  Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def prepare_data():
    """Run the full preparation pipeline and return splits.

    Call this once before training:

        from data.prepare import prepare_data
        train_df, val_df, test_df = prepare_data()

    Returns
    -------
    train_df, val_df, test_df
    """
    df       = load_labels()
    df       = filter_caucasian(df)
    df_clean = align_and_cache(df)
    return make_splits(df_clean)