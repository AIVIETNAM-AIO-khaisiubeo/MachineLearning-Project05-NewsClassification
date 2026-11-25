"""Project-wide settings: random seed, file paths and column names.

Import from here instead of hard-coding values in notebooks or scripts:

    from newsclf.config import SEED, RAW_DATA_PATH, set_seed

    set_seed()                      # call once at the top of every notebook/script
    LogisticRegression(random_state=SEED)
    train_test_split(..., random_state=SEED, stratify=y)

scikit-learn does not read a global seed, so always pass random_state=SEED
to splitters, models and cross-validation objects yourself.
"""

import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Paths (all relative to the repository root, so they work on every machine)
# src/newsclf/config.py -> parents[2] is the repository root
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "newsgroup_posts.csv"  # original file, never edit it
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned data (T5), can be regenerated
CLEAN_DATA_PATH = PROCESSED_DIR / "clean.csv"  # output of scripts/make_dataset.py, input of every model
SPLITS_DIR = DATA_DIR / "splits"  # train/val/test post_id lists (T6), committed to git

MODELS_DIR = ROOT_DIR / "models"  # trained models (T12), not committed to git
REPORTS_DIR = ROOT_DIR / "reports"  # written decisions and summaries for the report
FIGURES_DIR = REPORTS_DIR / "figures"  # charts for the report

# ---------------------------------------------------------------------------
# Dataset columns
# ---------------------------------------------------------------------------
ID_COL = "post_id"
TEXT_COL = "text"
TARGET_20 = "newsgroup"  # 20 fine-grained classes
TARGET_6 = "topic_group"  # 6 coarse classes
EXPECTED_COLUMNS = [ID_COL, TEXT_COL, TARGET_20, TARGET_6]
EXPECTED_ROWS = 18_846  # rows in the raw file, before cleaning


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if installed) PyTorch so results are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # PyTorch is only needed for the deep learning tasks
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
