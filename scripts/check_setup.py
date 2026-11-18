"""Check that this machine is ready to work on the project (task T1).

Run from the repository root:

    python scripts/check_setup.py

It checks the Python version, the installed packages, the dataset and the
random seed. Every line should show OK. Deep learning packages are optional
until you start the modern-model tasks (T10, T11).
"""

import importlib
import sys

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "seaborn",
    "plotly",
    "wordcloud",
    "nltk",
    "joblib",
    "streamlit",
    "pytest",
    "newsclf",
]
OPTIONAL_PACKAGES = ["torch", "transformers", "sentence_transformers", "lime", "shap"]

failures = 0


def report(ok: bool, label: str, detail: str = "", optional: bool = False) -> None:
    global failures
    if ok:
        status = "OK  "
    elif optional:
        status = "SKIP"
    else:
        status = "FAIL"
        failures += 1
    print(f"[{status}] {label}{(' - ' + detail) if detail else ''}")


# 1. Python version -----------------------------------------------------------
version = sys.version_info
report(version >= (3, 10), "Python version", f"{version.major}.{version.minor}.{version.micro} (need 3.10 or newer)")

# 2. Packages -----------------------------------------------------------------
for name in REQUIRED_PACKAGES:
    try:
        module = importlib.import_module(name)
        report(True, f"import {name}", getattr(module, "__version__", ""))
    except ImportError as exc:
        hint = "run: pip install -r requirements.txt" if name != "newsclf" else "run: pip install -e ."
        report(False, f"import {name}", f"{exc.__class__.__name__}; {hint}")

for name in OPTIONAL_PACKAGES:
    try:
        module = importlib.import_module(name)
        report(True, f"import {name} (deep learning)", getattr(module, "__version__", ""))
    except ImportError:
        report(False, f"import {name} (deep learning)", "not installed yet, needed from T10", optional=True)

# 3. Dataset and seed (only if our own package imports) -----------------------
try:
    import numpy as np
    import pandas as pd

    from newsclf import config
except ImportError:
    print("\nCannot check the dataset and seed until the packages above are installed.")
    sys.exit(1)

if config.RAW_DATA_PATH.exists():
    df = pd.read_csv(config.RAW_DATA_PATH)
    report(len(df) == config.EXPECTED_ROWS, "dataset rows", f"{len(df):,} (expected {config.EXPECTED_ROWS:,})")
    report(
        list(df.columns) == config.EXPECTED_COLUMNS,
        "dataset columns",
        ", ".join(df.columns),
    )
    report(df[config.TARGET_20].nunique() == 20, "newsgroup classes", str(df[config.TARGET_20].nunique()))
    report(df[config.TARGET_6].nunique() == 6, "topic_group classes", str(df[config.TARGET_6].nunique()))
else:
    report(False, "dataset file", f"not found at {config.RAW_DATA_PATH}")

config.set_seed()
first = np.random.rand(3)
config.set_seed()
second = np.random.rand(3)
report(bool(np.allclose(first, second)), "random seed", f"SEED = {config.SEED}, repeated draws match")

for folder in [config.PROCESSED_DIR, config.SPLITS_DIR, config.MODELS_DIR, config.FIGURES_DIR]:
    report(folder.is_dir(), "folder", str(folder.relative_to(config.ROOT_DIR)))

# Summary ---------------------------------------------------------------------
print()
if failures:
    print(f"{failures} check(s) failed. Fix them and run this script again.")
    sys.exit(1)
print("All required checks passed. This machine is ready for the project.")
