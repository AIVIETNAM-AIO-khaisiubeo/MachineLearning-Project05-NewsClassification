"""Basic tests for the project settings. Run with: pytest"""

import numpy as np
import pandas as pd
import pytest

from newsclf import config


def test_set_seed_makes_random_draws_repeatable():
    config.set_seed()
    first = np.random.rand(5)
    config.set_seed()
    second = np.random.rand(5)
    assert np.allclose(first, second)


def test_project_folders_exist():
    for folder in [config.PROCESSED_DIR, config.SPLITS_DIR, config.MODELS_DIR, config.FIGURES_DIR]:
        assert folder.is_dir(), f"missing folder: {folder}"


@pytest.mark.skipif(not config.RAW_DATA_PATH.exists(), reason="dataset not in data/raw yet")
def test_raw_dataset_has_expected_shape():
    df = pd.read_csv(config.RAW_DATA_PATH)
    assert list(df.columns) == config.EXPECTED_COLUMNS
    assert len(df) == config.EXPECTED_ROWS
    assert df[config.TARGET_20].nunique() == 20
    assert df[config.TARGET_6].nunique() == 6
