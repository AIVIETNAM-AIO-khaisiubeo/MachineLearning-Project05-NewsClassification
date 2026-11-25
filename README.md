# Newsgroup Text Classification

Machine Learning Project WiSe 2026/27, Project 5. A system that reads a forum post from the 20 Newsgroups corpus and predicts its discussion group (20 classes) and its topic group (6 classes).

## Team

| Name | Matriculation number |
|---|---|
| Duy Khai | TODO |
| Bao Tue | TODO |

## Setup

Requires Python 3.10 or newer (3.11 recommended).

```bash
git clone https://github.com/AIVIETNAM-AIO-khaisiubeo/MachineLearning-Project05-NewsClassification.git
cd MachineLearning-Project05-NewsClassification

python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt
python scripts/check_setup.py
```

Every line of `check_setup.py` should show `OK` (deep learning packages show `SKIP` until they are needed). For the modern models, also run `pip install -r requirements-dl.txt`.

## Running the project

Run every command from the repository root.

**1. Build the cleaned data set (T5).** Applies the rules in `reports/cleaning_decisions.md` and writes `data/processed/clean.csv`, which every model uses:

```bash
python scripts/make_dataset.py
pytest tests/test_clean.py      # optional: check the cleaning rules
```

TODO (T26): how to run training and how to start the UI.

## Project structure

```
app/                 Streamlit user interface
data/raw/            newsgroup_posts.csv (original data, never edited)
data/processed/      cleaned data (generated, not in git)
data/splits/         train/validation/test post_id lists (in git, shared by all models)
models/              trained models (generated, not in git)
notebooks/           exploration and experiments, named 01_eda.ipynb, 02_baseline.ipynb, ...
reports/figures/     charts for the report
scripts/             command-line scripts
src/newsclf/         shared Python code (config, cleaning, features, models, evaluation)
tests/               pytest tests
```

## Working rules

- One branch per task (for example `t2-data-audit`), merged through a pull request reviewed by the other team member.
- Random seed 42 everywhere: call `set_seed()` and pass `random_state=SEED` (both from `newsclf.config`).
- The test set is used only for the final evaluation. TF-IDF is fitted on the training data only.
