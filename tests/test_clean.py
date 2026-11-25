"""Tests for the cleaning rules in newsclf.clean (tasks T5 and B7). Run with: pytest"""

import numpy as np
import pandas as pd
import pytest

from newsclf import config
from newsclf.clean import (
    EMAIL_PATTERN,
    EMAIL_TOKEN,
    MIN_WORDS,
    REASON_CONFLICT,
    REASON_DUPLICATE,
    REASON_EMPTY,
    REASON_TOO_SHORT,
    URL_TOKEN,
    clean_posts,
    clean_text,
    is_encoded_line,
    is_quote_line,
    is_separator_line,
)

UUENCODE_LINE = "M1865%22DM75U=75U4)\"0IV=G9V=G9V=G9V=G9V=G9V=G9V=G9V=G9V=G9V="
BASE64_LINE = "mQCNAiuFrmUAAAEEAMmIZyb2dvb1R6SMXcHPgEtKKwGNtWJvUIKzKFZk5f5ExIlr0O"


# ---------------------------------------------------------------------------
# Line rules
# ---------------------------------------------------------------------------
def test_quote_rule():
    assert is_quote_line("> quoted text")
    assert is_quote_line("   >> quoted with spaces in front")
    assert not is_quote_line("a > b is true")


def test_encoded_rule_finds_uuencode_and_base64():
    assert is_encoded_line(UUENCODE_LINE)
    assert is_encoded_line(BASE64_LINE)
    assert is_encoded_line(BASE64_LINE[:60] + "==")  # base64 padding at the end


def test_encoded_rule_keeps_normal_lines():
    assert not is_encoded_line("My 486DX2 has 8 MB of RAM and runs Windows 3.1 without problems.")
    assert not is_encoded_line("XtAppMainLoop(app);")


def test_separator_rule():
    assert is_separator_line("----------------")
    assert is_separator_line("  *** *** ***  ")
    assert is_separator_line("________________")  # TF-IDF would keep this as a "word" (notebook 6.4)
    assert not is_separator_line("...")  # shorter than 4 characters
    assert not is_separator_line("-- Mike")  # has letters


# ---------------------------------------------------------------------------
# clean_text: one post
# ---------------------------------------------------------------------------
def test_noise_lines_are_removed():
    text = "My answer:\n> the quoted question\n" + UUENCODE_LINE + "\n----------\nthe end"
    assert clean_text(text) == "My answer: the end"


def test_email_is_replaced_with_one_token():
    assert clean_text("Mail jane.doe+news@cs.example.edu today") == f"Mail {EMAIL_TOKEN} today"


def test_email_goes_before_url():
    # name@ftp.example.edu contains "ftp." but must become ONE e-mail token
    assert clean_text("write to anon@ftp.example.edu") == f"write to {EMAIL_TOKEN}"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("get it from ftp.uu.net.", f"get it from {URL_TOKEN}."),
        ("at ftp.cica.indiana.edu:/pub/pc/win3/drivers, file x", f"at {URL_TOKEN}, file x"),
        ("see (http://info.cern.ch/hypertext/WWW.html)", f"see ({URL_TOKEN})"),
        ("www.example.com is new", f"{URL_TOKEN} is new"),
        ("available by anonymous ftp. Ask me", "available by anonymous ftp. Ask me"),  # not a host
    ],
)
def test_urls(text, expected):
    assert clean_text(text) == expected


def test_numbers_code_case_and_punctuation_are_kept():
    text = "I have a 486DX2 :-)\n    XtAppMainLoop(app);"
    assert clean_text(text) == "I have a 486DX2 :-) XtAppMainLoop(app);"


def test_whitespace_is_normalised():
    assert clean_text("  Hello,\n\n\tworld  \r\n  again ") == "Hello, world again"


@pytest.mark.parametrize("missing", [None, np.nan, "", "   \n  "])
def test_missing_or_blank_text_gives_empty_string(missing):
    assert clean_text(missing) == ""


def test_clean_text_is_idempotent():
    text = "Hi,\n> quote\nmail me@example.com or ftp.uu.net\n-----\nbye"
    once = clean_text(text)
    assert clean_text(once) == once


# ---------------------------------------------------------------------------
# clean_posts: the data set
# ---------------------------------------------------------------------------
def make_posts(rows):
    """A small data frame with the four columns of the raw file."""
    return pd.DataFrame(rows, columns=config.EXPECTED_COLUMNS)


def test_clean_posts_drops_with_the_right_reason():
    raw = make_posts([
        (1, "A post about the new hockey season.", "rec.sport.hockey", "recreation"),
        (2, None, "sci.med", "science"),  # empty
        (3, "Yep.", "rec.autos", "recreation"),  # too short
        (4, "> only a quote of\n> someone else here", "sci.med", "science"),  # too short AFTER cleaning
        (5, "Cross posted question about a 486 CPU", "sci.electronics", "science"),  # conflict
        (6, "Cross posted question about a 486 CPU", "comp.sys.ibm.pc.hardware", "comp"),  # conflict
        (7, "a post about the NEW hockey   season.", "rec.sport.hockey", "recreation"),  # duplicate of 1
    ])
    clean, dropped = clean_posts(raw)

    assert clean[config.ID_COL].tolist() == [1]
    reasons = dict(zip(dropped[config.ID_COL], dropped["reason"]))
    assert reasons == {
        2: REASON_EMPTY,
        3: REASON_TOO_SHORT,
        4: REASON_TOO_SHORT,
        5: REASON_CONFLICT,
        6: REASON_CONFLICT,
        7: REASON_DUPLICATE,
    }


def test_duplicates_keep_the_lowest_post_id_and_use_the_cleaned_text():
    # The two posts differ only in the e-mail address, so after cleaning they are copies
    raw = make_posts([
        (20, "For sale: Amiga 500, mail bob@x.edu", "misc.forsale", "forsale"),
        (10, "For sale: Amiga 500, mail alice@y.edu", "misc.forsale", "forsale"),
    ])
    clean, dropped = clean_posts(raw)
    assert clean[config.ID_COL].tolist() == [10]
    assert dropped[config.ID_COL].tolist() == [20]


def test_clean_posts_output_format_and_input_unchanged():
    raw = make_posts([(2, "Second post with\n\nseveral words", "sci.space", "science"),
                      (1, "First post, mail me@example.com", "sci.space", "science")])
    raw_before = raw.copy()
    clean, _ = clean_posts(raw)

    assert list(clean.columns) == config.EXPECTED_COLUMNS
    assert clean[config.ID_COL].tolist() == [1, 2]  # sorted by post_id
    assert clean[config.TEXT_COL].tolist() == [f"First post, mail {EMAIL_TOKEN}", "Second post with several words"]
    pd.testing.assert_frame_equal(raw, raw_before)


# ---------------------------------------------------------------------------
# The real data set (skipped if the CSV is missing)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_data():
    if not config.RAW_DATA_PATH.exists():
        pytest.skip("dataset not in data/raw")
    raw = pd.read_csv(config.RAW_DATA_PATH, keep_default_na=False)
    clean, dropped = clean_posts(raw)
    return raw, clean, dropped


def test_real_data_rules_hold(real_data):
    raw, clean, dropped = real_data
    texts = clean[config.TEXT_COL]
    keys = texts.str.lower()

    assert len(clean) + len(dropped) == len(raw)
    assert clean[config.ID_COL].is_unique
    assert not keys.duplicated().any()
    assert (texts.str.split().str.len() >= MIN_WORDS).all()
    assert not texts.str.contains(EMAIL_PATTERN).any()
    assert not texts.str.contains("\n").any()
    assert clean[config.TARGET_20].nunique() == 20
    assert clean[config.TARGET_6].nunique() == 6


def test_real_data_removes_little_and_keeps_every_newsgroup(real_data):
    raw, clean, dropped = real_data
    # cleaning_decisions.md: roughly 3 to 4% removed, every newsgroup keeps more than 93%
    assert 0.03 <= len(dropped) / len(raw) <= 0.04
    kept_share = clean[config.TARGET_20].value_counts() / raw[config.TARGET_20].value_counts()
    assert (kept_share > 0.93).all()


def test_real_data_cleaning_twice_changes_nothing(real_data):
    # Important for the UI: cleaning an already cleaned text must not change it
    _, clean, _ = real_data
    texts = clean[config.TEXT_COL]
    assert (texts.apply(clean_text) == texts).all()
