"""Text cleaning for the newsgroup posts (task T5).

The rules come from T4 (reports/cleaning_decisions.md); the evidence for each
rule is in notebooks/01_data_audit.ipynb. Guiding idea: light cleaning. Remove
what has no content or would let the model cheat, keep everything that carries
topic information (numbers, code, long posts).

There are two entry points:

    from newsclf.clean import clean_text, clean_posts

    clean_text(text)
        Cleans ONE post (T4 steps 2 to 4). Use it for the training data and for
        every text a user pastes into the UI, so the models always see the same
        kind of text.

    clean, dropped = clean_posts(raw)
        Cleans the whole data set (T4 steps 1 to 6): cleans every text, then drops
        empty posts, posts that are too short, texts with more than one newsgroup
        and duplicate texts. Returns the kept posts and a table of the dropped
        posts with the reason. scripts/make_dataset.py calls it and saves the
        result to data/processed/clean.csv.
"""

import re

import pandas as pd

from newsclf.config import EXPECTED_COLUMNS, ID_COL, TARGET_6, TARGET_20, TEXT_COL

# ---------------------------------------------------------------------------
# Settings (T4 decisions 4, 6 and 7)
# ---------------------------------------------------------------------------
# A placeholder must be ONE word, so the TF-IDF tokenizer keeps it as one token
# (notebook 6.4: "[e-mail]" would become the token "mail").
EMAIL_TOKEN = "emailaddress"
URL_TOKEN = "urladdress"

# Posts with fewer words than this (counted after cleaning) are dropped (notebook 5.1)
MIN_WORDS = 3

# Same pattern as in the audit notebook (6.1): name@domain.ending, for example jdoe@cs.example.edu
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

# A URL either starts with a scheme (http://, ftp://, ...) or is a bare host name that
# starts with www. or ftp. (in this 1990s data almost all URLs are FTP hosts such as
# ftp.uu.net:/pub/X11). The last character of a match may not be punctuation, so in
# "see ftp.uu.net." only the host is replaced and the full stop stays.
# A host needs at least one more dot-part, so "via anonymous ftp." at the end of a
# sentence is NOT treated as a URL.
URL_PATTERN = re.compile(
    r"""
      \b(?:https?|ftp|file|gopher|telnet|wais)://\S*[^\s.,;:!?'")\]}>]   # http://host/path
    | \b(?:www|ftp)\.[\w-]+(?:\.[\w-]+)+                                # www.host.edu or ftp.host.edu
      (?:[/:~]\S*[^\s.,;:!?'")\]}>])?                                   #   optional path, e.g. :/pub/win3
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Reasons in the table of dropped posts, in the order in which the steps run
REASON_EMPTY = "empty"
REASON_TOO_SHORT = "too short after cleaning"
REASON_CONFLICT = "same text under different newsgroups"
REASON_DUPLICATE = "duplicate"
DROP_REASONS = [REASON_EMPTY, REASON_TOO_SHORT, REASON_CONFLICT, REASON_DUPLICATE]


# ---------------------------------------------------------------------------
# Line rules (T4 decisions 10 to 12). Same rules as in notebook section 6.3:
# each function looks at one line and returns True or False.
# ---------------------------------------------------------------------------
def is_quote_line(line: str) -> bool:
    """A reply quote: the first visible character is '>'."""
    return line.strip().startswith(">")


def is_encoded_line(line: str) -> bool:
    """One line of a file that was put into the post as text.

    - uuencode: starts with 'M', 56 to 62 characters, no spaces and no lower-case letters
    - base64 (also PGP keys): 60 to 76 letters, digits, '+' or '/', maybe '=' at the end
    """
    line = line.strip()
    uuencode = (
        line.startswith("M")
        and 56 <= len(line) <= 62
        and " " not in line
        and not any(ch.islower() for ch in line)
    )
    body = line.rstrip("=")  # the '=' at the end of a base64 line is only padding
    base64 = 60 <= len(body) <= 76 and all(ch.isalnum() or ch in "+/" for ch in body)
    return uuencode or base64


def is_separator_line(line: str) -> bool:
    """At least 4 characters and no letter or digit, such as ------ or ****** (also ASCII art)."""
    line = line.strip()
    return len(line) >= 4 and not any(ch.isalnum() for ch in line)


def is_noise_line(line: str) -> bool:
    """True if the line is a reply quote, part of an encoded file or a separator."""
    return is_quote_line(line) or is_encoded_line(line) or is_separator_line(line)


# ---------------------------------------------------------------------------
# Cleaning one post (T4 steps 2 to 4)
# ---------------------------------------------------------------------------
def remove_noise_lines(text: str) -> str:
    """Step 2: remove the reply-quote, encoded-file and separator lines."""
    kept_lines = [line for line in text.splitlines() if not is_noise_line(line)]
    return "\n".join(kept_lines)


def replace_addresses(text: str) -> str:
    """Step 3: replace e-mail addresses with EMAIL_TOKEN, then URLs with URL_TOKEN.

    E-mail addresses go first: name@ftp.example.edu contains "ftp." and would
    otherwise be cut apart by the URL rule.
    """
    text = EMAIL_PATTERN.sub(EMAIL_TOKEN, text)
    return URL_PATTERN.sub(URL_TOKEN, text)


def normalise_whitespace(text: str) -> str:
    """Step 4: every run of spaces, tabs and line breaks becomes one space; strip both ends."""
    return " ".join(text.split())  # split() without an argument splits at any whitespace


def clean_text(text) -> str:
    """Clean one post: remove noise lines, replace addresses, normalise whitespace.

    Upper/lower case, punctuation, numbers and code are left as they are (T4
    decisions 8, 9 and 13). A missing text (None or NaN) gives an empty string.
    """
    if not isinstance(text, str):  # None or NaN from an empty CSV field
        return ""
    text = remove_noise_lines(text)
    text = replace_addresses(text)
    return normalise_whitespace(text)


def count_words(text: str) -> int:
    """Number of words, split at spaces and line breaks (as in the audit notebook)."""
    return len(text.split())


def duplicate_key(text: str) -> str:
    """The form in which two texts are compared: lower case, single spaces (T4 decision 2)."""
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Cleaning the data set (T4 steps 1 to 6)
# ---------------------------------------------------------------------------
def clean_posts(raw: pd.DataFrame, min_words: int = MIN_WORDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean every post and drop the posts the model should not see.

    Steps (reports/cleaning_decisions.md, section 2):
      1. drop empty posts (missing, or only spaces and line breaks);
      2.-4. clean each text with clean_text();
      5. drop posts with fewer than `min_words` words, counted after cleaning;
      6. drop ALL copies of a text that appears under more than one newsgroup,
         then keep only the first copy (lowest post_id) of every other text.
         Texts are compared in lower case with single spaces.

    The input is not changed. Returns two tables:
      clean   - the kept posts with the columns post_id, text (cleaned), newsgroup,
                topic_group, sorted by post_id;
      dropped - the removed posts with the columns post_id, newsgroup, topic_group,
                reason and text (the cleaned text, empty for empty posts).
    """
    # Sort by post_id so that "the first copy" of a duplicate is always the same post
    posts = raw[EXPECTED_COLUMNS].sort_values(ID_COL).reset_index(drop=True)
    original_text = posts[TEXT_COL].fillna("").astype(str)

    # reason stays "" for a post we keep; every step only looks at the posts still kept
    reason = pd.Series("", index=posts.index, dtype=object)

    # Step 1: empty posts
    reason[original_text.str.strip() == ""] = REASON_EMPTY

    # Steps 2 to 4: clean every text
    posts[TEXT_COL] = original_text.apply(clean_text)

    # Step 5: too short after cleaning (this also catches posts that were only quotes)
    word_counts = posts[TEXT_COL].apply(count_words)
    reason[(reason == "") & (word_counts < min_words)] = REASON_TOO_SHORT

    # Step 6a: the same text under more than one newsgroup has no single correct label
    key = posts[TEXT_COL].apply(duplicate_key)
    kept = reason == ""
    newsgroups_per_text = posts[kept].groupby(key[kept])[TARGET_20].nunique()
    conflicting_texts = newsgroups_per_text[newsgroups_per_text > 1].index
    reason[kept & key.isin(conflicting_texts)] = REASON_CONFLICT

    # Step 6b: keep the first copy of every remaining text
    kept = reason == ""
    is_extra_copy = key[kept].duplicated(keep="first")  # True for the 2nd, 3rd, ... copy
    extra_copies = is_extra_copy[is_extra_copy].index  # row numbers of those copies
    reason[extra_copies] = REASON_DUPLICATE

    kept = reason == ""
    clean = posts.loc[kept, EXPECTED_COLUMNS].reset_index(drop=True)

    dropped = posts.loc[~kept, [ID_COL, TARGET_20, TARGET_6]].copy()
    dropped["reason"] = reason[~kept]
    dropped[TEXT_COL] = posts.loc[~kept, TEXT_COL]
    return clean, dropped.reset_index(drop=True)
