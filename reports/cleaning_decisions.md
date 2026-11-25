# T4 - Cleaning decisions

Evidence: `notebooks/01_data_audit.ipynb` (sections 3 to 9). These rules are implemented in T5 (`src/newsclf/clean.py`, `scripts/make_dataset.py`) and the result is written to `data/processed/clean.csv`. Every model, for both the 20-class and the 6-class task, uses this same cleaned file.

Guiding idea: **light cleaning**. Remove what has no content or would let the model cheat (empty posts, copies, author addresses, quoted text from other posts, encoded files), and keep everything that carries topic information (numbers, code, long posts).

## 1. Decision per issue

| # | Issue | What the audit found | Decision |
|---|---|---|---|
| 1 | Empty posts | 515 posts (2.7%), all missing values; 0.6% to 5.4% per newsgroup | **Drop** |
| 2 | Duplicates | 69 exact extra copies, 78 when case and spacing are ignored | **Keep one copy** of each normalised text (lower case, single spaces) |
| 3 | Same text under different newsgroups | 14 texts: junk (`--`, `...`) and cross-posts | **Drop all copies**: there is no single correct label |
| 4 | Very short posts | 187 posts with fewer than 5 words; 1-2 word posts carry no topic ("Yep.", "Hello,"), 3-4 word posts sometimes do ("Squeaky BMW riders.") | **Drop posts with fewer than 3 words**, counted after cleaning |
| 5 | Very long posts | 150 posts with more than 2,000 words: FAQs and long documents, full of topic words | **Keep**, no upper limit. TF-IDF normalises every vector to length 1; transformers truncate to their maximum length (T10, T11) |
| 6 | E-mail addresses | 2,798 posts (15%); 74% of the repeated addresses stay in one newsgroup, so they identify the author | **Replace** with the token `emailaddress` |
| 7 | URLs | 121 posts | **Replace** with the token `urladdress` (after the e-mail addresses) |
| 8 | Numbers | Often topic words (486, 610, 1914) | **Keep** |
| 9 | Code snippets | Concentrated in the comp groups (all X toolkit code in comp.windows.x) | **Keep** |
| 10 | Reply quotes | 181 posts with lines starting with `>` | **Remove these lines** |
| 11 | Encoded files | 31 posts, 15,397 lines of random characters (uuencode, PGP keys) | **Remove these lines** |
| 12 | Separator lines, ASCII art | 2,170 posts; no words | **Remove these lines** (lines without any letter or digit) |
| 13 | Upper/lower case, punctuation | The TF-IDF tokenizer lower-cases and drops punctuation itself | **No change** in the cleaned text |
| 14 | Stop words, stemming | Not a data problem | **Not part of cleaning**: stop words are a TF-IDF setting to try in tuning (T13); no stemming, so the text stays readable for the UI and the transformers |
| 15 | Non-ASCII characters, typos | 25 posts with non-ASCII characters; typos rare and not systematic | **No action** |

## 2. Order of the steps in T5

1. Drop empty posts (missing, or only spaces and line breaks).
2. In each post, remove the reply-quote lines, the encoded-file lines and the separator lines (use the three line rules from section 6.3 of the notebook).
3. Replace e-mail addresses with `emailaddress`, then URLs with `urladdress`. E-mail addresses go first because an address such as `name@ftp.example.edu` contains `ftp.` and would otherwise be cut apart by the URL rule.
4. Turn every run of spaces and line breaks into a single space and strip both ends.
5. Count the words again and drop posts with fewer than 3 words. This also drops posts that consisted only of quotes or encoded lines.
6. Build the normalised text (lower case). Drop all copies of texts that appear under more than one newsgroup, then keep the first copy of every remaining text.
7. Save `post_id`, the cleaned `text`, `newsgroup` and `topic_group` to `data/processed/clean.csv`.

Expected effect: roughly 3 to 4% of the raw file is removed (exact number in notebook section 9.2), most of it empty posts. Every newsgroup keeps more than 93% of its posts, and the topic-group shares hardly change (notebook section 9.2).

## 3. Rules for the later tasks (not part of the cleaning)

- **Split after cleaning (T6):** stratified by `newsgroup` (this also stratifies `topic_group`, because each newsgroup belongs to exactly one topic group), fixed seed 42, the same split for every model.
- **TF-IDF on the training data only:** apart from the duplicate check, which must happen before the split on purpose, the cleaning rules look at one post at a time and learn nothing from the data set, so applying them before the split is safe. TF-IDF learns the vocabulary and the IDF weights from the data, so `TfidfVectorizer` and the classifier go into one scikit-learn `Pipeline` that is fitted on the training data (or the training folds in cross-validation) only (T8, T9, T13).
- **Unbalanced topic groups:** report macro-F1 next to accuracy and try `class_weight="balanced"` for the 6-class task, because `forsale` is only about 5% of the posts.
- **Close newsgroups:** expect confusions inside comp, politics and religion (20 classes), and between sci.electronics and comp and between talk.politics.misc and religion (6 classes). Check them in the error analysis (T15).
- **Near-duplicates** such as the three versions of the JPEG FAQ are not removed by the exact duplicate check; bonus task B5 measures their effect.
- **Dated vocabulary:** mention the 1990s text as a limitation in the report and test the UI with modern posts (T21, B10).
