# Preprocessing decisions (v2, checklist-compliant)

Single record of every preprocessing decision. Constants live in
`src/preprocessing/config.py`; behaviour is locked by `tests/` and proven by
`python -m src.preprocessing.verify_pipeline`.

## 1. Data and labels
| Dataset | Rows | Role |
|---|---|---|
| HumAID (official splits, never re-shuffled) | 53,531 / 7,793 / 15,160 | trains the 10-class classifier |
| Kaggle "NLP with Disaster Tweets" (labeled file only) | 7,613 raw -> 6,753 after dedup | trains the binary pre-filter |
| CrisisLexT26 (26 labeled event files) | 27,933 | external evaluation only, never fitted on |

* **HumAID has 10 classes, not 11.** The public card lists `dont_know_cant_judge`, but it never occurs in the supplied files (documented by `describe_label_schema`). Checklist item B7 (compare it with `not_humanitarian`) is therefore N/A.
* Largest/smallest training class = 14,891 / 250 (`missing_or_found_people`), imbalance ratio 59.6.
* HumAID CSVs have no event column; event-level analysis is not possible from the repository data (limitation).

## 2. Cleaning (one implementation: `src/preprocessing/text_cleaning.py`)
Order matters and is identical for every dataset:
1. unescape HTML entities **until stable** (fixes `&amp;amp;`)
2. remove known encoding damage (`\x89Û?`, `åÊ`, `Ì¢`, `ã¢`; occurs only in Kaggle)
3. collapse whitespace, strip **chained** `RT @a: RT @b:` prefixes
4. TF-IDF/BiLSTM path only: URL -> `URLTOKEN`, @mention -> `USERTOKEN`, drop `#` keep word, lower-case
5. Transformer path keeps casing, hashtags, URLs, mentions, emojis, informal spelling (BERTweet normalises URLs/mentions/emojis itself; needs `emoji==0.6.0`).

Deliberately **not** changed: elongated words, apostrophe-less contractions ("Im", "dont" - a source quirk present in all datasets), punctuation, emojis.

## 3. Tokenisation
* TF-IDF token pattern keeps words (with inner apostrophes), runs of `!`/`?` and emojis (the old pattern silently dropped emojis and punctuation).
* BiLSTM tokenizer splits punctuation/emojis off words, keeps `!!!` as one token and keeps `don't` whole.
* Hashtags (42% CamelCase) stay one token for TF-IDF; BERTweet's BPE splits them.

## 4. Models' inputs
| Path | Fitted on | Length |
|---|---|---|
| TF-IDF (50,000 features, 1-2 grams, min_df 2) | HumAID train only | n/a |
| BiLSTM vocabulary (min frequency 2, `<PAD>`/`<UNK>`) | HumAID train only | 64 (train p99 = 57; truncates 0.13% of train) |
| BERTweet tokenizer | downloaded, saved as `bertweet_tokenizer_humaid_v1` | **128** (model limit; 256 would crash). Exact truncation counts: `bertweet_preprocessing_report.json` |
| Numeric features (8, text-derived only) + scaler | HumAID train only | n/a |

## 5. Kaggle pre-filter pipeline
* Independent vectorizer (`tfidf_kaggle_prefilter_v1.joblib`), own seeded stratified 80/10/10 split (seed 42).
* **Deduplicated before splitting** using `normalize_for_dedup` (retweets/URLs/mentions/case/punctuation collapsed): 110 exact duplicates, 777 near-duplicates, **83 conflicting-label groups (306 rows) dropped entirely**. A naive split leaked 111 validation and 112 test rows into train; the pipeline now fails if any shared text exists.
* 669 rows with encoding damage repaired; 33 rows with other non-ASCII characters are listed for manual review in the report.
* `keyword` NOT used (shortcut risk: 4 keywords are 100% one class, 15% extreme), `location` NOT used (33% missing, 84% unique free text).
* Only the labeled `train` file is read (enforced by `verify_pipeline` C7).

## 6. CrisisLexT26 (frozen transformations only)
* Transforms with the saved HumAID TF-IDF vectorizer, vocabulary, label map and BERTweet tokenizer. Nothing is fitted (enforced by test + `verify_pipeline` D1).
* **Multilingual:** only ~71% of tweets are English; 8 of 26 events are under 50% English. `language_guess` / `is_english_guess` are added and evaluation flags require English.
* 11-class mapping uses `Information Type` (`good` vs `rough` flagged). CrisisLexT26 has **no counterpart** for `displaced_people_and_evacuations`, `missing_or_found_people`, `requests_or_urgent_needs` -> class-level evaluation covers 7 classes only.
* `Informativeness` ("Not related" = 0, "Related ..." = 1) is a separate label used to evaluate the Kaggle-trained pre-filter; "Not related" tweets are deliberately **not** mapped to `not_humanitarian`.
* Duplicates (retweets/copies) are flagged (`is_duplicate_text`), not dropped.
* Evaluation rows: `use_for_class_eval` (mapped, English, non-duplicate) and `use_for_prefilter_eval`.
* OOV against the frozen vocabulary: 19.5% (HumAID test: 3.0%) - an expected era/language shift.

## 7. Artifacts (all names in `config.py`, versioned and dataset-tagged)
`tfidf_humaid_v1.joblib`, `bilstm_vocabulary_humaid_v1.joblib/.json`, `numerical_scaler_humaid_v1.joblib`, `label_mapping_humaid_v1.json`, `bertweet_tokenizer_humaid_v1/`, `tfidf_kaggle_prefilter_v1.joblib`. `docs/artifact_manifest.json` records hashes and library versions. Modeling code must load artifacts by these constants, never "the latest file".

## 8. Reproduce
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.preprocessing.run_preprocessing
python -m src.preprocessing.run_sequences
python -m src.preprocessing.run_bertweet
python -m src.preprocessing.run_kaggle_preprocessing
python -m src.preprocessing.run_crisislex_preprocessing
python -m src.preprocessing.run_crisislex_bertweet
python -m src.analysis.run_audits
python -m src.preprocessing.verify_pipeline
```
Re-run everything after any change to cleaning code: `verify_pipeline` B1/B2 fail on stale artifacts by design.

## 9. Known distribution differences (modeling-time rules)
* **URLs:** HumAID text contains essentially no URLs (4 of 53,531 training rows) while 52% of Kaggle and 51% of CrisisLexT26 tweets do. The `urltoken` feature therefore has a weight learned from only 4 HumAID tweets. **Rule:** when a HumAID-trained model predicts on another dataset, mask URL features (TF-IDF columns containing `urltoken`; map `urltoken` to `<UNK>` for the BiLSTM). The Kaggle pre-filter keeps URLs because it is trained on URL-bearing data.
* **Mentions:** 38% of HumAID tweets vs 63% CrisisLexT26 vs 26% Kaggle. **Emoji:** 3.0% / 1.9% / 0%. **Length:** HumAID tweets are longer (p99 = 57 tokens vs 32-33).
* **Kaggle length vs target:** disaster tweets are slightly longer (mean 108 vs 96 characters, +6% tokens) - a mild effect that TF-IDF row normalisation largely removes.
