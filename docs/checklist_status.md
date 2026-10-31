# Checklist status - evidence for every item

Status: **Done** = implemented and checked automatically/by notebook output. **Done (N/A)** = not applicable, reason given. **Human step** = needs a person to read/judge; the material is generated. **Limitation** = cannot be satisfied with the available data; documented.

Evidence sources: `notebooks/01_eda_three_datasets.ipynb` (section IDs), `docs/eda/eda_checklist_scorecard.csv`, `python -m src.preprocessing.verify_pipeline` (check IDs), `tests/`.

## EDA checklist (EDA_Checklist_Three_Datasets.pdf)

| ID | Item | Status | Evidence |
|---|---|---|---|
| A1 | Row counts | Done | notebook A1 (asserts all five counts) |
| A2 | Null / empty text | Done | profile rows `null_text`, `empty_text` (all 0) |
| A3 | Exact duplicates | Done | profile; Kaggle 110, CrisisLex 2,625 -> handled in pipeline |
| A4 | Near-duplicates / retweet chains | Done | profile `near_duplicate_rows`; `docs/split_integrity_audit.json` |
| A5 | Empty after cleaning (URL / mention / emoji only) | Done | profile rows `empty_after_*`, `no_letters_or_digits_*` (all 0) |
| A6 | HTML entities | Done | 4,886 / 359 / 1,259 rows raw -> 0 after cleaning |
| A7 | Stripped apostrophes / contractions | Done | profile; left as-is by decision |
| A8 | Length distributions (chars + tokens, p50/p95/p99) | Done | notebook A8 table + `docs/eda/length_distributions.png` |
| A9 | Vocabulary size and OOV | Done | notebook A9; `docs/oov_audit.json` |
| A10 | Language | Done | notebook A10 + D7 (CrisisLexT26 is multilingual) |
| A11 | Hashtag frequency / dominance | Done | notebook A11-A13 |
| A12 | Mention / URL frequency | Done | notebook A11-A13 |
| A13 | Emoji | Done | notebook A11-A13 |
| A14 | Tweet ID uniqueness | Done (N/A for HumAID: no ID column) | notebook A14 |
| A15 | Manual reading of 15-20 random rows | **Human step** | samples generated in A15, B-manual; record notes in `docs/eda_manual_review_notes.md` |
| B1 | Class distribution per split | Done | notebook B1 + `docs/eda/humaid_class_distribution.png` |
| B2 | Imbalance ratio | Done | 59.6 |
| B3 | Event vs label | **Limitation** | no event column; keyword proxy shown |
| B4 | Distribution consistent across splits | Done | max gap 0.03 pp |
| B5 | Cross-split duplicate leakage | Done | 5 train->validation rows, 0 elsewhere |
| B6 | Classes under 500 rows | Done | `missing_or_found_people` |
| B7 | `dont_know_cant_judge` vs `not_humanitarian` | Done (N/A: class absent) | `not_humanitarian` vs `other_relevant_information` read instead |
| C1 | Kaggle target balance | Done | 57% / 43% raw |
| C2 | `keyword` shortcut analysis | Done | notebook C2 -> not used |
| C3 | `location` usability | Done | notebook C3 -> not used |
| C4 | Hard negatives | Done (+ **Human step** to read) | notebook C4 |
| C5 | Only labeled file used | Done | notebook C5 + `verify_pipeline` C7 |
| C6 | Length by target | Done | notebook C6 |
| D1 | Schema consistency across 26 files | Done | notebook D1 |
| D2 | Full text vs IDs | Done | notebook D2 (100% full text) |
| D3 | Three label dimensions | Done | notebook D3 |
| D4 | Event distribution | Done | notebook D4 + figure |
| D5 | Era vocabulary shift | Done | notebook D5 |
| D6 | `event_name` retained | Done | notebook D6 + `verify_pipeline` D6 |
| E1 | Vocabulary overlap | Done | notebook E1 |
| E2 | Length overlay vs max_len | Done | notebook A8 figure + E2 table |
| E3 | Label taxonomy feasibility | Done | notebook E3 + `src/preprocessing/labels.py` |
| E4 | Equivalent cleaning across datasets (5 rows each) | Done | notebook E4 |

## Feature engineering checklist (Feature_Engineering_Checklist_Three_Datasets.pdf)

| ID | Item | Status | Evidence |
|---|---|---|---|
| A1 | Pipeline decided per model type | Done | `docs/preprocessing_decisions.md` s.2-4 |
| A2 | HTML unescape first (until stable) | Done | `text_cleaning.unescape_html`; `verify_pipeline` A2; tests |
| A3 | URL placeholder | Done | `clean_tweet_for_tfidf`; transformer path left to BERTweet |
| A4 | Mention placeholder | Done | same |
| A5 | Strip RT prefixes (chained) | Done | `verify_pipeline` A5; tests |
| A6 | Hashtag rule | Done | `#` dropped, word kept (TF-IDF path) |
| A7 | Emoji strategy | Done | TF-IDF keeps emojis; BERTweet demojizes (`emoji==0.6.0`) |
| A8 | Elongations | Done | left as-is (decision table) |
| A9 | Casing per model | Done | lower-case TF-IDF/BiLSTM; cased transformer |
| A10 | Punctuation handling | Done | `verify_pipeline` A10; tests |
| A11 | Contractions not "fixed" | Done | decision doc s.2 |
| A12 | Max length from EDA percentile per model | Done | `run_sequences` raises if length < p99; BERTweet 128 (`verify_pipeline` A12a/b) |
| A13 | Deterministic | Done | `verify_pipeline` A13; tests |
| A14 | No leaking features | Done | `verify_pipeline` A14 |
| A15 | One shared cleaning module | Done | `verify_pipeline` A15; tests |
| B1 | TF-IDF fitted on HumAID train only | Done | `verify_pipeline` B1 (re-fit comparison) |
| B2 | Vocabulary train-only, cutoff, `<UNK>` | Done | `verify_pipeline` B2 |
| B3 | BERTweet tokenizer + length | Done after Colab run | `verify_pipeline` B3, `bertweet_preprocessing_report.json` |
| B4 | label2id saved and reusable | Done | `verify_pipeline` B4 |
| B5 | OOV logged | Done | `docs/oov_audit.json`; `verify_pipeline` B5 |
| B6 | Event info excluded | Done | event data not present; features text-only (A14) |
| B7 | Versioned artifact names | Done | `config.py`; `verify_pipeline` B7 |
| B8 | Re-verify cleaning on fresh rows | Done | notebook "Re-verification" cell |
| C1 | Own Kaggle vectorizer, own train split | Done | `verify_pipeline` C1 |
| C2 | `keyword` decision | Done | not used; evidence in notebook C2 |
| C3 | `location` decision | Done | not used; evidence in notebook C3 |
| C4 | Binary target encoded separately | Done | `verify_pipeline` C4 |
| C5 | Same cleaning rules | Done | `verify_pipeline` C5 |
| C6 | Distinct artifact filenames | Done | `config.py`; `verify_pipeline` B7, C1 |
| C7 | Only labeled file used | Done | `verify_pipeline` C7 |
| D1 | No fit on CrisisLexT26 | Done | `verify_pipeline` D1; test `test_crisislex_scripts_never_fit` |
| D2 | Same cleaning function | Done | `verify_pipeline` A15 (all scripts import it) |
| D3 | Rehydration | Done (N/A: full text shipped) | notebook D2 |
| D4 | Label mapping with saved IDs, unmapped flagged | Done | `verify_pipeline` D4 |
| D5 | OOV vs frozen vocabulary | Done | `docs/oov_audit.json` |
| D6 | Per-event preserved | Done | `verify_pipeline` D6 |
| E1 | Same feature dimensionality | Done | `verify_pipeline` E1 |
| E2 | Same max_len / padding | Done | `verify_pipeline` E2a/E2b |
| E3 | Spot-check 5 cleaned rows | Done | notebook E4 |
| E4 | Same label IDs | Done | `verify_pipeline` D4 |
| E5 | Kaggle artifacts never in HumAID/CrisisLex code | Done | `verify_pipeline` E5 |
| E6 | Artifacts named, versioned, loaded explicitly | Done | `config.py` + `docs/artifact_manifest.json` |
