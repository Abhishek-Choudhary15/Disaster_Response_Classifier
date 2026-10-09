# HumAID Preprocessing Decisions

## 1. Dataset and Splits

The project uses the HumAID disaster-related tweet dataset.

| Split | Number of examples |
|---|---:|
| Training | 53,531 |
| Validation | 7,793 |
| Test | 15,160 |
| Total | 76,484 |

All three splits contain the same 10 observed classes. No missing text or labels were found during preprocessing validation.

**Label-count discrepancy:** The development checklist refers to 11 labels, but the supplied dataset splits contain 10 distinct classes. No artificial class has been added. The intended dataset version should be verified before finalizing the project documentation.

## 2. Model-Specific Text Cleaning

Different text representations are used for TF-IDF and transformer models because these models rely on different text signals.

### TF-IDF

- Convert text to lowercase.
- Remove a leading retweet (`RT`) prefix.
- Replace URLs with `URLTOKEN`.
- Replace user mentions with `USERTOKEN`.
- Remove the `#` symbol while retaining the hashtag word.
- Preserve punctuation and emojis.
- Normalize whitespace.

### Transformer Input

- Preserve original casing.
- Remove a leading retweet prefix.
- Preserve hashtags, punctuation, emojis, elongated spellings, URLs, and mentions.
- Normalize HTML entities and whitespace.

These policies are implemented as separate cleaning functions.

## 3. TF-IDF Vectorization

The current vectorizer configuration is:

- Maximum vocabulary size: 50,000 features.
- Minimum document frequency: 2.
- N-gram range: unigram and bigram.
- Sublinear term frequency: enabled.
- Token pattern: `(?u)\b\w+\b`.

The vectorizer is fitted only on training text. The fitted vectorizer transforms validation and test text without refitting.

The fitted vectorizer is saved as:

`models/preprocessing/tfidf_vectorizer.joblib`

## 4. Numerical Feature Engineering

Eight numerical features are extracted from the original tweet text:

1. `char_length`
2. `word_count`
3. `exclamation_count`
4. `question_count`
5. `hashtag_count`
6. `mention_count`
7. `url_count`
8. `uppercase_ratio`

The feature tables are checked for missing and infinite values.

## 5. Numerical Feature Scaling

`StandardScaler` is fitted using the training numerical features only.

The same fitted scaler transforms the training, validation, and test features. Validation and test statistics are not used to fit the scaler.

The fitted scaler is saved as:

`models/preprocessing/numerical_scaler.joblib`

Raw and scaled feature tables are saved separately under `data/processed/`.

Scaling is intended for models sensitive to feature scale. Tree-based models may use the raw numerical features instead.

## 6. Label Encoding

The label mapping is built from training labels. Validation and test labels are checked against that mapping, and unknown labels cause preprocessing to stop.

Encoded labels are saved separately for each split. The mapping is saved as:

`models/preprocessing/label_mapping.json`

The observed dataset contains 10 classes; the checklist's reference to 11 classes remains to be verified.

## 7. Reproducibility and Data Leakage

- The original train, validation, and test splits are retained.
- TF-IDF is fitted only on training data.
- Numerical scaling is fitted only on training data.
- Validation and test data are transformed using the fitted training artifacts.
- Sparse TF-IDF matrices are stored separately instead of being converted to dense arrays.
- Preprocessing tests are run with `python -m unittest discover -s tests -v`.

## 8. Remaining Work

The following items are not yet confirmed as implemented:

- Select and document the tokenizer for any sequence-based or transformer model.
- Determine maximum sequence length from training-data exploration.
- Define padding and truncation policies if token sequences are used.
- Evaluate out-of-vocabulary behavior if an embedding-based model is implemented.
- Verify that disaster-event names or other event-specific information do not create unintended leakage.
- Record relevant library and artifact versions for reproducible experiments.
- Confirm whether the intended dataset version has 10 or 11 classes.

This document describes the current preprocessing pipeline and distinguishes implemented steps from work still pending.
## 9. Duplicate and Data-Leakage Audit

A read-only audit was performed on the original train, validation, and test CSV files.

### Findings

- Training: 0 exact duplicate tweet texts within the split.
- Validation: 1 repeated tweet text.
- Test: 1 repeated tweet text.
- Train–validation overlap: 4 exact shared tweet texts.
- Train–test overlap: 0 exact shared tweet texts.
- Validation–test overlap: 0 exact shared tweet texts.
- The four train–validation overlaps have matching labels.
- The repeated validation and test texts have matching labels within their respective splits.
- The available CSV files contain only `tweet_text` and `class_label`; event metadata is not available in these files.

### Decision

The original dataset splits have been preserved. No records were removed or modified.

The train–validation overlap is documented as a potential evaluation limitation. Because the available files contain no event identifiers, this audit cannot establish whether the splits are independent at the disaster-event level.

Tweets may naturally mention specific disaster events. Generalization to entirely unseen events has not been established by this audit and would require suitable event metadata or an independently defined event-based evaluation split.

## 10. BiLSTM Vocabulary and OOV Audit

The BiLSTM vocabulary was built using HumAID training tweets only, with a minimum token frequency of 2.

- Vocabulary size: 29,351 entries, including `<PAD>` and `<UNK>`.
- `<PAD>` ID: 0.
- `<UNK>` ID: 1.
- Maximum sequence length: 64 tokens.
- Padding and truncation: post-padding and post-truncation.

OOV audit results:

| Split | Total tokens | OOV tokens | OOV rate |
|---|---:|---:|---:|
| Training | 1,170,959 | 50,842 | 4.34% |
| Validation | 169,652 | 10,410 | 6.14% |
| Test | 331,808 | 19,794 | 5.97% |

Validation and test use the vocabulary learned from training data. Tokens excluded by the minimum-frequency cutoff or absent from the training vocabulary map to `<UNK>`. The nonzero training OOV rate is expected because tokens appearing fewer than twice are excluded from the vocabulary.
- **BERTweet maximum length:** 256 tokens, selected after measuring tokenized HumAID training tweets with the saved BERTweet tokenizer. At this limit, 52 of 53,531 training tweets (0.097%) exceed the limit. Padding and truncation use `max_length=256`.
## 11. CrisisLexT26 Transformation and Cross-Dataset Checks

### Transformation policy

CrisisLexT26 is used only for external transformation and evaluation preparation. The pipeline does not fit a new TF-IDF vectorizer, BiLSTM vocabulary, or BERTweet tokenizer on CrisisLexT26. It reuses the saved HumAID artifacts and preserves event names in the transformed table.

### Results

- Labeled event files processed: 26.
- Tweets processed: 27,933.
- TF-IDF matrix shape: 27,933 × 50,000.
- BiLSTM sequence shape: 27,933 × 64.
- BERTweet input IDs shape: 27,933 × 256.
- BERTweet attention mask shape: 27,933 × 256.
- BiLSTM vocabulary OOV tokens: 120,897 out of 425,886 tokens (28.39%).
- Existing unit tests: 8 passed.

### Label mapping

The EDA-proposed mapping was applied using the saved HumAID label IDs.

- Good mappings: 10,959 rows.
- Rough mappings: 13,555 rows.
- Unmapped rows: 3,419 rows.

Rough mappings are provisional and should not be interpreted as verified ground truth. Unmapped categories remain without a HumAID label. The project currently observes 10 HumAID classes, although the checklist refers to 11; no artificial class was added.

### Cross-dataset verification

- HumAID and CrisisLexT26 TF-IDF matrices both have 50,000 features.
- HumAID and CrisisLexT26 BiLSTM sequences both use length 64.
- HumAID and CrisisLexT26 BERTweet inputs both use length 256 with matching attention masks.
- The tested Kaggle and mapped CrisisLexT26 rows matched the shared TF-IDF cleaning function.
- Mapped CrisisLexT26 label IDs matched the saved HumAID label mapping.
- The Kaggle vectorizer is stored separately from the HumAID vectorizer.

### Limitations and pending verification

- CrisisLexT26's 28.38% BiLSTM OOV rate is higher than HumAID's test OOV rate; the HumAID vocabulary remains frozen.
- Event-level generalization has not been established by the available audit.
- Rough category mappings require cautious interpretation.
- Cross-dataset cleaning checks used samples, not exhaustive row-by-row comparisons.
- Artifact-loading paths should be checked in the relevant code before modeling.
- The optional `emoji` package is absent; BERTweet preprocessing completed with the tokenizer's current behavior.
