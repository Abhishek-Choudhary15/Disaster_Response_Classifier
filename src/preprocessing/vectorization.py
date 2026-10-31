"""Train-only TF-IDF fitting. One configuration shared by HumAID and Kaggle."""

from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.config import (
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)

# Words (with inner apostrophes), runs of ! or ? (urgency) and emojis.
# The old pattern r"\b\w+\b" silently dropped every emoji and all punctuation.
TFIDF_TOKEN_PATTERN = (
    r"(?u)\b\w+(?:['’]\w+)*\b|[!?]+|[\U0001F300-\U0001FAFF\u2600-\u27BF]"
)


def fit_tfidf(train_texts, max_features=TFIDF_MAX_FEATURES, min_df=TFIDF_MIN_DF):
    """Fit TF-IDF on training texts only."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        token_pattern=TFIDF_TOKEN_PATTERN,
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    return vectorizer, train_matrix


def transform_tfidf(vectorizer, texts):
    """Transform validation/test/external texts without refitting."""
    return vectorizer.transform(texts)
