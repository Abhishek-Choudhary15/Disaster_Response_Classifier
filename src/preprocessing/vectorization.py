
"""Train-only TF-IDF fitting for HumAID text classification."""

from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(train_texts, max_features=50000, min_df=2):
    """Fit TF-IDF on training texts only."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    train_matrix = vectorizer.fit_transform(train_texts)
    return vectorizer, train_matrix


def transform_tfidf(vectorizer, texts):
    """Transform validation/test texts without refitting."""
    return vectorizer.transform(texts)
