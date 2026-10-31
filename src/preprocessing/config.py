"""Single source of truth for preprocessing constants and artifact names.

Every script, test and notebook imports from here, so a number or a filename
is only ever written once (checklist A15, B7, E6).
"""

# ---------------------------------------------------------------- general
SEED = 42

# ---------------------------------------------------------------- TF-IDF
TFIDF_MAX_FEATURES = 50000
TFIDF_MIN_DF = 2
TFIDF_NGRAM_RANGE = (1, 2)

# ---------------------------------------------------------------- BiLSTM
BILSTM_MIN_FREQUENCY = 2
BILSTM_MAX_SEQUENCE_LENGTH = 64   # must stay >= the 99th percentile of train
                                  # token lengths (checked by run_sequences)

# ---------------------------------------------------------------- BERTweet
BERTWEET_MODEL_NAME = "vinai/bertweet-base"
# BERTweet-base has 130 position embeddings (128 usable tokens + specials).
# Anything above 128 cannot be fed to the model.
BERTWEET_MAX_LENGTH = 128
BERTWEET_HARD_LIMIT = 128

# ---------------------------------------------------------------- Kaggle split
KAGGLE_TEST_SIZE = 0.10
KAGGLE_VALIDATION_SIZE = 0.10

# ---------------------------------------------------------------- artifact names
# Versioned + dataset-tagged so HumAID / Kaggle artifacts can never be mixed up.
HUMAID_TFIDF_ARTIFACT = "tfidf_humaid_v1.joblib"
HUMAID_VOCAB_ARTIFACT = "bilstm_vocabulary_humaid_v1.joblib"
HUMAID_VOCAB_JSON = "bilstm_vocabulary_humaid_v1.json"
HUMAID_SCALER_ARTIFACT = "numerical_scaler_humaid_v1.joblib"
HUMAID_LABEL_MAP_ARTIFACT = "label_mapping_humaid_v1.json"
HUMAID_BERTWEET_TOKENIZER_DIR = "bertweet_tokenizer_humaid_v1"
KAGGLE_TFIDF_ARTIFACT = "tfidf_kaggle_prefilter_v1.joblib"

# HumAID labels listed on the public dataset card. The supplied files contain
# only 10 of them (dont_know_cant_judge never occurs) - see docs.
HUMAID_CARD_LABELS = (
    "caution_and_advice",
    "displaced_people_and_evacuations",
    "dont_know_cant_judge",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "not_humanitarian",
    "other_relevant_information",
    "requests_or_urgent_needs",
    "rescue_volunteering_or_donation_effort",
    "sympathy_and_support",
)
