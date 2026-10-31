"""Tests that lock in the behaviours the checklists require."""

import re
import unittest
from pathlib import Path

from src.preprocessing import config
from src.preprocessing.labels import (
    CRISISLEX_LABEL_MAP,
    CRISISLEX_PREFILTER_MAP,
    build_label_mapping,
    describe_label_schema,
)
from src.preprocessing.text_cleaning import (
    clean_tweet_for_tfidf,
    fix_mojibake,
    normalize_for_dedup,
    prepare_tweet_for_transformer,
    unescape_html,
)
from src.preprocessing.vectorization import TFIDF_TOKEN_PATTERN, fit_tfidf, transform_tfidf
from src.preprocessing.vocabulary import (
    PAD_TOKEN,
    UNK_TOKEN,
    build_vocabulary,
    encode_and_pad,
    tokenize_for_vocabulary,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "preprocessing"


class TestCleaning(unittest.TestCase):
    RAW = "RT @a: RT @b: HELP &amp;amp; boat!!! http://t.co/x #Flood"

    def test_deterministic_and_idempotent(self):
        once = clean_tweet_for_tfidf(self.RAW)
        self.assertEqual(once, clean_tweet_for_tfidf(self.RAW))
        self.assertEqual(clean_tweet_for_tfidf(once), once)

    def test_double_escaped_entities_decoded(self):
        self.assertEqual(unescape_html("a &amp;amp; b &amp;lt; c"), "a & b < c")
        for fn in (clean_tweet_for_tfidf, prepare_tweet_for_transformer):
            self.assertNotIn("&amp;", fn(self.RAW))

    def test_chained_retweet_prefix_removed_on_both_paths(self):
        self.assertTrue(clean_tweet_for_tfidf(self.RAW).startswith("help"))
        self.assertTrue(prepare_tweet_for_transformer(self.RAW).startswith("HELP"))

    def test_mojibake_removed_only_known_families(self):
        self.assertEqual(fix_mojibake("Carolina\xe5\xcaAblaze"), "Carolina Ablaze")
        self.assertNotIn("\x89", fix_mojibake("dash \x89\xdb\xd2 here"))
        self.assertEqual(fix_mojibake("caf\xe9 \u00f1"), "caf\xe9 \u00f1")   # real accents untouched

    def test_transformer_path_keeps_signals(self):
        out = prepare_tweet_for_transformer("HELP!!! #Rescue 🚨 https://x.co @team")
        for piece in ("HELP!!!", "#Rescue", "🚨", "https://x.co", "@team"):
            self.assertIn(piece, out)

    def test_dedup_key_collapses_retweets(self):
        self.assertEqual(normalize_for_dedup("RT @a: Help!!! http://x.co"), normalize_for_dedup("help"))


class TestTokenisation(unittest.TestCase):
    def test_punctuation_split_and_contractions_kept(self):
        self.assertEqual(tokenize_for_vocabulary("help!!!"), ["help", "!!!"])
        self.assertEqual(tokenize_for_vocabulary("can't stop 🚨"), ["can't", "stop", "🚨"])

    def test_tfidf_pattern_keeps_emoji_and_urgency(self):
        tokens = re.findall(TFIDF_TOKEN_PATTERN, "help!!! 🚨 don't")
        self.assertIn("!!!", tokens)
        self.assertIn("🚨", tokens)
        self.assertIn("don't", tokens)

    def test_vocabulary_train_only_and_unknown_mapping(self):
        vocab = build_vocabulary(["flood help", "flood boat"], min_frequency=1)
        self.assertNotIn("earthquake", vocab)
        ids = encode_and_pad("earthquake flood", vocab, max_length=4)
        self.assertEqual(ids[0], vocab[UNK_TOKEN])
        self.assertEqual(ids[2:], [vocab[PAD_TOKEN]] * 2)
        self.assertEqual(len(ids), 4)

    def test_tfidf_never_sees_validation_vocabulary(self):
        vec, _ = fit_tfidf(["flood help", "flood boat"], min_df=1)
        self.assertNotIn("earthquake", vec.vocabulary_)
        out = transform_tfidf(vec, ["earthquake flood"])
        self.assertEqual(out.shape[1], len(vec.vocabulary_))


class TestConfigAndLabels(unittest.TestCase):
    def test_bertweet_length_fits_model(self):
        self.assertLessEqual(config.BERTWEET_MAX_LENGTH, 128)

    def test_artifact_names_are_versioned(self):
        for name in (config.HUMAID_TFIDF_ARTIFACT, config.HUMAID_VOCAB_ARTIFACT,
                     config.HUMAID_SCALER_ARTIFACT, config.HUMAID_LABEL_MAP_ARTIFACT,
                     config.KAGGLE_TFIDF_ARTIFACT):
            self.assertRegex(name, r"_v\d+\.")
        self.assertNotEqual(config.HUMAID_TFIDF_ARTIFACT, config.KAGGLE_TFIDF_ARTIFACT)

    def test_schema_reports_missing_card_label(self):
        schema = describe_label_schema(
            [l for l in config.HUMAID_CARD_LABELS if l != "dont_know_cant_judge"])
        self.assertEqual(schema["num_classes_present"], 10)
        self.assertEqual(schema["card_labels_absent_from_data"], ["dont_know_cant_judge"])

    def test_crisislex_mapping_targets_exist_in_card_labels(self):
        for label, _ in CRISISLEX_LABEL_MAP.values():
            if label is not None:
                self.assertIn(label, config.HUMAID_CARD_LABELS)
        self.assertEqual(set(CRISISLEX_PREFILTER_MAP.values()), {0, 1})

    def test_label_mapping_contiguous(self):
        label2id, _ = build_label_mapping(["b", "a", "c"])
        self.assertEqual(list(label2id.values()), [0, 1, 2])


class TestLanguage(unittest.TestCase):
    def test_guess_language(self):
        try:
            import langdetect  # noqa: F401
        except ImportError:
            self.skipTest("langdetect not installed")
        from src.preprocessing.language import guess_language
        self.assertEqual(guess_language("Heavy flooding in the city, please send boats and blankets"), "en")
        self.assertEqual(guess_language("Fuerte inundacion en la ciudad, necesitamos ayuda urgente por favor"), "es")
        self.assertEqual(guess_language("ok"), "too_short")


class TestCodeHygiene(unittest.TestCase):
    def test_crisislex_scripts_never_fit(self):
        pattern = re.compile(r"\.(fit|fit_transform|partial_fit)\(|build_vocabulary\(|TfidfVectorizer\(")
        for path in SRC.glob("run_crisislex*.py"):
            self.assertIsNone(pattern.search(path.read_text()), f"fit call in {path.name}")

    def test_no_script_reimplements_cleaning(self):
        for path in SRC.glob("run_*.py"):
            self.assertIsNone(re.search(r"def (clean|normalize|prepare)_\w+\(", path.read_text()), path.name)

    def test_kaggle_script_never_reads_unlabeled_files(self):
        text = (SRC / "run_kaggle_preprocessing.py").read_text()
        self.assertNotIn("nlpgettingstarted_test", text)
        self.assertNotIn("sample_submission", text)


if __name__ == "__main__":
    unittest.main()
