
import unittest

from src.preprocessing.text_cleaning import (
    clean_tweet_for_tfidf,
    prepare_tweet_for_transformer,
)
from src.preprocessing.features import extract_tweet_features
from src.preprocessing.labels import build_label_mapping, encode_labels


class TestTextCleaning(unittest.TestCase):
    def test_retweet_prefix_removed(self):
        result = clean_tweet_for_tfidf(
            "RT @rescue_team: Help needed!"
        )
        self.assertTrue(result.startswith("help needed!"))

    def test_url_replaced(self):
        result = clean_tweet_for_tfidf(
            "See https://example.com for updates"
        )
        self.assertIn("urltoken", result)
        self.assertNotIn("https://", result)

    def test_mention_replaced(self):
        result = clean_tweet_for_tfidf("Contact @rescue_team now")
        self.assertIn("usertoken", result)

    def test_transformer_preserves_tweet_signals(self):
        result = prepare_tweet_for_transformer(
            "HELP!!! #Rescue 🚨"
        )
        self.assertIn("HELP!!!", result)
        self.assertIn("#Rescue", result)
        self.assertIn("🚨", result)

    def test_none_returns_empty_string(self):
        self.assertEqual(clean_tweet_for_tfidf(None), "")


class TestFeatureExtraction(unittest.TestCase):
    def test_feature_values(self):
        result = extract_tweet_features(
            ["HELP!!! #Rescue @team https://example.com"]
        )
        self.assertEqual(result.loc[0, "exclamation_count"], 3)
        self.assertEqual(result.loc[0, "hashtag_count"], 1)
        self.assertEqual(result.loc[0, "mention_count"], 1)
        self.assertEqual(result.loc[0, "url_count"], 1)


class TestLabelMapping(unittest.TestCase):
    def test_mapping_is_deterministic(self):
        mapping1, _ = build_label_mapping(["z", "a", "b"])
        mapping2, _ = build_label_mapping(["b", "z", "a"])
        self.assertEqual(mapping1, mapping2)

    def test_encode_labels(self):
        mapping, _ = build_label_mapping(["flood", "rescue"])
        encoded = encode_labels(["rescue", "flood"], mapping)
        self.assertEqual(encoded.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
