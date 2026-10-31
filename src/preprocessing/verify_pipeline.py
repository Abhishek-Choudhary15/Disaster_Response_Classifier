"""Machine-checkable evidence for the Feature Engineering checklist.

Run from the repo root AFTER every run_* script and run_audits:
    python -m src.preprocessing.verify_pipeline
Exit code 1 if any check FAILS. Writes docs/artifact_manifest.json.
Item IDs (A1..E6) follow the Feature Engineering checklist PDF.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.preprocessing import config
from src.preprocessing.features import FEATURE_COLUMNS
from src.preprocessing.text_cleaning import (
    clean_tweet_for_tfidf,
    normalize_for_dedup,
    prepare_tweet_for_transformer,
)
from src.preprocessing.vectorization import TFIDF_TOKEN_PATTERN, fit_tfidf
from src.preprocessing.vocabulary import build_vocabulary, tokenize_for_vocabulary

ROOT = Path(__file__).resolve().parents[2]
P, M, SRC = ROOT / "data/processed", ROOT / "models/preprocessing", ROOT / "src/preprocessing"
rows = []


def check(item, description, ok, detail=""):
    status = "SKIP" if ok is None else ("PASS" if bool(ok) else "FAIL")
    rows.append((item, description, status, detail))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ load
train_raw = pd.read_csv(ROOT / "data/raw/humaid_train.csv")
train_clean = train_raw.tweet_text.map(clean_tweet_for_tfidf).tolist()
vec = joblib.load(M / config.HUMAID_TFIDF_ARTIFACT)
vocab = joblib.load(M / config.HUMAID_VOCAB_ARTIFACT)
label2id = read_json(M / config.HUMAID_LABEL_MAP_ARTIFACT)["label2id"]
run_scripts = sorted(SRC.glob("run_*.py"))

# ================================================================== A
probe = "RT @a: RT @b: HELP &amp;amp; boat!!! \x89ÛÒ http://t.co/x #Flood"
tf, tr = clean_tweet_for_tfidf(probe), prepare_tweet_for_transformer(probe)
check("A2", "HTML double-escapes decoded on BOTH paths", "&amp;" not in tf and "&amp;" not in tr and "&" in tf)
check("A5", "Chained RT prefixes stripped on BOTH paths", not tf.startswith("rt ") and not tr.startswith("RT "))
check("A13", "Cleaning deterministic and idempotent",
      tf == clean_tweet_for_tfidf(probe) and clean_tweet_for_tfidf(tf) == tf)
toks = tokenize_for_vocabulary(clean_tweet_for_tfidf("help!!! 🚨 can't"))
check("A10", "Tokenizers keep '!!!', emojis and apostrophes",
      "!!!" in toks and "🚨" in toks and "can't" in toks and "🚨" in re.findall(TFIDF_TOKEN_PATTERN, "help 🚨"))
bil = read_json(P / "bilstm_sequence_report.json")
check("A12a", "BiLSTM max length >= train 99th percentile",
      bil["max_sequence_length"] >= bil["train_length_percentiles"]["p99"],
      f"max_len {bil['max_sequence_length']} vs p99 {bil['train_length_percentiles']['p99']:.0f}")
check("A12b", "BERTweet max length <= model limit (128)",
      config.BERTWEET_MAX_LENGTH <= config.BERTWEET_HARD_LIMIT == 128, f"{config.BERTWEET_MAX_LENGTH}")
check("A14", "Numeric features come from tweet text only (8 documented columns)",
      FEATURE_COLUMNS == ["char_length", "word_count", "exclamation_count", "question_count",
                          "hashtag_count", "mention_count", "url_count", "uppercase_ratio"])
own_cleaners = [p.name for p in run_scripts if re.search(r"def (clean|normalize|prepare)_\w+\(", p.read_text())]
missing_import = [p.name for p in run_scripts if "text_cleaning" not in p.read_text()]
check("A15", "One shared cleaning module; no script re-implements it",
      not own_cleaners and not missing_import, ", ".join(own_cleaners + missing_import) or "all scripts import text_cleaning")

# ================================================================== B
refit, _ = fit_tfidf(train_clean)
check("B1", "HumAID TF-IDF == fresh fit on train only (also detects stale artifacts)",
      set(refit.vocabulary_) == set(vec.vocabulary_), f"{len(vec.vocabulary_)} features")
check("B2", "BiLSTM vocabulary == fresh build from train only",
      build_vocabulary(train_clean, config.BILSTM_MIN_FREQUENCY) == vocab, f"{len(vocab)} entries")
ids_h = {s: P / f"{s}_bertweet_input_ids.npy" for s in ("train", "validation", "test")}
if all(p.exists() for p in ids_h.values()):
    widths = {s: np.load(p, mmap_mode="r").shape[1] for s, p in ids_h.items()}
    check("B3", "BERTweet inputs exist at config length", set(widths.values()) == {config.BERTWEET_MAX_LENGTH}, str(widths))
else:
    check("B3", "BERTweet inputs exist at config length", None, "run_bertweet.py not run yet")
check("B4", "label map: sorted, contiguous, 10 classes, absent card label documented",
      list(label2id.values()) == list(range(len(label2id))) and list(label2id) == sorted(label2id)
      and len(label2id) == 10 and "dont_know_cant_judge" not in label2id, f"{len(label2id)} classes")
oov_path, split_path = ROOT / "docs/oov_audit.json", ROOT / "docs/split_integrity_audit.json"
check("B5", "OOV audit exists for HumAID splits, Kaggle and CrisisLexT26",
      oov_path.exists() and {"humaid_train", "humaid_test", "kaggle_all_splits", "crisislex26_all"} <= set(read_json(oov_path)))
names = [config.HUMAID_TFIDF_ARTIFACT, config.HUMAID_VOCAB_ARTIFACT, config.HUMAID_SCALER_ARTIFACT,
         config.HUMAID_LABEL_MAP_ARTIFACT, config.KAGGLE_TFIDF_ARTIFACT]
check("B7", "Versioned, dataset-tagged artifacts exist", all((M / n).exists() for n in names) and
      all(re.search(r"_v\d+\.", n) for n in names))
align = {}
for s in ("train", "validation", "test"):
    n = len(pd.read_csv(ROOT / f"data/raw/humaid_{s}.csv", usecols=["tweet_text"]))
    sizes = {"tfidf": joblib.load(P / f"{s}_tfidf.joblib").shape[0],
             "numeric": len(pd.read_csv(P / f"{s}_scaled_numerical_features.csv")),
             "labels": len(pd.read_csv(P / f"{s}_labels.csv")),
             "sequences": np.load(P / f"{s}_bilstm_sequences.npy", mmap_mode="r").shape[0]}
    if ids_h[s].exists():
        sizes["bertweet"] = np.load(ids_h[s], mmap_mode="r").shape[0]
    align[s] = set(sizes.values()) == {n}
check("B-align", "Row counts identical across every HumAID representation", all(align.values()), str(align))

# ================================================================== C
kv = joblib.load(M / config.KAGGLE_TFIDF_ARTIFACT)
check("C1", "Kaggle vectorizer is a separate artifact", set(kv.vocabulary_) != set(vec.vocabulary_))
krep = read_json(P / "kaggle/kaggle_preprocessing_report.json")
kt = {s: pd.read_csv(P / f"kaggle/{s}_text.csv") for s in ("train", "validation", "test")}
kl = {s: pd.read_csv(P / f"kaggle/{s}_labels.csv")["target"] for s in kt}
check("C4", "Kaggle target is binary {0,1}, separate from HumAID label map",
      all(set(v.unique()) <= {0, 1} for v in kl.values()) and krep["target_mapping"] == {"0": "not disaster-related", "1": "disaster-related"})
sample = kt["train"].sample(300, random_state=0)
check("C5", "Kaggle cleaned_text equals the shared cleaning function",
      (sample.text.map(clean_tweet_for_tfidf) == sample.cleaned_text.fillna("")).all())
check("C-split", "Kaggle splits share no near-duplicate text (80/10/10, seeded)",
      krep["shared_texts_between_splits"] == 0 and sum(krep["split_sizes"].values()) == krep["rows_after_deduplication"],
      str(krep["split_sizes"]))
keys = pd.concat([d.text.map(normalize_for_dedup) for d in kt.values()])
check("C-dedupe", "No normalised Kaggle text appears twice", not keys.duplicated().any())
kscript = (SRC / "run_kaggle_preprocessing.py").read_text()
check("C7", "Kaggle script reads only the labeled training file",
      "nlpgettingstarted_test" not in kscript and "sample_submission" not in kscript and
      krep["competition_test_used_for_fitting"] is False)
check("C2/C3", "keyword and location documented as NOT used", krep["keyword_used"] is False and krep["location_used"] is False)

# ================================================================== D
fit_call = re.compile(r"\.(fit|fit_transform|partial_fit)\(|build_vocabulary\(|TfidfVectorizer\(")
bad = [p.name for p in run_scripts if p.name.startswith("run_crisislex") and fit_call.search(p.read_text())]
check("D1", "No fit/build call in any CrisisLexT26 script", not bad, ", ".join(bad) or "no fit calls found")
cl_dir = P / "crisislext26"
cl = pd.read_csv(cl_dir / "crisislex26_transformed_v1.csv")
m = cl.dropna(subset=["humaid_label"])
check("D4", "CrisisLex label IDs come from the saved HumAID mapping",
      (m.humaid_label.map(label2id) == m.label_id).all(), f"{len(m)} mapped rows")
check("D6", "Per-event identity preserved (26 events, event_name column)",
      cl["event_name"].nunique() == 26 and cl["event_name"].notna().all(), f"{cl['event_name'].nunique()} events")
check("D-flags", "Duplicate / language / prefilter / class-eval flags present",
      {"is_duplicate_text", "language_guess", "is_english_guess", "prefilter_label",
       "use_for_class_eval", "use_for_prefilter_eval"} <= set(cl.columns))
ev_rows = cl[cl["use_for_class_eval"] | cl["use_for_prefilter_eval"]]
check("D-lang", "Evaluation rows are English-guess and non-duplicate (CrisisLexT26 is multilingual)",
      bool(ev_rows["is_english_guess"].all() and not ev_rows["is_duplicate_text"].any()),
      f"{cl['is_english_guess'].mean()*100:.1f}% of all rows guessed English")

# ================================================================== E
cl_t = joblib.load(cl_dir / "crisislex26_humaid_tfidf_v1.joblib")
h_t = joblib.load(P / "test_tfidf.joblib")
check("E1", "TF-IDF dimensions identical (HumAID vs CrisisLex)",
      cl_t.shape[1] == h_t.shape[1] == len(vec.vocabulary_), f"{cl_t.shape[1]} / {h_t.shape[1]}")
seq_h = np.load(P / "test_bilstm_sequences.npy", mmap_mode="r")
seq_c = np.asarray(joblib.load(cl_dir / "crisislex26_humaid_sequences_v1.joblib"))
check("E2a", "BiLSTM sequence length identical", seq_h.shape[1] == seq_c.shape[1] == config.BILSTM_MAX_SEQUENCE_LENGTH,
      f"{seq_h.shape[1]} / {seq_c.shape[1]}")
ic = cl_dir / "crisislex26_bertweet_input_ids_v1.npy"
if ic.exists() and ids_h["test"].exists():
    a, b = np.load(ids_h["test"], mmap_mode="r"), np.load(ic, mmap_mode="r")
    check("E2b", "BERTweet length identical and <= 128", a.shape[1] == b.shape[1] == config.BERTWEET_MAX_LENGTH <= 128, f"{a.shape[1]} / {b.shape[1]}")
    check("E-align", "CrisisLex rows identical across tfidf/sequences/bertweet/csv",
          cl_t.shape[0] == seq_c.shape[0] == b.shape[0] == len(cl))
else:
    check("E2b", "BERTweet length identical and <= 128", None, "BERTweet outputs not generated yet")
    check("E-align", "CrisisLex rows identical across tfidf/sequences/bertweet/csv", cl_t.shape[0] == seq_c.shape[0] == len(cl))
leaky = [p.name for p in SRC.glob("*.py")
         if p.name not in ("run_kaggle_preprocessing.py", "verify_pipeline.py", "config.py")
         and re.search(r"KAGGLE_|tfidf_kaggle|processed/kaggle", p.read_text())]
check("E5", "Kaggle artifacts never referenced by HumAID/CrisisLex code", not leaky, ", ".join(leaky) or "clean")


# ------------------------------------------------------------------ manifest (E6)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


try:
    import transformers
    tf_version = transformers.__version__
except Exception:
    tf_version = "not installed"
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/artifact_manifest.json").write_text(json.dumps({
    "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__,
                 "numpy": np.__version__, "transformers": tf_version},
    "config": {"BILSTM_MAX_SEQUENCE_LENGTH": config.BILSTM_MAX_SEQUENCE_LENGTH,
               "BERTWEET_MAX_LENGTH": config.BERTWEET_MAX_LENGTH, "SEED": config.SEED},
    "artifacts": {p.name: sha(p) for p in sorted(M.glob("*.*"))},
}, indent=2), encoding="utf-8")
check("E6", "Artifact manifest written (names + hashes + library versions)", (ROOT / "docs/artifact_manifest.json").exists())

width = max(len(r[1]) for r in rows)
for item, desc, status, detail in rows:
    print(f"{status:5} {item:8} {desc:<{width}}  {detail}")
n_fail = sum(r[2] == "FAIL" for r in rows)
n_skip = sum(r[2] == "SKIP" for r in rows)
print(f"\n{len(rows) - n_fail - n_skip} passed, {n_fail} failed, {n_skip} skipped")
sys.exit(1 if n_fail else 0)
