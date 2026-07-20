"""
Ticket 2 validation / mechanism trace.

Reproduces every number cited in tickets/ticket-2-normalization.md under
"Held-out evidence" and "Concrete examples and the actual mechanism":
  - McNemar's exact test comparing baseline vs keep_hashtags_raw on heldout
  - per-flipped-id normalized text diff (checks whether the flip is actually
    caused by hashtag content in that specific tweet)
  - sklearn tokenizer inspection (confirms '#' is stripped regardless of config)
  - direct fitted-vocabulary diff between the two configs, traced back to source
    training tweets
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.data import load_raw, get_splits
from pipeline.model import build_texts, fit_tfidf_logreg, predict_scores, scores_to_preds, REFERENCE_REPRO_TFIDF_KWARGS, REFERENCE_REPRO_C
from pipeline.preprocess import apply_config
from sklearn.feature_extraction.text import TfidfVectorizer
from experiments.ticket1_validation import mcnemar_test


def run_config(train_df, eval_df, config_name):
    train_texts = build_texts(train_df, config_name)
    eval_texts = build_texts(eval_df, config_name)
    vec, clf = fit_tfidf_logreg(train_texts, train_df["target"].values, seed=3102,
                                 C=REFERENCE_REPRO_C, **REFERENCE_REPRO_TFIDF_KWARGS)
    scores = predict_scores(vec, clf, eval_texts)
    preds = scores_to_preds(scores, 0.5)
    return preds, scores, vec


def main():
    df = load_raw()
    train_df, dev_df, heldout_df = get_splits(df)
    y_true = heldout_df["target"].values

    preds_base, scores_base, vec_base = run_config(train_df, heldout_df, "baseline")
    preds_kh, scores_kh, vec_kh = run_config(train_df, heldout_df, "keep_hashtags_raw")

    mcnemar_test(y_true, preds_base, preds_kh, "baseline", "keep_hashtags_raw")

    print("\n=== Flipped heldout ids: per-example normalized-text diff ===")
    changed_ids = heldout_df["id"][preds_base != preds_kh].tolist()
    print(f"changed ids: {changed_ids}")
    for _id in changed_ids:
        row = heldout_df[heldout_df["id"] == _id].iloc[0]
        base_norm = apply_config(row["text"], "baseline")
        kh_norm = apply_config(row["text"], "keep_hashtags_raw")
        identical = base_norm == kh_norm
        print(f"--- id={_id} true={row['target']} normalized text identical across configs: {identical} ---")
        print(f"  raw: {row['text'][:100]!r}")
        if not identical:
            print(f"  baseline norm: {base_norm!r}")
            print(f"  keep_hashtags_raw norm: {kh_norm!r}")

    print("\n=== Tokenizer check: does sklearn's default tokenizer treat '#word' differently from 'word'? ===")
    vec_probe = TfidfVectorizer(**REFERENCE_REPRO_TFIDF_KWARGS)
    sample_hashtag = apply_config("#CCOT #TCOT #radiation Nuclear Emergency Tracking Center", "keep_hashtags_raw")
    sample_unwrapped = apply_config("#CCOT #TCOT #radiation Nuclear Emergency Tracking Center", "baseline")
    tokens_hashtag = vec_probe.build_analyzer()(sample_hashtag)
    tokens_unwrapped = vec_probe.build_analyzer()(sample_unwrapped)
    print(f"keep_hashtags_raw normalized: {sample_hashtag!r} -> tokens: {tokens_hashtag}")
    print(f"baseline normalized:          {sample_unwrapped!r} -> tokens: {tokens_unwrapped}")
    print("tokens identical:", tokens_hashtag == tokens_unwrapped)

    print("\n=== Fitted vocabulary diff between configs (trained on train_ids) ===")
    train_texts_base = build_texts(train_df, "baseline")
    train_texts_kh = build_texts(train_df, "keep_hashtags_raw")
    vec_b = TfidfVectorizer(**REFERENCE_REPRO_TFIDF_KWARGS).fit(train_texts_base)
    vec_k = TfidfVectorizer(**REFERENCE_REPRO_TFIDF_KWARGS).fit(train_texts_kh)
    vocab_b, vocab_k = set(vec_b.vocabulary_), set(vec_k.vocabulary_)
    only_base = vocab_b - vocab_k
    only_kh = vocab_k - vocab_b
    print(f"vocab sizes: baseline={len(vocab_b)}  keep_hashtags_raw={len(vocab_k)}")
    print(f"words only in baseline vocab ({len(only_base)}): {sorted(only_base)}")
    print(f"words only in keep_hashtags_raw vocab ({len(only_kh)}): {sorted(only_kh)}")

    print("\n=== Tracing merged/malformed tokens back to source training tweets ===")
    needles = list(only_base)
    for n in needles:
        key = n[:15].replace("#", "")
        matches = train_df[train_df["text"].str.lower().str.replace("#", "", regex=False)
                            .str.replace(" ", "", regex=False).str.contains(key, regex=False, na=False)]
        if len(matches):
            print(f"  {n!r} <- {matches.iloc[0]['text'][:110]!r}")


if __name__ == "__main__":
    main()
