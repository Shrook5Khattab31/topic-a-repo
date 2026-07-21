"""
TF-IDF + Logistic Regression baseline, plus a trivial floor model.
Also supports an alternate CPU classifier (Linear SVM via SGD) for Ticket 4.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

from pipeline.preprocess import apply_config


def build_texts(df, config_name="baseline"):
    """Return normalized tweet text in dataframe order for reproducible vectorization."""
    return df["text"].apply(lambda t: apply_config(t, config_name)).tolist()


def floor_model(train_df, eval_df):
    """Majority-class dummy classifier: the floor any real model must clear."""
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(np.zeros((len(train_df), 1)), train_df["target"])
    preds = clf.predict(np.zeros((len(eval_df), 1)))
    scores = clf.predict_proba(np.zeros((len(eval_df), 1)))[:, 1]
    return preds, scores


def fit_tfidf_logreg(train_texts, train_labels, seed=3102, C=1.0, class_weight=None, **tfidf_kwargs):
    default_tfidf = dict(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
    default_tfidf.update(tfidf_kwargs)
    vec = TfidfVectorizer(**default_tfidf)
    X_train = vec.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=1000, C=C, class_weight=class_weight, random_state=seed)
    clf.fit(X_train, train_labels)
    return vec, clf


# Config that reproduces the reference contract's heldout_f1_target_1 within
# tolerance (see tickets/ticket-1-baseline.md for the discrepancy diagnosis):
# unigrams only, no sublinear scaling, C=2.0 instead of the naive C=1.0 default.
REFERENCE_REPRO_TFIDF_KWARGS = dict(ngram_range=(1, 1), min_df=1, max_df=1.0, sublinear_tf=False)
REFERENCE_REPRO_C = 2.0


def fit_tfidf_sgd_svm(train_texts, train_labels, seed=3102, **tfidf_kwargs):
    """Second CPU classifier for Ticket 4: linear SVM trained with SGD (hinge loss)."""
    default_tfidf = dict(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
    default_tfidf.update(tfidf_kwargs)
    vec = TfidfVectorizer(**default_tfidf)
    X_train = vec.fit_transform(train_texts)
    clf = SGDClassifier(loss="modified_huber", random_state=seed, max_iter=2000, tol=1e-3)
    clf.fit(X_train, train_labels)
    return vec, clf


def predict_scores(vec, clf, texts):
    X = vec.transform(texts)
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    # SGDClassifier with modified_huber supports predict_proba; fallback to decision_function
    return clf.decision_function(X)


def scores_to_preds(scores, threshold=0.5):
    """Convert positive-class scores to binary predictions at a fixed threshold."""
    return (scores >= threshold).astype(int)


def evaluate(y_true, y_pred):
    return {
        "f1_target_1": f1_score(y_true, y_pred, pos_label=1),
        "precision_target_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_target_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
