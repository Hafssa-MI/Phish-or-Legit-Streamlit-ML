import random
import re

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

SEED = 42
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|org|info|ly|net|co|xyz|top)\b\S*", re.I)
MONEY_RE = re.compile(r"[$€£]\s?\d+(?:[.,]\d+)*|\b\d+(?:[.,]\d+)*\s?(?:usd|eur|dollars|euros)\b", re.I)
NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}


def preprocess(text: str) -> str:
    text = text.lower()
    text = EMAIL_RE.sub(" emailtok ", text)
    text = URL_RE.sub(" urltok ", text)
    text = MONEY_RE.sub(" moneytok ", text)
    return NUM_RE.sub(" numtok ", text)


def word_dropout(text, rng, p=0.1):
    words = text.split()
    kept = [w for w in words if rng.random() > p]
    return " ".join(kept) if kept else text


def obfuscate(text, rng, p=0.2):
    words = text.split()
    for i, w in enumerate(words):
        if len(w) >= 5 and w.isalpha() and rng.random() < p:
            idx = [k for k, c in enumerate(w.lower()) if c in LEET]
            if idx:
                k = rng.choice(idx)
                words[i] = w[:k] + LEET[w[k].lower()] + w[k + 1:]
    return " ".join(words)


def augment_df(df, n=1, label=1, seed=SEED):
    rng = random.Random(seed)
    rows = []
    for _, r in df[df.y == label].iterrows():
        for _ in range(n):
            rows.append({"text": obfuscate(word_dropout(r.text, rng), rng), "y": r.y,
                        "source": r.source + "_aug", "gid": r.gid})
    return pd.DataFrame(rows)


def make_features(mode="union"):
    word = TfidfVectorizer(preprocessor=preprocess, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(preprocessor=preprocess, analyzer="char_wb", ngram_range=(2, 5),
                        min_df=3, sublinear_tf=True, max_features=60000)
    if mode == "word":
        return word
    if mode == "char":
        return char
    return FeatureUnion([("word", word), ("char", char)])


def make_model(kind="lr", features="union", C=3.0, alpha=0.1, balanced=True, cal_cv=5):
    family = kind.split("_")[0]
    if family == "lr":
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced" if balanced else None,
                                random_state=SEED)
    elif family == "nb":
        clf = MultinomialNB(alpha=alpha, fit_prior=not balanced)
    else:
        raise ValueError(kind)
    if kind.endswith("_cal"):
        clf = CalibratedClassifierCV(clf, method="sigmoid", cv=cal_cv)
    return Pipeline([("tfidf", make_features(features)), ("clf", clf)])


def feature_weights(pipe):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        return clf.coef_[0]
    if hasattr(clf, "feature_log_prob_"):
        return clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    raise ValueError("Use an uncalibrated LR or NB pipeline for explanations.")


def contributions(pipe, text, include_char=False):
    vec = pipe.named_steps["tfidf"]
    X = vec.transform([text]).tocoo()
    names = vec.get_feature_names_out()
    w = feature_weights(pipe)
    out = {}
    for j, v in zip(X.col, X.data):
        n = names[j]
        if n.startswith("char__") and not include_char:
            continue
        out[n.split("__", 1)[-1]] = float(v * w[j])
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1])))


def pick_threshold(y_true, scores, min_precision=0.90):
    p, r, t = precision_recall_curve(y_true, scores)
    ok = np.where(p[:-1] >= min_precision)[0]
    if len(ok) == 0:
        return 0.5
    return float(t[ok[np.argmax(r[ok])]])


def pr_table(y, s):
    th = np.linspace(0, 1, 101)
    P, R = [], []
    for t in th:
        pred = s >= t
        tp = int((pred & (y == 1)).sum())
        P.append(tp / pred.sum() if pred.sum() else 1.0)
        R.append(tp / max(int((y == 1).sum()), 1))
    return {"thr": th.round(2).tolist(), "precision": np.round(P, 4).tolist(), "recall": np.round(R, 4).tolist()}
