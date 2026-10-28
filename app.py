"""Phish or Legit - Streamlit interface. Run: streamlit run app.py"""
import html
import json
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.nlp_utils import contributions, preprocess  # noqa: F401  (needed to load the saved models)

ROOT = Path(__file__).parent
NAMES = {"lr": "Logistic regression", "nb": "Multinomial Naive Bayes"}
NOTICE = ("This tool supports a human decision and does not block messages by itself. "
        "The data is synthetic.")

st.set_page_config(page_title="Phish or Legit", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1150px;}
#MainMenu, footer {visibility: hidden;}
.notice {border: 1px solid #D5DBE3; background: #F4F6F9; padding: .6rem .9rem; border-radius: 4px;
        font-size: .9rem; color: #3E4C59; margin-bottom: 1.2rem;}
.verdict {padding: .9rem 1.1rem; border-radius: 4px; border-left: 5px solid; margin-bottom: .8rem;}
.verdict.phish {background: #FBEDEC; border-color: #B3443B;}
.verdict.legit {background: #EDF5F0; border-color: #3F7D58;}
.verdict span {font-size: .9rem; color: #52606D;}
.msg {border: 1px solid #D5DBE3; border-radius: 4px; padding: .8rem 1rem; line-height: 1.7; background: #FFFFFF;}
.msg .pos {background: rgba(179, 68, 59, .20); border-radius: 3px; padding: 0 2px;}
.msg .neg {background: rgba(63, 125, 88, .22); border-radius: 3px; padding: 0 2px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models")
def load_artifacts():
    models = {p.stem: joblib.load(p) for p in sorted((ROOT / "models").glob("*.joblib"))}
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    examples = json.loads((ROOT / "models" / "examples.json").read_text())
    return models, meta, examples


models, meta, examples = load_artifacts()


def score(model_key, text):
    return float(models[model_key].predict_proba([text])[0][1])


def highlight(text, contrib):
    uni = {k: v for k, v in contrib.items() if " " not in k}
    peak = max((abs(v) for v in uni.values()), default=0.0)
    cut = max(0.05, 0.25 * peak)
    parts = []
    for tok in re.findall(r"\w+|\W+", text):
        w = uni.get(tok.lower(), 0.0)
        safe = html.escape(tok)
        if re.match(r"\w", tok) and abs(w) >= cut:
            parts.append(f'<span class="{"pos" if w > 0 else "neg"}">{safe}</span>')
        else:
            parts.append(safe)
    return "".join(parts).replace("\n", "<br>")


with st.sidebar:
    st.subheader("Settings")
    key = st.radio("Classifier", list(NAMES), format_func=NAMES.get)
    default_thr = float(meta["thresholds"][key])
    thr = st.slider("Decision threshold", 0.0, 1.0, default_thr, 0.01, key=f"thr_{key}")
    st.caption(f"Default for this model: {default_thr:.2f}, chosen on training data with the policy "
            f"\"{meta['policy']}\". Raising the threshold increases precision and lowers recall.")

st.title("Phish or Legit")
st.caption("Text classification with TF-IDF features, logistic regression and Naive Bayes.")
st.markdown(f'<div class="notice">{NOTICE} Training data combines the course messages with public datasets.</div>',
            unsafe_allow_html=True)

tab_cls, tab_batch, tab_perf, tab_about = st.tabs(["Classify", "Batch", "Performance", "About"])

with tab_cls:
    names = ["Custom message"] + [e["name"] for e in examples]
    st.session_state.setdefault("msg", examples[0]["text"] if examples else "")
    st.session_state.setdefault("analysed", None)

    def on_example():
        sel = st.session_state.example
        if sel != "Custom message":
            st.session_state.msg = next(e["text"] for e in examples if e["name"] == sel)
            st.session_state.analysed = None

    left, right = st.columns(2, gap="large")
    with left:
        st.selectbox("Example messages", names, key="example", on_change=on_example)
        st.text_area("Message", key="msg", height=190)
        if st.button("Classify", type="primary"):
            if st.session_state.msg.strip():
                st.session_state.analysed = st.session_state.msg
            else:
                st.warning("The message is empty. Paste some text to classify.")

    text = st.session_state.analysed
    with right:
        if text:
            raw, cal = score(key, text), score(f"{key}_cal", text)
            flagged = raw >= thr
            label = "Flagged as phishing" if flagged else "Considered legitimate"
            st.markdown(f'<div class="verdict {"phish" if flagged else "legit"}"><b>{label}</b><br>'
                        f'<span>Score {raw:.3f} against a threshold of {thr:.2f}</span></div>',
                        unsafe_allow_html=True)
            st.progress(min(max(raw, 0.0), 1.0))
            c1, c2 = st.columns(2)
            c1.metric("Classifier score", f"{raw:.3f}")
            c2.metric("Calibrated probability", f"{cal:.3f}")
            if key == "nb":
                st.caption("Naive Bayes scores are usually over-confident. Read the calibrated value as the "
                        "probability estimate.")
        else:
            st.info("Pick an example or paste a message, then press Classify.")

    if text:
        contrib = contributions(models[key], text)
        st.markdown("##### Words that influenced the decision")
        st.markdown(f'<div class="msg">{highlight(text, contrib)}</div>', unsafe_allow_html=True)
        st.caption("Red pushes towards phishing, green towards legitimate. Only strong contributions are marked.")
        if contrib:
            up = [(f, round(v, 3)) for f, v in contrib.items() if v > 0][:8]
            down = [(f, round(v, 3)) for f, v in contrib.items() if v < 0][:8]
            a, b = st.columns(2)
            a.markdown("**Features that increased the score**")
            a.dataframe(pd.DataFrame(up, columns=["Feature", "Contribution"]), hide_index=True,
                        use_container_width=True)
            b.markdown("**Features that decreased the score**")
            b.dataframe(pd.DataFrame(down, columns=["Feature", "Contribution"]), hide_index=True,
                        use_container_width=True)
            unit = "log-odds" if key == "lr" else "log-likelihood ratio"
            st.caption(f"Contribution = TF-IDF value x model weight ({unit}). Word and word-pair features only.")
        else:
            st.info("None of the words in this message are known to the model.")

with tab_batch:
    st.write("Score many messages at once with the selected classifier and threshold.")
    up = st.file_uploader("CSV file with one message per row", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        col = st.selectbox("Column containing the text", list(data.columns))
        if st.button("Score messages"):
            texts = data[col].fillna("").astype(str).tolist()[:5000]
            out = data.iloc[: len(texts)].copy()
            out["score"] = models[key].predict_proba(texts)[:, 1].round(4)
            st.session_state.batch = out
    if "batch" in st.session_state:
        out = st.session_state.batch
        hits = out[out["score"] >= thr]
        m1, m2 = st.columns(2)
        m1.metric("Messages scored", len(out))
        m2.metric("Flagged for review", len(hits))
        st.dataframe(hits.sort_values("score", ascending=False), hide_index=True, use_container_width=True)
        st.download_button("Download flagged messages", hits.to_csv(index=False).encode("utf-8"),
                        "flagged_messages.csv", "text/csv")
        st.caption("Flagged messages are suggestions for a human reviewer. Nothing is blocked or deleted.")

with tab_perf:
    st.markdown("##### Final results")
    st.dataframe(pd.DataFrame(meta["results"]), hide_index=True, use_container_width=True)
    st.caption("Reported once on test data that was never used to choose parameters or thresholds. "
            "Synthetic test: the course test split. Real test: held-out public messages.")
    ds = st.selectbox("Precision and recall against the threshold", ["synthetic", "real"],
                    format_func=lambda s: f"{s.capitalize()} test set")
    curve = pd.DataFrame(meta["curves"][key][ds]).set_index("thr")
    st.line_chart(curve[["precision", "recall"]])
    row = curve.iloc[int(round(thr * 100))]
    p1, p2 = st.columns(2)
    p1.metric(f"Precision at {thr:.2f}", f"{row['precision']:.2f}")
    p2.metric(f"Recall at {thr:.2f}", f"{row['recall']:.2f}")
    with st.expander("Effect of each improvement (threshold 0.5)"):
        st.dataframe(pd.DataFrame(meta["ablation"]), hide_index=True, use_container_width=True)

with tab_about:
    st.markdown(f"""
**Method.** Messages are normalised (links, e-mail addresses, amounts and numbers become placeholder tokens),
converted to TF-IDF vectors over word 1-2 grams and character 2-5 grams, then scored by logistic regression or
Multinomial Naive Bayes. Training used {meta['train_size']:,} messages, including augmented phishing examples
and class balancing.

**Threshold policy.** The default threshold follows the rule "{meta['policy']}", selected on out-of-fold
predictions over training data.

**Limits.**
- The course data is synthetic with fictional brands, and part of the training data is public spam and phishing
text. Real phishing changes constantly, so real-world performance would be lower.
- Once the discriminative words are known, an attacker can avoid them. The model needs regular retraining.
- Both error types matter: false positives affect legitimate senders, false negatives affect victims.
- Do not paste personal messages containing names or phone numbers.
""")