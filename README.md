# Phish or Legit

A web application that classifies a short message as phishing or legitimate, shows the classifier score, and explains each decision with per-feature contributions. Built for the NLP module (Lab 04) with scikit-learn and Streamlit.

The tool supports a human decision and does not block messages by itself. The course data is synthetic.

## Features

- Classification of a pasted message with a score and a calibrated probability
- Two classifiers: logistic regression and Multinomial Naive Bayes
- Live decision threshold slider
- Word highlighting and ranked lists of features that raised or lowered the score
- Example messages, including one the model misclassifies
- Batch mode: upload a CSV, review flagged messages, download them
- Performance tab with the results table and a precision/recall versus threshold plot

## Project structure

```
.
├── app.py                  Streamlit interface
├── notebook.ipynb          Data, training, evaluation and export
├── requirements.txt
├── .streamlit/config.toml  Light theme
├── data/messages.csv       Course dataset (2,000 synthetic messages)
├── src/
│   ├── pipeline.py         Course baseline code
│   └── nlp_utils.py        Preprocessing, augmentation, models, explanations
└── models/                 Saved pipelines, thresholds and example messages
```

## Run locally

Requires Python 3.10 or later.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Reproduce the models

Open `notebook.ipynb` and run all cells from top to bottom. The notebook rebuilds everything in `models/`. The saved models only load with the scikit-learn version pinned in `requirements.txt`.

## Method

1. Stratified train/test split before any fitting (seed 42).
2. Normalisation: links, e-mail addresses, amounts and numbers become placeholder tokens.
3. Additional public data (SMS spam collection and a public phishing text dataset from Hugging Face).
4. Augmentation of phishing training examples (word dropout and look-alike character swaps), applied to the training set only.
5. Features: TF-IDF over word 1-2 grams and character 2-5 grams.
6. Class balancing, then hyperparameter search with group-aware cross-validation so augmented copies never leak across folds.
7. Sigmoid (Platt) calibration of the scores.
8. Threshold chosen on out-of-fold training predictions from a written policy: at least 90% precision, then maximise recall.

## Results

Reported once on test sets that were not used to choose any parameter or threshold.

| Classifier | Test set | Threshold | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|
| (paste from notebook section 11) | | | | | | |

## Data and ethics

- The course dataset is synthetic with fictional brands. Public datasets are used under their own licences.
- SMS spam is not phishing, so the real-data test set measures spam or phishing-style text against legitimate text.
- No personal messages are included. Do not paste real messages containing names or phone numbers.
- False positives affect legitimate senders and false negatives affect victims. The classifier never blocks or deletes anything.

## Limitations

Real phishing changes constantly, so performance on real traffic would be lower. Once the discriminative words are known, an attacker can avoid them, so the model needs regular retraining. Calibrated probabilities refer to the training data mix, not to real-world prevalence.

## Deployment

Deployed on Streamlit Community Cloud: connect this repository on share.streamlit.io, select `app.py` as the main file, and deploy.

Public URL: (add here)