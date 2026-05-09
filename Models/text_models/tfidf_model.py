"""
Text Model 2 — TF-IDF + Logistic Regression
=============================================
Simple, fast, and interpretable. Works well on small datasets.
No GPU needed. Trains in seconds.

Run:  python tfidf_model.py
Saves: Models/text_models/saved/tfidf_pipeline.pkl

Install: pip install scikit-learn pandas numpy joblib
"""

import os, re, warnings
import pandas as pd
import joblib

from sklearn.pipeline        import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import classification_report, accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH  = os.path.join(BASE_DIR, "Data", "raw_data", "listings(Cleaned).csv")
SAVE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
MODEL_SAVE = os.path.join(SAVE_DIR, "tfidf_pipeline.pkl")

os.makedirs(SAVE_DIR, exist_ok=True)


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean(text: str) -> str:
    text = str(text).lower().strip()
    # Keep Arabic, English letters, digits
    text = re.sub(r"[^\w\s؀-ۿ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_text(df: pd.DataFrame) -> list:
    combined = df["title"].fillna("") + " " + df["description"].fillna("")
    return combined.apply(clean).tolist()


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    print("=" * 55)
    print("TF-IDF + LOGISTIC REGRESSION — TRAINING")
    print("=" * 55)

    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    print(f"Loaded {len(df)} rows | Label dist: {dict(df['label'].value_counts())}")

    texts  = prepare_text(df)
    labels = df["label"].astype(int).tolist()

    # Split first — TF-IDF fit only on training text (no leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # Pipeline: TF-IDF → Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features  = 8000,
            ngram_range   = (1, 2),    # unigrams + bigrams
            sublinear_tf  = True,      # log(tf) smoothing
            min_df        = 2,         # ignore very rare words
            analyzer      = "word",
            strip_accents = None,      # keep Arabic diacritics
        )),
        ("clf", LogisticRegression(
            C             = 1.0,
            class_weight  = "balanced",
            max_iter      = 1000,
            random_state  = 42,
            solver        = "lbfgs"
        )),
    ])

    # 5-fold CV on training set
    print("Running 5-fold cross-validation...")
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                cv=cv, scoring="roc_auc")
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Test ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Trusted", "Scam"]))

    # Top words per class
    tfidf  = pipeline.named_steps["tfidf"]
    clf    = pipeline.named_steps["clf"]
    names  = tfidf.get_feature_names_out()
    coefs  = clf.coef_[0]

    top_scam    = [names[i] for i in coefs.argsort()[-10:][::-1]]
    top_trusted = [names[i] for i in coefs.argsort()[:10]]
    print(f"\nTop scam words:    {top_scam}")
    print(f"Top trusted words: {top_trusted}")

    joblib.dump(pipeline, MODEL_SAVE)
    print(f"\n✅ Saved pipeline → {MODEL_SAVE}")


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(title: str, description: str = "") -> float:
    """Returns scam probability 0.0–1.0."""
    pipeline = joblib.load(MODEL_SAVE)
    text     = re.sub(r"\s+", " ", f"{title} {description}".lower().strip())
    prob     = pipeline.predict_proba([text])[0][1]
    return round(float(prob), 4)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()

    print("\n--- Inference sanity check ---")
    s1 = predict("iPhone 15 Pro Max 256GB brand new sealed", "Zero cases warranty 1 year")
    s2 = predict("ايفون للبيع جديد زيرو", "ايفون 15 برو كسر زيرو بسعر مميز")
    print(f"Trusted-looking    → scam score: {s1}")
    print(f"Suspicious-looking → scam score: {s2}")
