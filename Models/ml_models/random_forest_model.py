"""
ML Model 1 — Random Forest
===========================
Run:  python random_forest_model.py
Saves: Models/ml_models/saved/random_forest_pipeline.pkl
       Models/ml_models/saved/rf_phone_encoder.pkl

Install: pip install scikit-learn joblib pandas numpy
"""

import os, re, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble        import RandomForestClassifier
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.impute          import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import classification_report, accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH    = os.path.join(BASE_DIR, "Data", "raw_data", "listings(Cleaned).csv")
PRICES_PATH  = os.path.join(BASE_DIR, "Data", "raw_data", "prices.csv")
SAVE_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
MODEL_SAVE   = os.path.join(SAVE_DIR, "random_forest_pipeline.pkl")
ENCODER_SAVE = os.path.join(SAVE_DIR, "rf_phone_encoder.pkl")

os.makedirs(SAVE_DIR, exist_ok=True)


# ── Load market prices ────────────────────────────────────────────────────────
def load_prices() -> dict:
    if not os.path.exists(PRICES_PATH):
        print("[WARNING] prices.csv not found — price_gap_pct will be 0 for all rows.")
        return {}
    df = pd.read_csv(PRICES_PATH)
    prices = {}
    for _, row in df.iterrows():
        try:
            mn = float(row["price_min"])
            mx = float(row["price_max"])
            # Skip corrupted rows (price concatenated with row id, e.g. 1000011)
            if mn > 500000 or mx > 500000:
                continue
            avg = (mn + mx) / 2
            if avg > 0:
                prices[str(row["phone_model"]).strip().lower()] = avg
        except Exception:
            pass
    print(f"[Prices] Loaded {len(prices)} valid market price entries.")
    return prices


# ── Feature engineering ───────────────────────────────────────────────────────
def is_arabic(text: str) -> int:
    return 1 if re.search(r'[؀-ۿ]', str(text)) else 0


def price_gap(price, model_name, market) -> float:
    try:
        p = float(str(price).replace(",", ""))
    except Exception:
        return 0.0
    key = str(model_name).strip().lower()
    if key in market and market[key] > 0 and p > 0:
        return round((p - market[key]) / market[key], 4)
    return 0.0


def build_features(df: pd.DataFrame, market: dict, encoder: LabelEncoder,
                   fit_encoder: bool = False) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    # --- Numeric ---
    feat["price_listed"]  = pd.to_numeric(df["price_listed"], errors="coerce").fillna(0)
    feat["seller_rating"] = pd.to_numeric(df["seller_rating"], errors="coerce").fillna(0)

    # Price gap vs market average (uses external prices.csv only — no label info)
    feat["price_gap_pct"] = [
        price_gap(r["price_listed"], r["phone_model"], market)
        for _, r in df.iterrows()
    ]

    # --- Text-derived ---
    feat["title_len"]       = df["title"].fillna("").str.len()
    feat["desc_len"]        = df["description"].fillna("").str.len()
    feat["has_description"] = (feat["desc_len"] > 10).astype(int)
    feat["is_arabic"]       = df["title"].apply(is_arabic)
    feat["has_image"]       = df["image_paths"].fillna("").apply(
                                  lambda x: 1 if str(x).strip() else 0)
    feat["has_rating"]      = (feat["seller_rating"] > 0).astype(int)

    # NOTE: source_site is intentionally EXCLUDED.
    # In our dataset OLX=scam and Jumia/Amazon=trusted by construction,
    # so including it would give the model a direct shortcut to the label
    # and make it useless on real-world links from any source.

    # --- Phone model label encoding ---
    models_raw = df["phone_model"].fillna("unknown").astype(str).str.strip()
    if fit_encoder:
        feat["phone_model_enc"] = encoder.fit_transform(models_raw)
    else:
        known = set(encoder.classes_)
        safe  = models_raw.apply(lambda x: x if x in known else "unknown")
        if "unknown" not in known:
            encoder.classes_ = np.append(encoder.classes_, "unknown")
        feat["phone_model_enc"] = encoder.transform(safe)

    return feat


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    print("=" * 55)
    print("RANDOM FOREST — TRAINING")
    print("=" * 55)

    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    print(f"Loaded {len(df)} rows | Label dist: {dict(df['label'].value_counts())}")

    market  = load_prices()
    encoder = LabelEncoder()

    X_all = build_features(df, market, encoder, fit_encoder=True)
    y_all = df["label"].astype(int)

    # Stratified split — preserves 50/50 balance in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Features: {list(X_all.columns)}\n")

    # Pipeline: imputer → scaler → Random Forest
    rf_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("rf",      RandomForestClassifier(
            n_estimators      = 300,
            max_depth         = 8,
            min_samples_split = 4,
            min_samples_leaf  = 2,
            max_features      = "sqrt",
            class_weight      = "balanced",
            random_state      = 42,
            n_jobs            = -1,
        )),
    ])

    # 5-fold cross-validation on training set only
    print("Running 5-fold cross-validation...")
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_pipeline, X_train, y_train,
                                cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit on full training set
    rf_pipeline.fit(X_train, y_train)

    # Evaluation on held-out test set
    y_pred  = rf_pipeline.predict(X_test)
    y_proba = rf_pipeline.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Test ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Trusted", "Scam"]))

    importances = sorted(
        zip(X_all.columns, rf_pipeline.named_steps["rf"].feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("Feature Importances:")
    for name, imp in importances:
        bar = "█" * int(imp * 50)
        print(f"  {name:<22} {imp:.4f}  {bar}")

    joblib.dump(rf_pipeline, MODEL_SAVE)
    joblib.dump(encoder,     ENCODER_SAVE)
    print(f"\n✅ Saved pipeline → {MODEL_SAVE}")
    print(f"✅ Saved encoder  → {ENCODER_SAVE}")


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(title: str, description: str, price, phone_model: str,
            image_path: str = "", seller_rating=None) -> float:
    """Returns scam probability 0.0–1.0. Call from fusion model / FastAPI."""
    pipeline = joblib.load(MODEL_SAVE)
    encoder  = joblib.load(ENCODER_SAVE)
    market   = load_prices()

    row = pd.DataFrame([{
        "price_listed":  price,
        "phone_model":   phone_model,
        "title":         title,
        "description":   description,
        "image_paths":   image_path,
        "seller_rating": seller_rating,
    }])

    X    = build_features(row, market, encoder, fit_encoder=False)
    prob = pipeline.predict_proba(X)[0][1]
    return round(float(prob), 4)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()

    print("\n--- Inference sanity check ---")
    s1 = predict("iPhone 15 Pro Max 256GB brand new sealed",
                 "Zero cases warranty 1 year", 65000, "iPhone 15 Pro Max",
                 seller_rating=4.8)
    s2 = predict("ايفون للبيع جديد زيرو", "ايفون 15 برو كسر زيرو",
                 5000, "iPhone 15 Pro Max", seller_rating=None)
    print(f"Trusted-looking    → scam score: {s1}")
    print(f"Suspicious-looking → scam score: {s2}")
