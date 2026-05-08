"""
ML Model 1 — Random Forest (Member 5)
=======================================
Built from scratch on tabular features for scam detection.
Also used as a price predictor to compute price_gap_pct.

Features used:
  - price_listed       : listed price in EGP
  - price_gap_pct      : % difference from market average (from prices.csv)
  - title_length       : number of characters in title
  - has_description    : 1 if description exists and is not empty
  - has_image          : 1 if image exists
  - has_seller_rating  : 1 if seller rating exists
  - is_arabic          : 1 if title contains Arabic characters
  - phone_model_encoded: label-encoded phone model name

Output: scam probability 0.0 – 1.0

Install dependencies:
    pip install scikit-learn pandas numpy joblib
"""

import os
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH     = os.path.join(BASE_DIR, "Data", "raw_data", "listings.csv")
PRICES_PATH   = os.path.join(BASE_DIR, "Data", "raw_data", "prices.csv")
MODEL_SAVE    = os.path.join(BASE_DIR, "Models", "ml", "random_forest.pkl")
ENCODER_SAVE  = os.path.join(BASE_DIR, "Models", "ml", "rf_label_encoder.pkl")


# ── Feature engineering ───────────────────────────────────────────────────────
def load_prices() -> dict:
    """Load market price averages from prices.csv."""
    if not os.path.exists(PRICES_PATH):
        print("  [WARNING] prices.csv not found. price_gap_pct will be 0.")
        return {}
    df = pd.read_csv(PRICES_PATH)
    prices = {}
    for _, row in df.iterrows():
        model = str(row.get("phone_model", "")).strip().lower()
        try:
            avg = (float(row["price_min"]) + float(row["price_max"])) / 2
            prices[model] = avg
        except:
            pass
    return prices


def compute_price_gap(price: float, phone_model: str, market_prices: dict) -> float:
    """How far is the listed price from market average? Negative = too cheap (suspicious)."""
    model_key = str(phone_model).strip().lower()
    if model_key in market_prices and market_prices[model_key] > 0 and price > 0:
        gap = (price - market_prices[model_key]) / market_prices[model_key]
        return round(gap, 4)
    return 0.0


def is_arabic(text: str) -> int:
    return 1 if re.search(r'[؀-ۿ]', str(text)) else 0


def build_features(df: pd.DataFrame, market_prices: dict, encoder: LabelEncoder = None, fit_encoder: bool = False):
    features = pd.DataFrame()

    # Price
    features["price_listed"] = pd.to_numeric(df["price_listed"], errors="coerce").fillna(0)

    # Price gap from market
    features["price_gap_pct"] = [
        compute_price_gap(
            float(str(row["price_listed"]).replace(",", "") or 0),
            row["phone_model"],
            market_prices
        )
        for _, row in df.iterrows()
    ]

    # Text features
    features["title_length"]    = df["title"].fillna("").str.len()
    features["has_description"] = df["description"].fillna("").apply(lambda x: 1 if len(str(x)) > 10 else 0)
    features["is_arabic"]       = df["title"].apply(is_arabic)

    # Image & seller features
    features["has_image"]         = df["image_paths"].fillna("").apply(lambda x: 1 if str(x).strip() else 0)
    features["has_seller_rating"] = df["seller_rating"].apply(lambda x: 0 if pd.isna(x) else 1)

    # Phone model — label encode
    phone_models = df["phone_model"].fillna("unknown").astype(str)
    if fit_encoder:
        encoder = LabelEncoder()
        features["phone_model_encoded"] = encoder.fit_transform(phone_models)
    else:
        # Handle unseen labels gracefully
        known = set(encoder.classes_)
        phone_models = phone_models.apply(lambda x: x if x in known else "unknown")
        if "unknown" not in known:
            encoder.classes_ = np.append(encoder.classes_, "unknown")
        features["phone_model_encoded"] = encoder.transform(phone_models)

    return features, encoder


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    print("\n[Random Forest] Loading data...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    print(f"  Total rows: {len(df)}")

    market_prices = load_prices()
    print(f"  Market prices loaded: {len(market_prices)} models")

    X, encoder = build_features(df, market_prices, fit_encoder=True)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Features: {list(X.columns)}")

    # ── Train Random Forest ──
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",  # handles imbalanced data
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Evaluation ──
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n[Random Forest] Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Trusted", "Scam"]))

    # Feature importance
    importances = sorted(
        zip(X.columns, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\nFeature Importances:")
    for feat, imp in importances:
        print(f"  {feat:<25}: {imp:.4f}")

    # ── Save ──
    joblib.dump(model,   MODEL_SAVE)
    joblib.dump(encoder, ENCODER_SAVE)
    print(f"\n[Random Forest] Model saved to {MODEL_SAVE}")


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(
    price: float,
    phone_model: str,
    title: str,
    description: str = "",
    image_path: str = "",
    seller_rating=None,
) -> float:
    """
    Returns scam probability (0.0 = trusted, 1.0 = scam).
    Call this from the backend/fusion model.
    """
    model   = joblib.load(MODEL_SAVE)
    encoder = joblib.load(ENCODER_SAVE)
    market_prices = load_prices()

    row = {
        "price_listed":  price,
        "phone_model":   phone_model,
        "title":         title,
        "description":   description,
        "image_paths":   image_path,
        "seller_rating": seller_rating,
    }
    df_row = pd.DataFrame([row])
    X, _   = build_features(df_row, market_prices, encoder=encoder, fit_encoder=False)

    probs     = model.predict_proba(X)
    scam_prob = probs[0][1]
    return round(float(scam_prob), 4)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()

    print("\n--- Inference Test ---")
    score = predict(
        price=5000,
        phone_model="iPhone 15 Pro Max",
        title="iPhone 15 Pro Max 256GB brand new",
        description="Zero cases, warranty included",
        image_path="images/trusted/abc123_0.jpg",
    )
    print(f"Sample scam score: {score}")
