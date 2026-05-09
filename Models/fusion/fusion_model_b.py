"""
Fusion Model B — Weighted Average of 3 Models (one per modality)
================================================================
Combines scores from:
  • Text    : TF-IDF + Logistic Regression  (40% weight)
  • Image   : EfficientNet-B0               (35% weight)
  • Tabular : Random Forest                 (25% weight)

Total = 100%

This is the alternative fusion sequence.
Fusion Model A (lstm + resnet50 + xgboost) is in fusion_model.py

Run:  python fusion_model_b.py
Use:  from fusion_model_b import predict

Install: pip install torch torchvision scikit-learn joblib pandas Pillow
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "Data", "raw_data", "listings(Cleaned).csv")
IMG_DIR      = os.path.join(BASE_DIR, "Data", "raw_data")
SAVE_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
WEIGHTS_PATH = os.path.join(SAVE_DIR, "fusion_weights_b.json")

os.makedirs(SAVE_DIR, exist_ok=True)

# Add model directories to path
sys.path.insert(0, os.path.join(MODELS_DIR, "text_models"))
sys.path.insert(0, os.path.join(MODELS_DIR, "image_models"))
sys.path.insert(0, os.path.join(MODELS_DIR, "ml_models"))

# ── Weights ───────────────────────────────────────────────────────────────────
WEIGHTS = {
    "tfidf":         0.40,
    "efficientnet":  0.35,
    "random_forest": 0.25,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"


# ── Lazy model imports ────────────────────────────────────────────────────────
def _get_tfidf_score(title, description):
    import tfidf_model as tfidf
    return tfidf.predict(title, description)


def _get_efficientnet_score(image_path):
    import efficientnet_model as eff
    return eff.predict(image_path)


def _get_rf_score(title, description, price, phone_model, image_path, seller_rating):
    import random_forest_model as rf
    return rf.predict(title, description, price, phone_model, image_path, seller_rating)


# ── Core predict function ─────────────────────────────────────────────────────
def predict(
    title:        str,
    description:  str  = "",
    price              = None,
    phone_model:  str  = "unknown",
    image_path:   str  = "",
    seller_rating      = None,
    verbose:      bool = False,
) -> dict:
    """
    Returns a dict with individual scores and the final fused scam probability.

    Parameters
    ----------
    title        : listing title (str)
    description  : listing description (str, optional)
    price        : listed price (numeric, optional)
    phone_model  : phone model name (str, optional)
    image_path   : local file path or http URL to listing image (str, optional)
    seller_rating: seller rating 0–5 (numeric, optional)
    verbose      : if True, prints a breakdown of all scores

    Returns
    -------
    {
        "scores": {
            "tfidf": 0.xx, "efficientnet": 0.xx, "random_forest": 0.xx
        },
        "weights": { ... },
        "final_score": 0.xx,
        "verdict": "Scam" | "Suspicious" | "Trusted",
        "risk_level": "High" | "Medium" | "Low"
    }
    """
    scores = {}

    # ── Text score ──
    try:
        scores["tfidf"] = _get_tfidf_score(title, description)
    except Exception as e:
        print(f"[WARN] TF-IDF failed: {e}")
        scores["tfidf"] = 0.5

    # ── Image score ──
    if image_path:
        try:
            scores["efficientnet"] = _get_efficientnet_score(image_path)
        except Exception as e:
            print(f"[WARN] EfficientNet failed: {e}")
            scores["efficientnet"] = 0.5
    else:
        # No image provided — neutral score so weight doesn't bias result
        scores["efficientnet"] = 0.5

    # ── Tabular score ──
    try:
        scores["random_forest"] = _get_rf_score(
            title, description, price, phone_model, image_path, seller_rating
        )
    except Exception as e:
        print(f"[WARN] Random Forest failed: {e}")
        scores["random_forest"] = 0.5

    # ── Weighted average ──
    final_score = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    final_score = round(final_score, 4)

    # ── Verdict ──
    if final_score >= 0.65:
        verdict    = "Scam"
        risk_level = "High"
    elif final_score >= 0.40:
        verdict    = "Suspicious"
        risk_level = "Medium"
    else:
        verdict    = "Trusted"
        risk_level = "Low"

    result = {
        "scores":      scores,
        "weights":     WEIGHTS,
        "final_score": final_score,
        "verdict":     verdict,
        "risk_level":  risk_level,
    }

    if verbose:
        print("\n" + "=" * 50)
        print("FUSION MODEL B — SCORE BREAKDOWN")
        print("=" * 50)
        for model_name, score in scores.items():
            w   = WEIGHTS[model_name]
            bar = "█" * int(score * 20)
            print(f"  {model_name:<15} score={score:.4f}  weight={w:.0%}  contribution={w*score:.4f}  {bar}")
        print(f"\n  Final Score : {final_score:.4f}")
        print(f"  Verdict     : {verdict}")
        print(f"  Risk Level  : {risk_level}")
        print("=" * 50)

    return result


# ── Evaluate on full dataset ──────────────────────────────────────────────────
def evaluate():
    """
    Runs Fusion Model B on the full dataset and reports accuracy + ROC-AUC.
    """
    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

    print("=" * 55)
    print("FUSION MODEL B — DATASET EVALUATION")
    print("=" * 55)

    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} rows")

    all_preds  = []
    all_probs  = []
    all_labels = df["label"].astype(int).tolist()

    for i, row in df.iterrows():
        image_path = ""
        if pd.notna(row.get("image_paths")) and str(row["image_paths"]).strip():
            first_img  = str(row["image_paths"]).split(",")[0].strip()
            image_path = os.path.join(IMG_DIR, first_img.replace("\\", os.sep))
            if not os.path.exists(image_path):
                image_path = ""

        result = predict(
            title         = str(row.get("title", "")),
            description   = str(row.get("description", "")),
            price         = row.get("price_listed"),
            phone_model   = str(row.get("phone_model", "unknown")),
            image_path    = image_path,
            seller_rating = row.get("seller_rating"),
        )

        all_probs.append(result["final_score"])
        all_preds.append(1 if result["final_score"] >= 0.65 else 0)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(df)} rows...")

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\nFusion B Accuracy : {acc:.4f}")
    print(f"Fusion B ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Trusted", "Scam"]))

    json.dump(WEIGHTS, open(WEIGHTS_PATH, "w"), indent=2)
    print(f"\n✅ Saved weights → {WEIGHTS_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("FUSION MODEL B — SANITY CHECK")
    print("=" * 55)

    r1 = predict(
        title         = "iPhone 15 Pro Max 256GB brand new sealed",
        description   = "Zero cases warranty 1 year Apple store receipt",
        price         = 65000,
        phone_model   = "iPhone 15 Pro Max",
        seller_rating = 4.8,
        verbose       = True,
    )

    r2 = predict(
        title         = "ايفون للبيع جديد زيرو",
        description   = "ايفون 15 برو كسر زيرو بسعر مميز",
        price         = 5000,
        phone_model   = "iPhone 15 Pro Max",
        seller_rating = None,
        verbose       = True,
    )

    print(f"\nTrusted-looking → {r1['final_score']} ({r1['verdict']})")
    print(f"Scam-looking    → {r2['final_score']} ({r2['verdict']})")
