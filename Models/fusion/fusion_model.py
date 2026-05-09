"""
Fusion Model — Weighted Average of 3 Models (one per modality)
===============================================================
Combines scores from:
  • Text    : LSTM       (40% weight)
  • Image   : ResNet50   (35% weight)
  • Tabular : XGBoost    (25% weight)

Total = 100%

Run:  python fusion_model.py          (sanity check on two listings)
Use:  from fusion_model import predict (returns scam probability 0.0–1.0)

Saves: Models/fusion/saved/fusion_weights.json  (weight config)

Install: pip install torch torchvision scikit-learn xgboost joblib pandas Pillow
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "Data", "raw_data", "listings(Cleaned).csv")
IMG_DIR     = os.path.join(BASE_DIR, "Data", "raw_data")
SAVE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
WEIGHTS_PATH = os.path.join(SAVE_DIR, "fusion_weights.json")

os.makedirs(SAVE_DIR, exist_ok=True)

# Add model directories to path so we can import predict() from each model
sys.path.insert(0, os.path.join(MODELS_DIR, "text_models"))
sys.path.insert(0, os.path.join(MODELS_DIR, "image_models"))
sys.path.insert(0, os.path.join(MODELS_DIR, "ml_models"))

# ── Weights ───────────────────────────────────────────────────────────────────
# Must sum to 1.0
WEIGHTS = {
    "lstm":    0.40,
    "resnet50": 0.35,
    "xgboost": 0.25,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"


# ── Lazy model imports (avoid loading GPU models at import time) ──────────────
def _get_lstm_score(title, description):
    import lstm_model as lstm
    return lstm.predict(title, description)


def _get_resnet50_score(image_path):
    import resnet50_model as res
    return res.predict(image_path)


def _get_xgb_score(title, description, price, phone_model, image_path, seller_rating):
    import xgboost_model as xgb
    return xgb.predict(title, description, price, phone_model, image_path, seller_rating)


# ── Core predict function ─────────────────────────────────────────────────────
def predict(
    title:        str,
    description:  str   = "",
    price               = None,
    phone_model:  str   = "unknown",
    image_path:   str   = "",
    seller_rating       = None,
    verbose:      bool  = False,
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
            "lstm": 0.xx, "resnet50": 0.xx, "xgboost": 0.xx
        },
        "weights": { ... },
        "final_score": 0.xx,       # weighted average
        "verdict": "Scam" | "Suspicious" | "Trusted",
        "risk_level": "High" | "Medium" | "Low"
    }
    """
    scores = {}

    # ── Text score ──
    try:
        scores["lstm"] = _get_lstm_score(title, description)
    except Exception as e:
        print(f"[WARN] LSTM failed: {e}")
        scores["lstm"] = 0.5

    # ── Image score ──
    if image_path:
        try:
            scores["resnet50"] = _get_resnet50_score(image_path)
        except Exception as e:
            print(f"[WARN] ResNet50 failed: {e}")
            scores["resnet50"] = 0.5
    else:
        # No image provided — use neutral score so image weight doesn't bias result
        scores["resnet50"] = 0.5

    # ── Tabular score ──
    try:
        scores["xgboost"] = _get_xgb_score(
            title, description, price, phone_model, image_path, seller_rating
        )
    except Exception as e:
        print(f"[WARN] XGBoost failed: {e}")
        scores["xgboost"] = 0.5

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
        print("FUSION MODEL — SCORE BREAKDOWN")
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
    Runs the fusion model on the full dataset and reports accuracy + ROC-AUC.
    This is slow (runs all 3 models on every row) — use a subset for quick checks.
    """
    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

    print("=" * 55)
    print("FUSION MODEL — DATASET EVALUATION")
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
            title        = str(row.get("title", "")),
            description  = str(row.get("description", "")),
            price        = row.get("price_listed"),
            phone_model  = str(row.get("phone_model", "unknown")),
            image_path   = image_path,
            seller_rating= row.get("seller_rating"),
        )

        all_probs.append(result["final_score"])
        all_preds.append(1 if result["final_score"] >= 0.65 else 0)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(df)} rows...")

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\nFusion Accuracy : {acc:.4f}")
    print(f"Fusion ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Trusted", "Scam"]))

    # Save weights config
    json.dump(WEIGHTS, open(WEIGHTS_PATH, "w"), indent=2)
    print(f"\n✅ Saved weights → {WEIGHTS_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick sanity check — two listings
    print("FUSION MODEL — SANITY CHECK")
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
