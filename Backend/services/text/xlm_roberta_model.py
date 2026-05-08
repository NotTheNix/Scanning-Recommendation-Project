"""
Text Model 1 — XLM-RoBERTa (Member 1)
=======================================
Pretrained multilingual transformer fine-tuned for scam detection.
Handles Arabic + English mixed text natively.

Input : title + description (text)
Output: scam probability 0.0 – 1.0

Install dependencies:
    pip install transformers torch scikit-learn pandas
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH   = os.path.join(BASE_DIR, "Data", "raw_data", "listings.csv")
MODEL_SAVE  = os.path.join(BASE_DIR, "Models", "text", "xlm_roberta_finetuned")

MODEL_NAME  = "xlm-roberta-base"   # pretrained multilingual model
MAX_LEN     = 128                   # max tokens per sample
BATCH_SIZE  = 8
EPOCHS      = 3
LR          = 2e-5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


# ── Dataset ───────────────────────────────────────────────────────────────────
class ListingsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── Load & prepare data ───────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    # Combine title + description as input text
    df["text"] = (
        df["title"].fillna("") + " " + df["description"].fillna("")
    ).str.strip()

    df = df[df["text"].str.len() > 3].reset_index(drop=True)

    texts  = df["text"].tolist()
    labels = df["label"].astype(int).tolist()

    return train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    print("\n[XLM-RoBERTa] Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    print("[XLM-RoBERTa] Loading pretrained tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    train_dataset = ListingsDataset(X_train, y_train, tokenizer)
    test_dataset  = ListingsDataset(X_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ── Training loop ──
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
                labels         = batch["labels"].to(DEVICE),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

    # ── Evaluation ──
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
            )
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n[XLM-RoBERTa] Test Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["Trusted", "Scam"]))

    # ── Save ──
    os.makedirs(MODEL_SAVE, exist_ok=True)
    model.save_pretrained(MODEL_SAVE)
    tokenizer.save_pretrained(MODEL_SAVE)
    print(f"[XLM-RoBERTa] Model saved to {MODEL_SAVE}")


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(title: str, description: str = "") -> float:
    """
    Returns scam probability (0.0 = trusted, 1.0 = scam).
    Call this from the backend/fusion model.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE)
    model.eval()
    model.to(DEVICE)

    text = f"{title} {description}".strip()
    enc  = tokenizer(text, return_tensors="pt", truncation=True,
                     padding="max_length", max_length=MAX_LEN)
    enc  = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=1)
        scam_prob = probs[0][1].item()

    return round(scam_prob, 4)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()

    # Quick inference test
    print("\n--- Inference Test ---")
    score = predict("iPhone 15 Pro Max 256GB brand new sealed", "Zero cases, warranty included")
    print(f"Sample scam score: {score}")
