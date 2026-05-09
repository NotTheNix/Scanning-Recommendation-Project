"""
Text Model 1 — Bidirectional LSTM
===================================
Run:  python lstm_model.py
Saves: Models/text_models/saved/lstm_model.pth
       Models/text_models/saved/lstm_vocab.pkl

Install: pip install torch scikit-learn pandas numpy joblib
"""

import os, re, warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data    import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics     import classification_report, accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH   = os.path.join(BASE_DIR, "Data", "raw_data", "listings(Cleaned).csv")
SAVE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
MODEL_SAVE  = os.path.join(SAVE_DIR, "lstm_model.pth")
VOCAB_SAVE  = os.path.join(SAVE_DIR, "lstm_vocab.pkl")

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_VOCAB   = 10000   # keep top N most frequent words
MAX_LEN     = 64      # max tokens per sample
EMBED_DIM   = 64      # embedding size (learned from scratch)
HIDDEN_DIM  = 128
NUM_LAYERS  = 2
DROPOUT     = 0.4
BATCH_SIZE  = 32
EPOCHS      = 15
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


# ── Text preprocessing ────────────────────────────────────────────────────────
def clean(text: str) -> str:
    text = str(text).lower().strip()
    # Keep Arabic, English letters, digits — remove everything else
    text = re.sub(r"[^\w\s؀-ۿ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list:
    return clean(text).split()


def build_vocab(texts: list) -> dict:
    from collections import Counter
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(MAX_VOCAB - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: dict) -> list:
    tokens = tokenize(text)[:MAX_LEN]
    ids    = [vocab.get(t, 1) for t in tokens]      # 1 = UNK
    ids   += [0] * (MAX_LEN - len(ids))              # 0 = PAD
    return ids


# ── Dataset ───────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.X = [encode(t, vocab) for t in texts]
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# ── Model ─────────────────────────────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.lstm      = nn.LSTM(
            EMBED_DIM, HIDDEN_DIM,
            num_layers    = NUM_LAYERS,
            batch_first   = True,
            bidirectional = True,
            dropout       = DROPOUT if NUM_LAYERS > 1 else 0,
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.fc      = nn.Linear(HIDDEN_DIM * 2, 2)   # *2 for bidirectional

    def forward(self, x):
        emb = self.dropout(self.embedding(x))          # (B, L, E)
        _, (hn, _) = self.lstm(emb)                    # hn: (2*layers, B, H)
        # Concat last hidden from both directions
        hidden = torch.cat([hn[-2], hn[-1]], dim=1)    # (B, 2H)
        return self.fc(self.dropout(hidden))


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    print("=" * 55)
    print("LSTM — TRAINING")
    print("=" * 55)

    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    print(f"Loaded {len(df)} rows | Label dist: {dict(df['label'].value_counts())}")

    df["text"] = (df["title"].fillna("") + " " + df["description"].fillna("")).str.strip()
    df = df[df["text"].str.len() > 3].reset_index(drop=True)

    texts  = df["text"].tolist()
    labels = df["label"].astype(int).tolist()

    # Split FIRST — build vocab only on training text (prevents leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    vocab = build_vocab(X_train)
    joblib.dump(vocab, VOCAB_SAVE)
    print(f"Vocab size: {len(vocab)}")

    train_loader = DataLoader(TextDataset(X_train, y_train, vocab),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TextDataset(X_test,  y_test,  vocab),
                              batch_size=BATCH_SIZE)

    model     = BiLSTMClassifier(len(vocab)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_auc   = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # ── Eval ──
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                probs = torch.softmax(model(X_batch.to(DEVICE)), dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y_batch.numpy())

        auc = roc_auc_score(all_labels, all_probs)
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1:02d}/{EPOCHS} — Loss: {avg_loss:.4f} | ROC-AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load best checkpoint
    model.load_state_dict(best_state)
    model.eval()

    # Final evaluation
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            out   = model(X_batch.to(DEVICE))
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\nBest Test Accuracy : {acc:.4f}")
    print(f"Best Test ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Trusted", "Scam"]))

    torch.save({"model_state": best_state, "vocab_size": len(vocab)}, MODEL_SAVE)
    print(f"\n✅ Saved model → {MODEL_SAVE}")
    print(f"✅ Saved vocab → {VOCAB_SAVE}")


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(title: str, description: str = "") -> float:
    """Returns scam probability 0.0–1.0."""
    vocab      = joblib.load(VOCAB_SAVE)
    checkpoint = torch.load(MODEL_SAVE, map_location=DEVICE)
    model      = BiLSTMClassifier(checkpoint["vocab_size"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    text   = f"{title} {description}".strip()
    tensor = torch.tensor([encode(text, vocab)], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        return round(probs[0][1].item(), 4)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()

    print("\n--- Inference sanity check ---")
    s1 = predict("iPhone 15 Pro Max 256GB brand new sealed", "Zero cases warranty 1 year")
    s2 = predict("ايفون للبيع جديد زيرو", "ايفون 15 برو كسر زيرو بسعر مميز")
    print(f"Trusted-looking    → scam score: {s1}")
    print(f"Suspicious-looking → scam score: {s2}")
