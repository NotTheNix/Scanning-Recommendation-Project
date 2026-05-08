"""
Text Model 2 — LSTM + GloVe Embeddings (Member 2)
===================================================
Pretrained GloVe word embeddings + Bidirectional LSTM for scam detection.
Handles both Arabic and English (GloVe covers English; Arabic tokens get random init).

Input : title + description (text)
Output: scam probability 0.0 – 1.0

Install dependencies:
    pip install torch scikit-learn pandas numpy

Download GloVe embeddings (run once):
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
    # Place glove.6B.100d.txt in Models/text/
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH    = os.path.join(BASE_DIR, "Data", "raw_data", "listings.csv")
GLOVE_PATH   = os.path.join(BASE_DIR, "Models", "text", "glove.6B.100d.txt")
MODEL_SAVE   = os.path.join(BASE_DIR, "Models", "text", "lstm_model.pth")
VOCAB_SAVE   = os.path.join(BASE_DIR, "Models", "text", "lstm_vocab.npy")

EMBED_DIM    = 100      # must match GloVe file (glove.6B.100d = 100 dims)
HIDDEN_DIM   = 128
NUM_LAYERS   = 2
MAX_LEN      = 64       # max tokens per sample
BATCH_SIZE   = 16
EPOCHS       = 10
LR           = 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


# ── Text preprocessing ────────────────────────────────────────────────────────
def tokenize(text: str) -> list:
    text = str(text).lower()
    text = re.sub(r"[^\w\s؀-ۿ]", " ", text)  # keep Arabic chars
    return text.split()


def build_vocab(texts: list) -> dict:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def load_glove(vocab: dict) -> np.ndarray:
    """Load GloVe vectors for words in vocab. Unknown words get random init."""
    embed_matrix = np.random.uniform(-0.1, 0.1, (len(vocab), EMBED_DIM)).astype(np.float32)
    embed_matrix[0] = 0  # PAD = zeros

    if not os.path.exists(GLOVE_PATH):
        print(f"  [WARNING] GloVe file not found at {GLOVE_PATH}")
        print("  Download: http://nlp.stanford.edu/data/glove.6B.zip")
        print("  Using random embeddings instead.")
        return embed_matrix

    print(f"  Loading GloVe from {GLOVE_PATH}...")
    found = 0
    with open(GLOVE_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word  = parts[0]
            if word in vocab:
                embed_matrix[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1
    print(f"  Loaded {found}/{len(vocab)} words from GloVe")
    return embed_matrix


def encode(text: str, vocab: dict) -> list:
    tokens = tokenize(text)[:MAX_LEN]
    ids    = [vocab.get(t, 1) for t in tokens]  # 1 = UNK
    ids   += [0] * (MAX_LEN - len(ids))          # 0 = PAD
    return ids


# ── Dataset ───────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.encodings = [encode(t, vocab) for t in texts]
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx],    dtype=torch.long),
        )


# ── Model ─────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embed_matrix), freeze=False, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.dropout = nn.Dropout(0.4)
        self.fc      = nn.Linear(HIDDEN_DIM * 2, 2)  # *2 for bidirectional

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        out, (hn, _) = self.lstm(embedded)
        # Concat last hidden state from both directions
        hidden = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.fc(self.dropout(hidden))


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    print("\n[LSTM] Loading data...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df["text"] = (df["title"].fillna("") + " " + df["description"].fillna("")).str.strip()
    df = df[df["text"].str.len() > 3].reset_index(drop=True)

    texts  = df["text"].tolist()
    labels = df["label"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    print("[LSTM] Building vocabulary...")
    vocab = build_vocab(X_train)
    np.save(VOCAB_SAVE, vocab)
    print(f"  Vocab size: {len(vocab)}")

    embed_matrix = load_glove(vocab)

    train_ds = TextDataset(X_train, y_train, vocab)
    test_ds  = TextDataset(X_test,  y_test,  vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model     = LSTMClassifier(len(vocab), embed_matrix).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

    # ── Evaluation ──
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(DEVICE))
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n[LSTM] Test Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["Trusted", "Scam"]))

    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"[LSTM] Model saved to {MODEL_SAVE}")


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(title: str, description: str = "") -> float:
    """
    Returns scam probability (0.0 = trusted, 1.0 = scam).
    Call this from the backend/fusion model.
    """
    vocab        = np.load(VOCAB_SAVE, allow_pickle=True).item()
    embed_matrix = np.random.uniform(-0.1, 0.1, (len(vocab), EMBED_DIM)).astype(np.float32)

    model = LSTMClassifier(len(vocab), embed_matrix).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE))
    model.eval()

    text    = f"{title} {description}".strip()
    encoded = torch.tensor([encode(text, vocab)], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits    = model(encoded)
        probs     = torch.softmax(logits, dim=1)
        scam_prob = probs[0][1].item()

    return round(scam_prob, 4)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()

    print("\n--- Inference Test ---")
    score = predict("iPhone 15 Pro Max 256GB brand new", "Zero cases warranty included")
    print(f"Sample scam score: {score}")
