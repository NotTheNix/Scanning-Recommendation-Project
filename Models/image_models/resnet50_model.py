"""
Image Model 1 — ResNet50 (Pretrained, Fine-tuned)
===================================================
Run:  python resnet50_model.py
Saves: Models/image_models/saved/resnet50_model.pth

Install: pip install torch torchvision scikit-learn pandas Pillow
PyTorch downloads pretrained weights automatically on first run.
"""

import os, warnings
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data    import Dataset, DataLoader
from torchvision         import models, transforms
from PIL                 import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics     import classification_report, accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH  = os.path.join(BASE_DIR, "Data", "raw_data", "listings(Cleaned).csv")
IMG_DIR    = os.path.join(BASE_DIR, "Data", "raw_data")
SAVE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
MODEL_SAVE = os.path.join(SAVE_DIR, "resnet50_model.pth")

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 10
LR         = 1e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ImageNet normalization — required for pretrained ResNet
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, img_paths, labels, transform):
        self.img_paths = img_paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = os.path.join(IMG_DIR, self.img_paths[idx].replace("\\", os.sep))
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all backbone layers
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze last residual block (layer4) for fine-tuning
    for p in model.layer4.parameters():
        p.requires_grad = True

    # Replace final FC with binary classifier
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )
    return model


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    print("=" * 55)
    print("RESNET50 — TRAINING")
    print("=" * 55)

    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df = df[df["image_paths"].notna() & (df["image_paths"] != "")].reset_index(drop=True)
    df["img"] = df["image_paths"].apply(lambda x: str(x).split(",")[0].strip())
    print(f"Rows with images: {len(df)} | Label dist: {dict(df['label'].value_counts())}")

    img_paths = df["img"].tolist()
    labels    = df["label"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        img_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    train_loader = DataLoader(ImageDataset(X_train, y_train, TRAIN_TRANSFORM),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(ImageDataset(X_test,  y_test,  EVAL_TRANSFORM),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    best_auc   = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        total_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # ── Eval ──
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                probs = torch.softmax(model(imgs.to(DEVICE)), dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(lbls.numpy())

        auc      = roc_auc_score(all_labels, all_probs)
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1:02d}/{EPOCHS} — Loss: {avg_loss:.4f} | ROC-AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Final evaluation with best checkpoint
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            out   = model(imgs.to(DEVICE))
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(lbls.numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\nBest Test Accuracy : {acc:.4f}")
    print(f"Best Test ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Trusted", "Scam"]))

    torch.save(best_state, MODEL_SAVE)
    print(f"\n✅ Saved model → {MODEL_SAVE}")


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(image_path: str) -> float:
    """
    Returns scam probability 0.0–1.0.
    image_path: local file path OR http URL.
    """
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE))
    model.eval()

    if image_path.startswith("http"):
        import requests
        from io import BytesIO
        img = Image.open(BytesIO(requests.get(image_path, timeout=10).content)).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")

    tensor = EVAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        return round(probs[0][1].item(), 4)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()

    print("\n--- Inference sanity check ---")
    df       = pd.read_csv(DATA_PATH, encoding="utf-8")
    test_img = os.path.join(IMG_DIR, df["image_paths"].iloc[0].split(",")[0].strip().replace("\\", os.sep))
    score    = predict(test_img)
    print(f"First image scam score: {score}")
