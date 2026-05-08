"""
Image Model 1 — ResNet50 (Member 3)
=====================================
Pretrained ResNet50 (ImageNet) with fine-tuned classifier head for scam detection.
Freezes all backbone layers, only trains the final FC layer + last residual block.

Input : product image (path)
Output: scam probability 0.0 – 1.0

Install dependencies:
    pip install torch torchvision scikit-learn pandas Pillow
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH   = os.path.join(BASE_DIR, "Data", "raw_data", "listings.csv")
IMG_DIR     = os.path.join(BASE_DIR, "Data", "raw_data")
MODEL_SAVE  = os.path.join(BASE_DIR, "Models", "image", "resnet50_finetuned.pth")

IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS      = 5
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ImageNet normalization (required for pretrained ResNet)
TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

TRANSFORM_EVAL = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(IMG_DIR, self.image_paths[idx].replace("\\", os.sep))
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Return blank image if file missing
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last residual block (layer4) for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final FC with binary classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )
    return model


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    print("\n[ResNet50] Loading data...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    # Keep only rows with images
    df = df[df["image_paths"].notna() & (df["image_paths"] != "")].reset_index(drop=True)
    print(f"  Rows with images: {len(df)}")

    # Use first image path per row
    df["img"] = df["image_paths"].apply(lambda x: str(x).split(",")[0].strip())

    img_paths = df["img"].tolist()
    labels    = df["label"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        img_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    train_ds = ImageDataset(X_train, y_train, TRANSFORM_TRAIN)
    test_ds  = ImageDataset(X_test,  y_test,  TRANSFORM_EVAL)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, labels_batch in train_loader:
            images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

    # ── Evaluation ──
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels_batch in test_loader:
            outputs = model(images.to(DEVICE))
            preds   = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n[ResNet50] Test Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["Trusted", "Scam"]))

    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"[ResNet50] Model saved to {MODEL_SAVE}")


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(image_path: str) -> float:
    """
    Returns scam probability (0.0 = trusted, 1.0 = scam).
    image_path: local file path OR URL to product image.
    """
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE))
    model.eval()

    # Load image from path or URL
    if image_path.startswith("http"):
        import requests
        from io import BytesIO
        resp = requests.get(image_path, timeout=10)
        img  = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")

    tensor = TRANSFORM_EVAL(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits    = model(tensor)
        probs     = torch.softmax(logits, dim=1)
        scam_prob = probs[0][1].item()

    return round(scam_prob, 4)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()

    print("\n--- Inference Test ---")
    # Test with first image in dataset
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df = df[df["image_paths"].notna()].reset_index(drop=True)
    if len(df) > 0:
        test_img = os.path.join(IMG_DIR, df["image_paths"].iloc[0].split(",")[0].strip())
        score    = predict(test_img)
        print(f"Sample scam score: {score}")
