import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from preprocess import RetinopathyDataset, get_transforms
from model import RetinopathyModel


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_FILE = os.path.join(BASE_DIR, "dataset", "train.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "colored_images")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# -----------------------------
# Transforms
# -----------------------------
train_transform, val_transform = get_transforms()


# -----------------------------
# Dataset
# -----------------------------
full_dataset = RetinopathyDataset(
    CSV_FILE,
    IMAGE_DIR,
    transform=train_transform
)

print("Total images:", len(full_dataset))


# -----------------------------
# Train / validation split
# -----------------------------
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# -----------------------------
# DataLoaders
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)



val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# -----------------------------
# Model
# -----------------------------
model = RetinopathyModel(num_classes=5)
model = model.to(device)


# -----------------------------
# Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)


# -----------------------------
# Training
# -----------------------------
epochs = 5
best_val_loss = float("inf")

for epoch in range(epochs):

    # ---- Training ----
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        
        
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = 100 * train_correct / train_total


    # ---- Validation ----
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    average_train_loss = train_loss / len(train_loader)
    average_val_loss = val_loss / len(val_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {average_train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {average_val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )


    # ---- Save best model ----
    if average_val_loss < best_val_loss:

        best_val_loss = average_val_loss

        model_path = os.path.join(
            MODEL_DIR,
            "best_model.pth"
        )

        torch.save(model.state_dict(), model_path)

        print("Best model saved!")


print()
print("Training completed!")
print("Model saved in:", MODEL_DIR)