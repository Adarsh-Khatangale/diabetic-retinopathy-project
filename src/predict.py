import os
import sys

import torch
from PIL import Image

from torchvision import transforms

from model import RetinopathyModel


# -----------------------------
# Class names
# -----------------------------
CLASS_NAMES = {
    0: "No Diabetic Retinopathy",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy",
}


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pth"
)


# -----------------------------
# Device
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Load model
# -----------------------------
model = RetinopathyModel(num_classes=5)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")


# -----------------------------
# Prediction function
# -----------------------------
def predict_image(image_path):

    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        return

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = predicted.item()

    confidence_value = confidence.item() * 100

    print()
    print("Prediction")
    print("-----------------------------")
    print("Class:", predicted_class)
    print("Diagnosis:", CLASS_NAMES[predicted_class])
    print(f"Confidence: {confidence_value:.2f}%")
    print("-----------------------------")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    image_path = input("Enter image path: ").strip()

    if not image_path:
        print("No image path entered.")
    else:
        predict_image(image_path)