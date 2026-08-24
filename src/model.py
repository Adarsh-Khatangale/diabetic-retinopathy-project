import torch
import torch.nn as nn
from torchvision import models


class RetinopathyModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.model = models.resnet18(weights="DEFAULT")

        # Replace the final layer for our 5 DR classes
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = RetinopathyModel(num_classes=5)

    print(model)
    print("Model created successfully!")