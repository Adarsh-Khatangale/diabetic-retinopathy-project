import os
import pandas as pd
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_id = str(self.data.iloc[index]["id_code"])
        label = int(self.data.iloc[index]["diagnosis"])

        

        image_files = glob.glob(
        os.path.join(self.image_dir, "**", image_id + ".*"),
        recursive=True
        )

        if not image_files:
         raise FileNotFoundError(
            f"Image not found: {image_id} inside {self.image_dir}"
        )

        image_path = image_files[0]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform