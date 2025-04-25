from torchvision import transforms
from torch.utils.data import Dataset
from io import BytesIO
from PIL import Image
from src.utils.preprocessing import normalize_int8, truncate_normalization, crop_to_roi
import torch
import numpy as np
import pandas as pd


def get_augmentations():
    return [
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
    ]


class VindrMammoDataset(Dataset):
    def __init__(self, bucket, split='training', transform=None):
        self.bucket = bucket
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            *get_augmentations()
        ])

        csv_blob = self.bucket.blob("finding_annotations.csv")
        csv_data = csv_blob.download_as_text()
        annotations_df = pd.read_csv(BytesIO(csv_data.encode()))

        df = annotations_df[annotations_df['split'] == self.split]
        df = df[['image_id', 'breast_birads']].reset_index(drop=True)
        df['label'] = df['breast_birads'].apply(self.birads_to_binary)
        df = df[df['label'].isin([0, 1])]
        df_balanced = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(n=200, random_state=42)).reset_index(drop=True)
        self.df = df_balanced[['image_id', 'label']]
        self.labels = self.df['label'].values

    def birads_to_binary(self, value):
        try:
            num = int(value.split()[-1])
            return 0 if num <= 2 else 1
        except Exception as e:
            print(f"Error parsing BI-RADS value: {value} | Exception: {e}")
            return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id, label = self.df.iloc[idx]
        label = torch.tensor(label, dtype=torch.float32)
        img_path = f"images/{image_id}.png"
        blob = self.bucket.blob(img_path)
        image_data = blob.download_as_bytes()
        image = Image.open(BytesIO(image_data)).convert('L')
        cropped_image = crop_to_roi(np.array(image))[0]
        image = truncate_normalization(
            (cropped_image, crop_to_roi(np.array(image))[1]))
        image = Image.fromarray(normalize_int8(image))
        image = self.transform(image)
        return image, label
