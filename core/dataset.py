import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, labels_csv=None, flip_labels=False):
        self.image_dir = image_dir
        self.transform = transform
        self.flip_labels = flip_labels
        self.use_csv = labels_csv is not None

        if self.use_csv:
            self.df = pd.read_csv(labels_csv)
            self.df['label'] = self.df['label'].astype(int)
        else:
            self.filenames = sorted(os.listdir(image_dir))
        # print(self.__len__())


    def __len__(self):
        return len(self.df) if self.use_csv else len(self.filenames)

    def __getitem__(self, idx):
        if self.use_csv:
            row = self.df.iloc[idx]
            fname = row['image']
            label = int(row['label'])
        else:
            fname = self.filenames[idx]
            try:
                label = int(fname.split('_')[-1].split('.')[0])
            except ValueError:
                raise ValueError(f"Failed to parse label from filename: {fname}")

        if self.flip_labels:
            label = 1 - label

        img_path = os.path.join(self.image_dir, fname)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} not found.")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # print(f"_\t{label} ({type(label)})\t{fname} ({type(fname)})")

        return image, label, fname