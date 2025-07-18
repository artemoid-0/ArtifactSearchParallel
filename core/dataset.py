from torch.utils.data import Dataset
from PIL import Image
import os


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, flip_labels=False):
        self.image_dir = image_dir
        self.filenames = sorted(os.listdir(image_dir))
        self.transform = transform
        self.flip_labels = flip_labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        try:
            label = int(fname.split('_')[-1].split('.')[0])
        except ValueError:
            raise ValueError(f"Failed to parse label from filename: {fname}")

        if self.flip_labels:
            label = 1 - label

        img_path = os.path.join(self.image_dir, fname)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, fname
