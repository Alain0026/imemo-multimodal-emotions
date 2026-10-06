"""
Dataset PyTorch per IMEmo — legge da train.csv / val.csv
"""
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class IMEmoDataset(Dataset):

    # Quale colonna usare come label
    LABEL_MODES = {
        "sentiment":  ("sentiment_idx",  3),   # Positive/Negative/Neutral
        "basic":      ("basic_idx",       7),   # 6 emozioni base + Neutral come la 7°sima classe
        "fine":       ("emotion_idx",    17),   # 16 emozioni fine-grained + Neutral come la 17°sima classe
    }

    def __init__(self, csv_path, mode="sentiment", transform=None):
        """
        csv_path : percorso a train.csv o val.csv
        mode     : 'sentiment' | 'basic' | 'fine'
        transform: trasformazioni torchvision
        """
        assert mode in self.LABEL_MODES, \
            f"mode deve essere uno tra: {list(self.LABEL_MODES.keys())}"

        self.df        = pd.read_csv(csv_path)
        self.transform = transform
        self.mode      = mode

        label_col, self.num_classes = self.LABEL_MODES[mode]

        # Filtra righe con label non valida
        self.df = self.df[self.df[label_col] >= 0].reset_index(drop=True)

        self.labels = self.df[label_col].values
        self.paths  = self.df["frame_path"].values

        print(f"[IMEmoDataset] mode={mode} | "
              f"classi={self.num_classes} | "
              f"campioni={len(self.df)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label    = int(self.labels[idx])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Se il file è corrotto, restituisce immagine nera
            image = Image.new("RGB", (224, 224), 0)

        if self.transform:
            image = self.transform(image)

        return image, label
