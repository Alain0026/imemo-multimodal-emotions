"""
Dataset a livello di segmento.
Carica tutte le frame di un segmento, le processa e le mean-poola.
"""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

FACES_PATH = "data/processed/faces_cropped"

class SegmentDataset(Dataset):
    LABEL_MODES = {
        "sentiment": ("sentiment_idx", 3),
        "basic":     ("basic_idx",     6),
        "fine":      ("fine_idx",      16),
    }

    def __init__(self, csv_path, mode="sentiment", transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        label_col, self.num_classes = self.LABEL_MODES[mode]

        # Filtra segmenti con label valida
        self.df = self.df[self.df[label_col] >= 0].reset_index(drop=True)
        self.labels     = self.df[label_col].values
        self.clip_ids   = self.df["clip_id"].values
        self.frame_from = self.df["frame_from"].values
        self.frame_to   = self.df["frame_to"].values
        self.transform  = transform

        print(f"[SegmentDataset] mode={mode} | "
              f"classi={self.num_classes} | "
              f"segmenti={len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        clip      = str(self.clip_ids[idx]).zfill(3)
        f_from    = int(self.frame_from[idx])
        f_to      = int(self.frame_to[idx])
        label     = int(self.labels[idx])
        faces_dir = os.path.join(FACES_PATH, clip)

        frames = []
        for fn in range(f_from, f_to + 1):
            fpath = os.path.join(
                faces_dir, f"frame_det_00_{fn:06d}.jpg"
            )
            if not os.path.exists(fpath):
                continue
            try:
                img = Image.open(fpath).convert("RGB")
            except Exception:
                continue
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        if not frames:
            # Segmento senza frame → frame nera
            frames = [torch.zeros(3, 224, 224)]

        # Stack → (N_frames, 3, 224, 224)
        frames = torch.stack(frames)
        return frames, label
