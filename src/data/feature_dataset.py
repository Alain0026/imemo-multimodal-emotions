"""
Dataset che carica feature pre-estratte da disco.
Ogni segmento = un vettore 512-dim già calcolato.
Nessun forward DAN durante il training → zero OOM.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

VISUAL_FEAT_PATH = "data/processed/visual_features"

class FeatureDataset(Dataset):
    LABEL_MODES = {
        "sentiment": ("sentiment_idx", 3),
        "basic":     ("basic_idx",     7),  #<- 6-> 7
        "fine":      ("fine_idx",      17), #<- 16 -> 17
    }

    def __init__(self, csv_path, mode="sentiment"):
        df = pd.read_csv(csv_path)
        label_col, self.num_classes = self.LABEL_MODES[mode]

        df = df[df[label_col] >= 0].reset_index(drop=True)

        # Filtra segmenti che hanno la feature su disco
        valid = []
        for _, row in df.iterrows():
            clip = str(row["clip_id"]).zfill(3)
            seg  = int(row["segment_id"])
            feat_path = os.path.join(VISUAL_FEAT_PATH, clip, f"seg_{seg:03d}.npy")
            if os.path.exists(feat_path):
                valid.append((feat_path, int(row[label_col])))

        self.samples = valid
        self.labels  = np.array([s[1] for s in valid])

        print(f"[FeatureDataset] mode={mode} | "
              f"classi={self.num_classes} | "
              f"segmenti={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, label = self.samples[idx]
        feat = np.load(feat_path)               # (512,)
        return torch.tensor(feat, dtype=torch.float32), label
