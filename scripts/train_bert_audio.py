"""
Training Speech-only con HuBERT (BERT Audio Encoder).
Confronto diretto con train_speech.py (Wav2Vec 2.0):
  Wav2Vec 2.0 → 42.40% sentiment
  HuBERT      → ?

Uso:
  python scripts/train_bert_audio.py --mode sentiment
  python scripts/train_bert_audio.py --mode basic
  python scripts/train_bert_audio.py --mode fine
"""
import os, sys, argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score, classification_report
import pandas as pd

HUBERT_FEAT_PATH = "data/processed/hubert_features"
AUDIO_DIM        = 128


class HubertDataset(Dataset):
    LABEL_MODES = {
        "sentiment": ("sentiment_idx", 3),
        "basic":     ("basic_idx",     7),
        "fine":      ("fine_idx",      17),
    }

    def __init__(self, csv_path, mode="sentiment"):
        df = pd.read_csv(csv_path)
        label_col, self.num_classes = self.LABEL_MODES[mode]
        df = df[df[label_col] >= 0].reset_index(drop=True)

        valid = []
        for _, row in df.iterrows():
            clip   = str(row["clip_id"]).zfill(3)
            seg    = int(row["segment_id"])
            a_path = os.path.join(HUBERT_FEAT_PATH, clip, f"seg_{seg:03d}.npy")
            if os.path.exists(a_path):
                valid.append((a_path, int(row[label_col])))

        self.samples = valid
        self.labels  = np.array([s[1] for s in valid], dtype=np.int64)
        print(f"[HubertDataset] mode={mode} | "
              f"classi={self.num_classes} | segmenti={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        a_path, label = self.samples[idx]
        feat = np.load(a_path)   # (128,)
        return torch.tensor(feat, dtype=torch.float32), label


class BertAudioClassifier(nn.Module):
    """
    Classificatore MLP su feature HuBERT 128-dim.
    Architettura identica a SpeechClassifier per confronto diretto.
    """
    def __init__(self, num_classes=3, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(AUDIO_DIM, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       default="sentiment",
                   choices=["sentiment","basic","fine"])
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int,   default=10)
    return p.parse_args()


def run():
    args      = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {args.mode} | "
          f"BERT Audio (HuBERT) Speech-only | {timestamp}")

    train_ds = HubertDataset("data/splits/train_segments.csv", mode=args.mode)
    test_ds  = HubertDataset("data/splits/test_segments.csv",  mode=args.mode)
    num_classes = train_ds.num_classes

    # WeightedRandomSampler
    label_counts  = np.bincount(train_ds.labels, minlength=num_classes)
    label_counts  = np.where(label_counts == 0, 1, label_counts)
    class_weights = 1.0 / torch.tensor(label_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    sample_w = torch.tensor(
        np.array(class_weights.numpy(), copy=True)[train_ds.labels],
        dtype=torch.float)
    sampler = WeightedRandomSampler(sample_w, len(train_ds), replacement=True)
    print(f"Weights: {class_weights.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    model     = BertAudioClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=1e-6)

    ckpt_dir  = os.path.join("checkpoints", "bert_audio", args.mode)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, f"best_{args.mode}.pth")

    best_test_acc    = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for feats, labels in tqdm(train_loader,
                desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += labels.size(0)

        train_acc = correct / total
        scheduler.step()

        model.eval()
        correct_t, total_t = 0, 0
        with torch.no_grad():
            for feats, labels in test_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
                correct_t += (logits.argmax(1) == labels).sum().item()
                total_t   += labels.size(0)

        test_acc = correct_t / total_t
        print(f"[Epoch {epoch:3d}] Loss: {run_loss/len(train_loader):.3f} | "
              f"Train: {train_acc:.4f} | Test: {test_acc:.4f} | "
              f"Patience: {patience_counter}/{args.patience}")

        if test_acc > best_test_acc:
            best_test_acc    = test_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"  [BEST] (Test={best_test_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping] Stop a epoca {epoch}")
                break

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for feats, labels in test_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())

    test_acc = np.mean(np.array(y_true) == np.array(y_pred))
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    print(f"\nTest Acc:     {test_acc:.4f}")
    print(f"Balanced Acc: {bal_acc:.4f}")
    print(f"Best Test:    {best_test_acc:.4f}")
    print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")

if __name__ == "__main__":
    run()
