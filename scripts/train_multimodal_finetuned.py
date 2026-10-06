"""
Training Multimodale con HuBERT (BERT Audio Encoder) + LSTM.
Architettura fedele a CVPRW2025 (Figure 2) ma con HuBERT invece di Wav2Vec 2.0.

Pipeline per ogni segmento i:
  V_i  = DAN features (512) → Linear → 128-dim  (da visual_features_finetuned/)
  S_i  = HuBERT features (128-dim)               (da hubert_features/)
  fused_i = V_i + S_i        (somma, 128-dim)
  emb_i   = f(emotion_{i-1}) (Embedding, 128-dim)
  lstm_in  = concat(fused, emb) → 256-dim → LSTM → Classification

Uso:
  python scripts/train_multimodal_hubert.py --mode sentiment
  python scripts/train_multimodal_hubert.py --mode basic
  python scripts/train_multimodal_hubert.py --mode fine
"""
import os, sys, argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, classification_report
import pandas as pd

VISUAL_FEAT_PATH = "data/processed/visual_features_finetuned"
HUBERT_FEAT_PATH = "data/processed/hubert_features"
VISUAL_DIM       = 512
AUDIO_DIM        = 128
PROJ_DIM         = 128


# ── Dataset ───────────────────────────────────────────────────────────────────
class MultimodalHubertDataset(Dataset):
    LABEL_MODES = {
        "sentiment": ("sentiment_idx", 3),
        "basic":     ("basic_idx",     7),
        "fine":      ("fine_idx",      17),
    }

    def __init__(self, csv_path, mode="sentiment", min_segments=2):
        df = pd.read_csv(csv_path)
        label_col, self.num_classes = self.LABEL_MODES[mode]
        df = df[df[label_col] >= 0].reset_index(drop=True)

        clips = []
        for clip_id, group in df.groupby("clip_id"):
            group = group.sort_values("segment_id")
            segs  = []
            for _, row in group.iterrows():
                clip   = str(row["clip_id"]).zfill(3)
                seg    = int(row["segment_id"])
                v_path = os.path.join(VISUAL_FEAT_PATH, clip, f"seg_{seg:03d}.npy")
                a_path = os.path.join(HUBERT_FEAT_PATH, clip, f"seg_{seg:03d}.npy")
                if os.path.exists(v_path):
                    segs.append({
                        "v":     v_path,
                        "a":     a_path,
                        "label": int(row[label_col]),
                    })
            if len(segs) >= min_segments:
                clips.append(segs)

        self.clips  = clips
        self.labels = np.array(
            [s["label"] for clip in clips for s in clip], dtype=np.int64)
        print(f"[MultimodalHubertDataset] mode={mode} | "
              f"classi={self.num_classes} | "
              f"clip={len(clips)} | segs={len(self.labels)}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        segs = self.clips[idx]
        n    = len(segs)
        V    = np.zeros((n, VISUAL_DIM), dtype=np.float32)
        A    = np.zeros((n, AUDIO_DIM),  dtype=np.float32)
        lbl  = np.zeros(n, dtype=np.int64)

        for i, s in enumerate(segs):
            V[i]   = np.load(s["v"])
            if os.path.exists(s["a"]):
                A[i] = np.load(s["a"])
            lbl[i] = s["label"]

        prev = np.concatenate([[self.num_classes], lbl[:-1]])
        return (
            torch.tensor(V,    dtype=torch.float32),
            torch.tensor(A,    dtype=torch.float32),
            torch.tensor(prev, dtype=torch.long),
            torch.tensor(lbl,  dtype=torch.long),
        )


def collate_clips(batch):
    V_list, A_list, prev_list, lbl_list = zip(*batch)
    max_len = max(v.shape[0] for v in V_list)
    B = len(batch)
    V_pad = torch.zeros(B, max_len, VISUAL_DIM)
    A_pad = torch.zeros(B, max_len, AUDIO_DIM)
    P_pad = torch.zeros(B, max_len, dtype=torch.long)
    L_pad = torch.full((B, max_len), -1, dtype=torch.long)
    for i, (v, a, p, l) in enumerate(zip(V_list, A_list, prev_list, lbl_list)):
        n = v.shape[0]
        V_pad[i,:n] = v
        A_pad[i,:n] = a
        P_pad[i,:n] = p
        L_pad[i,:n] = l
    return V_pad, A_pad, P_pad, L_pad


# ── Modello — identico a train_multimodal.py ─────────────────────────────────
class MultimodalLSTM(nn.Module):
    def __init__(self, num_classes, proj_dim=128,
                 hidden_dim=256, lstm_layers=1, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.visual_proj = nn.Sequential(
            nn.Linear(VISUAL_DIM, proj_dim), nn.ReLU())
        self.emotion_emb = nn.Embedding(num_classes + 1, proj_dim)
        self.lstm = nn.LSTM(
            input_size=proj_dim*2, hidden_size=hidden_dim,
            num_layers=lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, V, A, prev_emotions):
        V_proj  = self.visual_proj(V)              # (B,T,128)
        fused   = V_proj + A                        # (B,T,128) — somma come nel paper
        emb     = self.emotion_emb(prev_emotions)   # (B,T,128)
        out, _  = self.lstm(torch.cat([fused, emb], dim=-1))
        return self.classifier(out)                 # (B,T,C)


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",        default="sentiment",
                   choices=["sentiment","basic","fine"])
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--hidden",      type=int,   default=256)
    p.add_argument("--lstm_layers", type=int,   default=1)
    return p.parse_args()


# ── Training ──────────────────────────────────────────────────────────────────
def run():
    args      = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {args.mode} | "
          f"Multimodal HuBERT + LSTM | {timestamp}")

    train_ds = MultimodalHubertDataset("data/splits/train_segments.csv",
                                        mode=args.mode)
    test_ds  = MultimodalHubertDataset("data/splits/test_segments.csv",
                                        mode=args.mode)
    num_classes = train_ds.num_classes

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_clips,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_clips,
                              num_workers=0)

    model = MultimodalLSTM(num_classes=num_classes,
                           hidden_dim=args.hidden,
                           lstm_layers=args.lstm_layers).to(device)

    label_counts  = np.bincount(train_ds.labels, minlength=num_classes)
    label_counts  = np.where(label_counts == 0, 1, label_counts)
    class_weights = 1.0 / torch.tensor(label_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    print(f"Num classes: {num_classes} | Weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device),
                                     ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=1e-6)

    ckpt_dir  = os.path.join("checkpoints", "multimodal_hubert", args.mode)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, f"best_{args.mode}.pth")

    best_test_acc    = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for V, A, prev, labels in tqdm(train_loader,
                desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            V, A, prev, labels = (V.to(device), A.to(device),
                                   prev.to(device), labels.to(device))
            optimizer.zero_grad()
            logits = model(V, A, prev)
            B, T, C = logits.shape
            loss = criterion(logits.reshape(B*T, C), labels.reshape(B*T))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            run_loss += loss.item()
            mask      = labels.reshape(B*T) != -1
            correct  += (logits.reshape(B*T,C).argmax(1)[mask]
                         == labels.reshape(B*T)[mask]).sum().item()
            total    += mask.sum().item()

        train_acc = correct / max(total, 1)
        scheduler.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for V, A, prev, labels in test_loader:
                V, A, prev, labels = (V.to(device), A.to(device),
                                       prev.to(device), labels.to(device))
                logits = model(V, A, prev)
                B, T, C = logits.shape
                mask  = labels.reshape(B*T) != -1
                preds = logits.reshape(B*T,C).argmax(1)
                y_true.extend(labels.reshape(B*T)[mask].cpu().numpy())
                y_pred.extend(preds[mask].cpu().numpy())

        test_acc = np.mean(np.array(y_true) == np.array(y_pred))
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
        for V, A, prev, labels in test_loader:
            V, A, prev, labels = (V.to(device), A.to(device),
                                   prev.to(device), labels.to(device))
            logits = model(V, A, prev)
            B, T, C = logits.shape
            mask  = labels.reshape(B*T) != -1
            preds = logits.reshape(B*T,C).argmax(1)
            y_true.extend(labels.reshape(B*T)[mask].cpu().numpy())
            y_pred.extend(preds[mask].cpu().numpy())

    test_acc = np.mean(np.array(y_true) == np.array(y_pred))
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    print(f"\nTest Acc:     {test_acc:.4f}")
    print(f"Balanced Acc: {bal_acc:.4f}")
    print(f"Best Test:    {best_test_acc:.4f}")
    print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")

if __name__ == "__main__":
    run()
