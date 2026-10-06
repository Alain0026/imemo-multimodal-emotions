"""
Fine-tuning DAN end-to-end — Unimodale Facial-only.
Usa gradient checkpointing + AMP per ridurre l'uso della VRAM.
"""
import os, sys, argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torchvision import transforms
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, classification_report
import pandas as pd

sys.path.insert(0, "DAN")
sys.path.insert(0, "src")
from networks.dan import DAN

FACES_PATH = "data/processed/faces_cropped"
VISUAL_DIM = 512
PROJ_DIM   = 128
SUB_BATCH  = 2   # molto piccolo per risparmiare VRAM


# ── Dataset ───────────────────────────────────────────────────────────────────
class FacialFinetuneDataset(Dataset):
    LABEL_MODES = {
        "sentiment": ("sentiment_idx", 3),
        "basic":     ("basic_idx",     7),
        "fine":      ("fine_idx",      17),
    }

    def __init__(self, csv_path, mode="sentiment", min_segments=2):
        df = pd.read_csv(csv_path)
        label_col, self.num_classes = self.LABEL_MODES[mode]
        df = df[df[label_col] >= 0].reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

        clips = []
        for clip_id, group in df.groupby("clip_id"):
            group = group.sort_values("segment_id")
            segs  = []
            for _, row in group.iterrows():
                clip   = str(row["clip_id"]).zfill(3)
                f_from = int(row["frame_from"])
                f_to   = int(row["frame_to"])
                sample = os.path.join(FACES_PATH, clip,
                                       f"frame_det_00_{f_from:06d}.jpg")
                if not os.path.exists(sample):
                    continue
                segs.append({
                    "clip":   clip,
                    "f_from": f_from,
                    "f_to":   f_to,
                    "label":  int(row[label_col]),
                })
            if len(segs) >= min_segments:
                clips.append(segs)

        self.clips  = clips
        self.labels = np.array(
            [s["label"] for c in clips for s in c], dtype=np.int64)
        print(f"[FacialFinetuneDataset] mode={mode} | "
              f"classi={self.num_classes} | "
              f"clip={len(clips)} | segs={len(self.labels)}")

    def __len__(self):
        return len(self.clips)

    def load_frames(self, clip, f_from, f_to):
        frames = []
        for fn in range(f_from, f_to + 1):
            path = os.path.join(FACES_PATH, clip,
                                 f"frame_det_00_{fn:06d}.jpg")
            if not os.path.exists(path):
                continue
            try:
                img = Image.open(path).convert("RGB")
                frames.append(self.transform(img))
            except Exception:
                continue
        if not frames:
            frames = [torch.zeros(3, 224, 224)]
        return torch.stack(frames)

    def __getitem__(self, idx):
        segs   = self.clips[idx]
        items  = [(self.load_frames(s["clip"], s["f_from"], s["f_to"]),
                   s["label"]) for s in segs]
        labels = [it[1] for it in items]
        prev   = [self.num_classes] + labels[:-1]
        return [(it[0],) for it in items], prev, labels


def collate_fn(batch):
    return batch


# ── Modello ───────────────────────────────────────────────────────────────────
class DanLSTM(nn.Module):
    def __init__(self, num_classes, proj_dim=128,
                 hidden_dim=256, lstm_layers=1, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.dan = DAN(num_class=num_classes, num_head=4)

        self.visual_proj = nn.Sequential(
            nn.Linear(VISUAL_DIM, proj_dim), nn.ReLU())
        self.emotion_emb = nn.Embedding(num_classes + 1, proj_dim)
        self.lstm = nn.LSTM(
            input_size=proj_dim*2, hidden_size=hidden_dim,
            num_layers=lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def freeze_early_layers(self):
        for p in self.dan.parameters():
            p.requires_grad = False
        # Sblocca layer3, layer4, attention, bn, fc
        for p in self.dan.features[6:].parameters():
            p.requires_grad = True
        for i in range(self.dan.num_head):
            for p in getattr(self.dan, f"cat_head{i}").parameters():
                p.requires_grad = True
        for p in self.dan.bn.parameters():
            p.requires_grad = True
        for p in self.dan.fc.parameters():
            p.requires_grad = True

    def dan_forward(self, sub):
        """Wrapper per gradient checkpointing."""
        _, feat, _ = self.dan(sub)
        return feat

    def extract_feat(self, frames, device, training):
        """Estrae feature 512-dim con sub-batch + gradient checkpointing."""
        all_gaps = []
        n = len(frames)
        if n == 1:
            frames = frames.repeat(2, 1, 1, 1)

        for i in range(0, len(frames), SUB_BATCH):
            sub = frames[i:i+SUB_BATCH].to(device)
            if len(sub) == 1:
                sub = sub.repeat(2, 1, 1, 1)
                if training:
                    feat = grad_checkpoint(self.dan_forward, sub,
                                           use_reentrant=False)
                else:
                    feat = self.dan_forward(sub)
                gap = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                all_gaps.append(gap[:1])
            else:
                if training:
                    feat = grad_checkpoint(self.dan_forward, sub,
                                           use_reentrant=False)
                else:
                    feat = self.dan_forward(sub)
                gap = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                all_gaps.append(gap)

            torch.cuda.empty_cache()

        gaps = torch.cat(all_gaps, dim=0)          # (N, 512)
        return gaps.mean(dim=0, keepdim=True)       # (1, 512)

    def forward_sequence(self, items, prevs, device, training):
        v_list, emb_list = [], []
        for (frames,), prev_em in zip(items, prevs):
            v     = self.extract_feat(frames, device, training)
            v_list.append(self.visual_proj(v))
            em    = torch.tensor([prev_em], dtype=torch.long, device=device)
            emb_list.append(self.emotion_emb(em))

        v_seq   = torch.stack(v_list,   dim=1)
        emb_seq = torch.stack(emb_list, dim=1)
        out, _  = self.lstm(torch.cat([v_seq, emb_seq], dim=-1))
        return self.classifier(out).squeeze(0)      # (T, C)


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",     default="sentiment",
                   choices=["sentiment","basic","fine"])
    p.add_argument("--epochs",   type=int,   default=50)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--lr_dan",   type=float, default=1e-4)
    p.add_argument("--patience", type=int,   default=7)
    p.add_argument("--hidden",   type=int,   default=256)
    return p.parse_args()


# ── Training ──────────────────────────────────────────────────────────────────
def run():
    args      = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {args.mode} | "
          f"DAN Fine-tuning Facial-only | {timestamp}")

    train_ds = FacialFinetuneDataset("data/splits/train_segments.csv",
                                      mode=args.mode)
    test_ds  = FacialFinetuneDataset("data/splits/test_segments.csv",
                                      mode=args.mode)
    num_classes = train_ds.num_classes

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)

    model = DanLSTM(num_classes=num_classes,
                    hidden_dim=args.hidden).to(device)

    # Carica backbone DAN (escludi fc e bn che dipendono da num_classes)
    ckpt_path = "checkpoints/dan/sentiment/best_sentiment.pth"
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        keys_to_remove = [k for k in state
                          if k.startswith("fc.") or k.startswith("bn.")]
        for k in keys_to_remove:
            del state[k]
        model.dan.load_state_dict(state, strict=False)
        print(f"DAN backbone caricato ({len(keys_to_remove)} layer esclusi)")

    model.freeze_early_layers()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parametri trainabili: {n_train:,}/{n_total:,} "
          f"({100*n_train/n_total:.1f}%)")

    label_counts  = np.bincount(train_ds.labels, minlength=num_classes)
    label_counts  = np.where(label_counts == 0, 1, label_counts)
    class_weights = 1.0 / torch.tensor(label_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    dan_params  = [p for p in model.dan.parameters() if p.requires_grad]
    lstm_params = (list(model.visual_proj.parameters()) +
                   list(model.emotion_emb.parameters()) +
                   list(model.lstm.parameters()) +
                   list(model.classifier.parameters()))

    optimizer = torch.optim.Adam([
        {'params': dan_params,  'lr': args.lr_dan},
        {'params': lstm_params, 'lr': args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    ckpt_dir  = os.path.join("checkpoints", "finetune_facial", args.mode)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, f"best_{args.mode}.pth")

    best_test_acc    = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(train_loader,
                          desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            items, prevs, labels = batch[0]
            optimizer.zero_grad()

            with autocast():
                logits = model.forward_sequence(items, prevs, device, True)
                lbl    = torch.tensor(labels, dtype=torch.long, device=device)
                loss   = criterion(logits, lbl)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.empty_cache()

            run_loss += loss.item()
            correct  += (logits.argmax(1) == lbl).sum().item()
            total    += len(labels)

        train_acc = correct / max(total, 1)
        scheduler.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                items, prevs, labels = batch[0]
                with autocast():
                    logits = model.forward_sequence(items, prevs, device, False)
                lbl = torch.tensor(labels, dtype=torch.long, device=device)
                y_true.extend(lbl.cpu().numpy())
                y_pred.extend(logits.argmax(1).cpu().numpy())

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
        for batch in test_loader:
            items, prevs, labels = batch[0]
            with autocast():
                logits = model.forward_sequence(items, prevs, device, False)
            lbl = torch.tensor(labels, dtype=torch.long, device=device)
            y_true.extend(lbl.cpu().numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())

    test_acc = np.mean(np.array(y_true) == np.array(y_pred))
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    print(f"\nTest Acc:     {test_acc:.4f}")
    print(f"Balanced Acc: {bal_acc:.4f}")
    print(f"Best Test:    {best_test_acc:.4f}")
    print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")


if __name__ == "__main__":
    run()
