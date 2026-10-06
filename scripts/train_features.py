"""
Training su feature pre-estratte DAN (512-dim per segmento).
Nessun forward DAN → velocissimo e zero OOM.
Uso:
  python scripts/train_features.py --mode sentiment
"""
import os, sys, argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
import sys
sys.path.insert(0, "scripts")

sys.path.insert(0, "src")
from data.feature_dataset import FeatureDataset


class SegmentClassifier(nn.Module):
    """Piccolo MLP su feature 512-dim → num_classes"""
    def __init__(self, feat_dim=512, num_classes=3, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       default="sentiment",
                        choices=["sentiment","basic","fine"])
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=10)
    return parser.parse_args()


def run():
    args      = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {args.mode} | Started: {timestamp}")

    train_ds = FeatureDataset("data/splits/train_segments.csv", mode=args.mode)
    test_ds  = FeatureDataset("data/splits/test_segments.csv",  mode=args.mode)
    num_classes = train_ds.num_classes
    if len(train_ds) == 0 or len(test_ds) == 0:
        print(f"[ERRORE] Dataset vuoto per mode={args.mode}. "
              f"Train={len(train_ds)}, Test={len(test_ds)}")
        print("Verifica che basic_idx sia presente nel CSV dei segmenti.")
        return

    # WeightedRandomSampler
    label_counts  = np.bincount(train_ds.labels, minlength=num_classes)
    label_counts  = np.where(label_counts == 0, 1, label_counts)
    class_weights = 1.0 / torch.tensor(label_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    sample_w = torch.tensor(
        np.array(class_weights.numpy(), copy=True)[train_ds.labels],
        dtype=torch.float)
    sampler = WeightedRandomSampler(sample_w, len(train_ds), replacement=True)
    print(f"Num classes: {num_classes} | Weights: {class_weights.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    model     = SegmentClassifier(feat_dim=512, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=1e-6)

    ckpt_dir  = os.path.join("checkpoints", "features", args.mode)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, f"best_{args.mode}.pth")

    best_test_acc    = 0.0
    patience_counter = 0
    history = {"epoch":[], "loss":[], "acc":[], "test_acc":[]}

    for epoch in range(1, args.epochs + 1):

        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for feats, labels in tqdm(train_loader,
                desc=f"Epoch {epoch}/{args.epochs} [train]",
                leave=False):
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
        history["epoch"].append(epoch)
        history["loss"].append(run_loss / len(train_loader))
        history["acc"].append(train_acc)
        scheduler.step()

        model.eval()
        correct_t, total_t = 0, 0
        with torch.no_grad():
            for feats, labels in test_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
                correct_t += (logits.argmax(1) == labels).sum().item()
                total_t   += labels.size(0)

        test_acc_epoch = correct_t / total_t
        history["test_acc"].append(test_acc_epoch)

        print(f"[Epoch {epoch:3d}] Loss: {run_loss/len(train_loader):.3f} | "
              f"Train: {train_acc:.4f} | Test: {test_acc_epoch:.4f} | "
              f"Patience: {patience_counter}/{args.patience}")

        if test_acc_epoch > best_test_acc:
            best_test_acc    = test_acc_epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"  [BEST] salvato (Test={best_test_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping] Stop a epoca {epoch}.")
                break

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    print("\nValutazione finale...")
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
    print(f"Test Acc: {test_acc:.4f} | Balanced Acc: {bal_acc:.4f}")
    print(f"Best Test Acc: {best_test_acc:.4f}")
    print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")

    # ── Generazione PDF ──────────────────────────────────────────────
    from scripts.train_dan import generate_pdf_report
    import argparse as _ap
    _fake = _ap.Namespace(
        mode       = args.mode,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        num_head   = 0,
        patience   = args.patience,
    )
    report_dir = os.path.join("experiments", "reports")
    generate_pdf_report(
        args        = _fake,
        history     = history,
        test_acc    = test_acc,
        bal_acc     = bal_acc,
        y_true      = y_true,
        y_pred      = y_pred,
        class_names = [str(i) for i in range(num_classes)],
        report_dir  = report_dir,
        timestamp   = timestamp,
    )

if __name__ == "__main__":
    run()
