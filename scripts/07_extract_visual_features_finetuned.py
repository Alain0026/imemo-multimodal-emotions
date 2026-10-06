"""
Estrae feature visive 512-dim usando il DAN fine-tuned.
Checkpoint: checkpoints/finetune_facial/sentiment/best_sentiment.pth
Output: data/processed/visual_features_finetuned/001/seg_000.npy ...
"""
import os, json, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, "DAN")
from networks.dan import DAN

DATASET_PATH     = "data/raw/IMEmo"
FACES_PATH       = "data/processed/faces_cropped"
VISUAL_FEAT_PATH = "data/processed/visual_features_finetuned"
BATCH_SIZE       = 16

SKIP_CLIPS = {
    "015","016","017","018","019","020","023",
    "025","076","079","080","081","084"
}

os.makedirs(VISUAL_FEAT_PATH, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Carica DAN fine-tuned
model = DAN(num_class=3, num_head=4).to(device)
ckpt  = "checkpoints/finetune_facial/sentiment/best_sentiment.pth"
if os.path.exists(ckpt):
    state = torch.load(ckpt, map_location=device)
    # Estrai solo i pesi DAN (prefix "dan.")
    dan_state = {k.replace("dan.", ""): v
                 for k, v in state.items() if k.startswith("dan.")}
    model.load_state_dict(dan_state, strict=False)
    print(f"DAN fine-tuned caricato: {ckpt}")
else:
    print(f"[WARN] Checkpoint non trovato: {ckpt} — uso pesi random")

model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

clips = sorted([c for c in os.listdir(DATASET_PATH)
                if os.path.isdir(os.path.join(DATASET_PATH, c))
                and c not in SKIP_CLIPS])

for clip in tqdm(clips, desc="Clips"):
    clip_path  = os.path.join(DATASET_PATH, clip)
    json_files = [f for f in os.listdir(clip_path) if "Frames_annotation" in f]
    if not json_files:
        continue

    with open(os.path.join(clip_path, json_files[0])) as f:
        shots = json.load(f)

    faces_dir = os.path.join(FACES_PATH, clip)
    out_dir   = os.path.join(VISUAL_FEAT_PATH, clip)
    os.makedirs(out_dir, exist_ok=True)

    for i, shot in enumerate(shots):
        out_file = os.path.join(out_dir, f"seg_{i:03d}.npy")
        if os.path.exists(out_file):
            continue

        frame_from = int(shot["Frame From"].split("_")[-1])
        frame_to   = int(shot["Frame to"].split("_")[-1])

        all_gaps = []
        batch    = []

        def flush_batch(batch):
            if not batch:
                return
            t = torch.stack(batch).to(device)
            with torch.no_grad():
                _, feats, _ = model(t)
            gap = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
            all_gaps.append(gap.cpu())

        for fn in range(frame_from, frame_to + 1):
            fpath = os.path.join(faces_dir, f"frame_det_00_{fn:06d}.jpg")
            if not os.path.exists(fpath):
                continue
            try:
                img = Image.open(fpath).convert("RGB")
                batch.append(transform(img))
            except Exception:
                continue
            if len(batch) == BATCH_SIZE:
                flush_batch(batch)
                batch = []
        if batch:
            flush_batch(batch)

        if all_gaps:
            feat = torch.cat(all_gaps, dim=0).mean(dim=0).numpy()
        else:
            feat = np.zeros(512, dtype=np.float32)

        np.save(out_file, feat.astype(np.float32))

print(f"\nFeature DAN fine-tuned salvate in: {VISUAL_FEAT_PATH}")
