"""
Estrae feature visive DAN 512-dim per segmento e le salva su disco.
Output: data/processed/visual_features/001/seg_000.npy ...
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
VISUAL_FEAT_PATH = "data/processed/visual_features"
SKIP_CLIPS = {
    "015","016","017","018","019","020","023","076"
}

os.makedirs(VISUAL_FEAT_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = DAN(num_class=3, num_head=4).to(device)

best_ckpt = "checkpoints/dan/sentiment/best_sentiment.pth"
if os.path.exists(best_ckpt):
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"DAN caricato da: {best_ckpt}")
else:
    print("Usando pesi MS-Celeb predefiniti")

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

        # Carica frames del segmento in mini-batch di 16
        all_gaps = []
        batch = []

        def flush(batch):
            if not batch:
                return
            t = torch.stack(batch).to(device)  # (N, 3, 224, 224)
            with torch.no_grad():
                _, feats, _ = model(t)          # (N, 512, 7, 7)
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
            if len(batch) == 16:
                flush(batch); batch = []

        if batch:
            flush(batch)

        if all_gaps:
            all_gaps_t = torch.cat(all_gaps, dim=0)    # (N, 512)
            seg_feat   = all_gaps_t.mean(dim=0).numpy() # (512,)
        else:
            seg_feat = np.zeros(512, dtype=np.float32)

        np.save(out_file, seg_feat.astype(np.float32))

print("Feature visive estratte e salvate.")
