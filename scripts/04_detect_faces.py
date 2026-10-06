"""
Rileva e ritaglia le facce da ogni frame usando MTCNN.
Input:  data/processed/faces/{clip_id}/frame_det_00_XXXXXX.jpg
Output: data/processed/faces_cropped/{clip_id}/frame_det_00_XXXXXX.jpg
"""
import os
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

FRAMES_PATH  = "data/processed/faces"
CROPPED_PATH = "data/processed/faces_cropped"
FACE_SIZE    = 224  # dimensione attesa da DAN

os.makedirs(CROPPED_PATH, exist_ok=True)

# Inizializza MTCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn  = MTCNN(
    image_size=FACE_SIZE,
    margin=20,
    keep_all=False,      # tiene solo la faccia più grande
    post_process=False,  # restituisce pixel grezzi [0-255]
    device=device
)

print(f"Usando device: {device}")

total_saved   = 0
total_missing = 0

clip_dirs = sorted(os.listdir(FRAMES_PATH))

for clip_id in clip_dirs:
    clip_in  = os.path.join(FRAMES_PATH, clip_id)
    clip_out = os.path.join(CROPPED_PATH, clip_id)

    if not os.path.isdir(clip_in):
        continue

    os.makedirs(clip_out, exist_ok=True)

    frame_files = sorted([
        f for f in os.listdir(clip_in)
        if f.endswith(".jpg")
    ])

    saved   = 0
    missing = 0

    for fname in tqdm(frame_files, desc=f"[{clip_id}]", leave=False):
        in_path  = os.path.join(clip_in, fname)
        out_path = os.path.join(clip_out, fname)

        if os.path.exists(out_path):
            saved += 1
            continue

        try:
            img  = Image.open(in_path).convert("RGB")
            face = mtcnn(img)  # tensore [3, 224, 224] o None

            if face is not None:
                # Converte tensore → immagine PIL → salva
                face_np  = face.permute(1, 2, 0).numpy().astype(np.uint8)
                face_img = Image.fromarray(face_np)
                face_img.save(out_path)
                saved += 1
            else:
                missing += 1

        except Exception as e:
            missing += 1

    total_saved   += saved
    total_missing += missing
    tqdm.write(f"[{clip_id}] Salvati: {saved} | Nessuna faccia: {missing}")

print(f"\nTotale facce salvate:     {total_saved}")
print(f"Totale frame senza faccia: {total_missing}")
print(f"Completato in: {CROPPED_PATH}")