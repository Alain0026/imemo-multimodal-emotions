"""
Estrae feature audio 128-dim per segmento usando HuBERT (BERT Audio Encoder).
Sostituisce Wav2Vec 2.0 con architettura BERT-style per l'audio.

- Segmento con audio  → HuBERT → mean pool → Linear 768→128-dim
- Segmento silenzioso → vettore zero (strategia del paper)

Output: data/processed/hubert_features/001/seg_000.npy ...
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import librosa
    from transformers import HubertModel, Wav2Vec2Processor
except ImportError:
    print("Installa: pip install transformers torchaudio librosa")
    exit(1)

DATASET_PATH     = "data/raw/IMEmo"
AUDIO_PATH       = "data/processed/audio"
HUBERT_FEAT_PATH = "data/processed/hubert_features"
FPS              = 25
SAMPLE_RATE      = 16000
HUBERT_DIM       = 768   # dimensione output HuBERT-base
FEAT_DIM         = 128   # dimensione target

SKIP_CLIPS = {
    "015","016","017","018","019","020","023","076"
}

os.makedirs(HUBERT_FEAT_PATH, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Carica HuBERT — usa lo stesso processor di Wav2Vec2 (compatibile)
print("Caricamento HuBERT (facebook/hubert-base-ls960)...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
hubert    = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
hubert.eval()

# Proiezione lineare 768 → 128 (stessa di Wav2Vec)
projector = nn.Linear(HUBERT_DIM, FEAT_DIM).to(device)
projector.eval()

# Salva/carica projector per riuso
proj_path = "models/hubert_projector.pth"
os.makedirs("models", exist_ok=True)
if not os.path.exists(proj_path):
    torch.save(projector.state_dict(), proj_path)
    print(f"HuBERT projector salvato: {proj_path}")
else:
    projector.load_state_dict(torch.load(proj_path, map_location=device))
    print(f"HuBERT projector caricato: {proj_path}")

clips = sorted([
    c for c in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, c))
    and c not in SKIP_CLIPS
])

for clip in tqdm(clips, desc="Clips"):
    clip_path  = os.path.join(DATASET_PATH, clip)
    json_files = [f for f in os.listdir(clip_path)
                  if "Frames_annotation" in f]
    if not json_files:
        continue

    with open(os.path.join(clip_path, json_files[0])) as f:
        shots = json.load(f)

    audio_file = os.path.join(AUDIO_PATH, f"{clip}.wav")
    out_dir    = os.path.join(HUBERT_FEAT_PATH, clip)
    os.makedirs(out_dir, exist_ok=True)

    # Carica audio
    if os.path.exists(audio_file):
        try:
            waveform, _ = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
            has_audio = True
        except Exception as e:
            print(f"[WARN] {clip}: {e}")
            has_audio = False
    else:
        has_audio = False

    for i, shot in enumerate(shots):
        out_file = os.path.join(out_dir, f"seg_{i:03d}.npy")
        if os.path.exists(out_file):
            continue

        if not has_audio:
            np.save(out_file, np.zeros(FEAT_DIM, dtype=np.float32))
            continue

        # Timestamp segmento
        frame_from = int(shot["Frame From"].split("_")[-1])
        frame_to   = int(shot["Frame to"].split("_")[-1])
        t_start    = frame_from / FPS
        t_end      = frame_to   / FPS
        s_start    = int(t_start * SAMPLE_RATE)
        s_end      = int(t_end   * SAMPLE_RATE)
        segment    = waveform[s_start:s_end]

        # Segmento troppo corto o silenzioso → zero vector
        if len(segment) < 400:
            np.save(out_file, np.zeros(FEAT_DIM, dtype=np.float32))
            continue

        rms = np.sqrt(np.mean(segment**2))
        if rms < 0.002:
            np.save(out_file, np.zeros(FEAT_DIM, dtype=np.float32))
            continue

        # HuBERT → (1, T, 768) → mean pool → (768,) → proj → (128,)
        try:
            inputs = processor(
                segment,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                outputs  = hubert(input_values)
                hidden   = outputs.last_hidden_state   # (1, T, 768)
                pooled   = hidden.squeeze(0).mean(dim=0)  # (768,)
                feat_128 = projector(pooled)               # (128,)

            np.save(out_file, feat_128.cpu().numpy().astype(np.float32))

        except Exception as e:
            print(f"[ERR] {clip} seg {i}: {e}")
            np.save(out_file, np.zeros(FEAT_DIM, dtype=np.float32))

print(f"\nFeature HuBERT estratte in: {HUBERT_FEAT_PATH}")
