"""
Estrae feature audio 128-dim per segmento usando Wav2Vec 2.0.
- Se il segmento ha audio → Wav2Vec 2.0 → mean pool → proiezione 128-dim
- Se il segmento e' silenzioso o manca → vettore zero (come nel paper)
Output: data/processed/audio_features/001/seg_000.npy ...
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import librosa
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
except ImportError:
    print("Installa: pip install transformers torchaudio librosa")
    exit(1)

DATASET_PATH     = "data/raw/IMEmo"
AUDIO_PATH       = "data/processed/audio"
AUDIO_FEAT_PATH  = "data/processed/audio_features"
FPS              = 25
SAMPLE_RATE      = 16000
WAV2VEC_DIM      = 768   # dimensione output Wav2Vec2-base
FEAT_DIM         = 128   # dimensione target (come nel paper)

SKIP_CLIPS = {
    "015","016","017","018","019","020","023","076"
}

os.makedirs(AUDIO_FEAT_PATH, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Carica Wav2Vec 2.0
print("Caricamento Wav2Vec 2.0 (facebook/wav2vec2-base)...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec   = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
wav2vec.eval()

# Proiezione lineare 768 → 128
projector = nn.Linear(WAV2VEC_DIM, FEAT_DIM).to(device)
projector.eval()  # pesi casuali ma fissi — si puo' addestrare insieme al classifier

# Salva il projector per riuso
proj_path = "models/audio_projector.pth"
os.makedirs("models", exist_ok=True)
if not os.path.exists(proj_path):
    torch.save(projector.state_dict(), proj_path)
    print(f"Projector salvato: {proj_path}")
else:
    projector.load_state_dict(torch.load(proj_path, map_location=device))
    print(f"Projector caricato: {proj_path}")

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

    audio_path = os.path.join(AUDIO_PATH, f"{clip}.wav")
    out_dir    = os.path.join(AUDIO_FEAT_PATH, clip)
    os.makedirs(out_dir, exist_ok=True)

    # Carica audio del clip
    if os.path.exists(audio_path):
        try:
            waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            has_audio = True
        except Exception as e:
            print(f"[WARN] {clip}: errore audio: {e}")
            has_audio = False
    else:
        has_audio = False

    for i, shot in enumerate(shots):
        out_file = os.path.join(out_dir, f"seg_{i:03d}.npy")
        if os.path.exists(out_file):
            continue

        # Zero vector se non c'e' audio
        if not has_audio:
            np.save(out_file, np.zeros(FEAT_DIM, dtype=np.float32))
            continue

        # Calcola timestamp del segmento
        frame_from = int(shot["Frame From"].split("_")[-1])
        frame_to   = int(shot["Frame to"].split("_")[-1])
        t_start    = frame_from / FPS
        t_end      = frame_to   / FPS

        s_start = int(t_start * SAMPLE_RATE)
        s_end   = int(t_end   * SAMPLE_RATE)
        segment = waveform[s_start:s_end]

        # Segmento troppo corto (< 0.025 sec) → zero vector
        if len(segment) < 400:
            np.save(out_file, np.zeros(FEAT_DIM, dtype=np.float32))
            continue

        # Verifica se il segmento e' silenzioso (RMS < soglia)
        rms = np.sqrt(np.mean(segment**2))
        if rms < 0.002:
            np.save(out_file, np.zeros(FEAT_DIM, dtype=np.float32))
            continue

        # Wav2Vec 2.0 → (T, 768) → mean pool → (768,) → proiezione → (128,)
        try:
            inputs = processor(
                segment,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                outputs  = wav2vec(input_values)
                hidden   = outputs.last_hidden_state  # (1, T, 768)
                pooled   = hidden.squeeze(0).mean(dim=0)  # (768,)
                feat_128 = projector(pooled)               # (128,)

            np.save(out_file, feat_128.cpu().numpy().astype(np.float32))

        except Exception as e:
            print(f"[ERR] {clip} seg {i}: {e}")
            np.save(out_file, np.zeros(FEAT_DIM, dtype=np.float32))

print(f"\nFeature audio estratte in: {AUDIO_FEAT_PATH}")
