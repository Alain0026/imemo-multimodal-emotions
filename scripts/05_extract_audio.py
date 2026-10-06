"""
Estrae audio WAV 16kHz mono da ogni clip video.
Output: data/processed/audio/001.wav, 002.wav, ...
"""
import os, subprocess

VIDEOS_PATH = "data/raw/videos"
AUDIO_PATH  = "data/processed/audio"
os.makedirs(AUDIO_PATH, exist_ok=True)

SKIP_CLIPS = {
    "015","016","017","018","019","020","023","076"
}

clips = sorted([f for f in os.listdir(VIDEOS_PATH) if f.endswith(".mp4")])
for clip_file in clips:
    clip_id = clip_file.replace(".mp4","")
    if clip_id in SKIP_CLIPS:
        continue
    out_file = os.path.join(AUDIO_PATH, f"{clip_id}.wav")
    if os.path.exists(out_file):
        print(f"[SKIP] {clip_id}")
        continue
    print(f"[{clip_id}] Estraggo audio...")
    subprocess.run([
        "ffmpeg", "-i", os.path.join(VIDEOS_PATH, clip_file),
        "-ac", "1", "-ar", "16000", "-q:a", "0",
        out_file, "-y", "-loglevel", "error"
    ], check=True)
    print(f"  → {out_file}")

print(f"Audio estratti: {len(os.listdir(AUDIO_PATH))}")
