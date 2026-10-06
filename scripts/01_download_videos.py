"""
Scarica e ritaglia i video di IMEmo da YouTube.
Output: data/raw/videos/001.mp4, 002.mp4, ...
"""
import os
import subprocess

DATASET_PATH = "data/raw/IMEmo"
OUTPUT_PATH  = r"data/raw/videos"
os.makedirs(OUTPUT_PATH, exist_ok=True)

def parse_txt(txt_path):
    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    url = lines[0]

    start, end = "0", "0"
    for line in lines[1:]:
        l = line.lower()
        if "start" in l:
            start = l.replace("start", "").replace(":", "").strip()
        elif "end" in l:
            end = l.replace("end", "").replace(":", "").strip()

    def to_hhmmss(t):
        t = t.strip()
        if not t or t == "0":
            return "00:00:00"
        parts = t.split(".")
        m = int(parts[0])
        s = int(parts[1]) if len(parts) > 1 else 0
        return f"00:{m:02d}:{s:02d}"

    return url, to_hhmmss(start), to_hhmmss(end)

SKIP_CLIPS = {"015","016","017","018","019","020","023","076"}
clips = sorted(os.listdir(DATASET_PATH))

for clip in clips:
    clip_path = os.path.join(DATASET_PATH, clip)
    if not os.path.isdir(clip_path):
        continue
    if clip in SKIP_CLIPS:
        print(f"[SKIP] {clip} — irrecuperabile")
        continue

    txt_files = [f for f in os.listdir(clip_path) if f.endswith(".txt")
                 and "Age" not in f]
    if not txt_files:
        continue

    txt_path = os.path.join(clip_path, txt_files[0])
    out_file  = os.path.join(OUTPUT_PATH, f"{clip}.mp4")

    if os.path.exists(out_file):
        print(f"[SKIP] {clip} già scaricato")
        continue

    try:
        url, start, end = parse_txt(txt_path)
        print(f"[{clip}] Scarico: {url}  ({start} → {end})")

        # Scarica il video completo in temp
        temp_file = os.path.join(OUTPUT_PATH, f"{clip}_full.mp4")
        subprocess.run([
            "yt-dlp", "-f", "bestvideo+bestaudio/best", "--merge-output-format", "mp4", "--no-playlist",
            "-o", temp_file, url
        ], check=True)

        # Ritaglia con ffmpeg
        subprocess.run([
            "ffmpeg", "-i", temp_file,
            "-ss", start, "-to", end,
            "-c", "copy", out_file, "-y"
        ], check=True)

        os.remove(temp_file)
        print(f"  → Salvato: {out_file}")

    except Exception as e:
        print(f"  [ERRORE] {clip}: {e}")