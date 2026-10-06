"""
Estrae i frame dai video scaricati.
Output: data/processed/faces/001/frame_det_00_000001.jpg ...
"""
import os
import cv2
from tqdm import tqdm

VIDEOS_PATH = "data/raw/videos"
FRAMES_PATH = "data/processed/faces"
FPS_TARGET  = 25  # frame al secondo da estrarre

os.makedirs(FRAMES_PATH, exist_ok=True)

for video_file in sorted(os.listdir(VIDEOS_PATH)):
    if not video_file.endswith(".mp4"):
        continue

    clip_id    = video_file.replace(".mp4", "")
    video_path = os.path.join(VIDEOS_PATH, video_file)
    out_dir    = os.path.join(FRAMES_PATH, clip_id)
    os.makedirs(out_dir, exist_ok=True)

    cap         = cv2.VideoCapture(video_path)
    fps_orig    = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step        = max(1, int(fps_orig / FPS_TARGET))

    print(f"[{clip_id}] FPS: {fps_orig:.1f} | Frames totali: {frame_count}")

    idx      = 0
    saved    = 0
    with tqdm(total=frame_count, desc=clip_id) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                fname = f"frame_det_00_{saved+1:06d}.jpg"
                cv2.imwrite(os.path.join(out_dir, fname), frame)
                saved += 1
            idx += 1
            pbar.update(1)

    cap.release()
    print(f"  → {saved} frame salvati in {out_dir}")