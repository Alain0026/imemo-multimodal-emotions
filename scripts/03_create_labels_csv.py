"""
Legge i Frames_annotation.json e crea train.csv e test.csv.
Ogni riga: clip_id, frame_path, emotion, basic_emotion, sentiment
"""
import os
import json
import random
import pandas as pd

DATASET_PATH = "data/raw/IMEmo"
FRAMES_PATH  = "data/processed/faces_cropped"
SPLITS_PATH  = "data/splits"
os.makedirs(SPLITS_PATH, exist_ok=True)

SENTIMENT_MAP = {"Positive": 0, "Negative": 1, "Neutral": 2}

BASIC_MAP = {
    "Anger":    0, "Angry":     0,  # alias
    "Disgust":  1,
    "Fear":     2,
    "Happy":    3, "Happiness": 3,  # alias
    "Sadness":  4, "Sad":       4,  # alias
    "Surprise": 5,
    "Neutral":  6,  # ← AGGIUNTO come classe 6
}

FINE_MAP = {
    # 0 — Sadness
    "Sadness":       0, "Hurt":         0,
    "Disappointment":0, "Regret":       0, "Pain": 0,

    # 1 — Shame
    "Shame":         1, "Embarrassment":1, "Shyness": 1,

    # 2 — Fear
    "Fear":          2,

    # 3 — Distress
    "Distress":      3, "Frustration":  3,
    "Confusion":     3, "Concern":      3,

    # 4 — Anger
    "Anger":         4, "Defensiveness":4,

    # 5 — Contempt
    "Contempt":      5,

    # 6 — Rage
    "Rage":          6,

    # 7 — Disgust
    "Disgust":       7,

    # 8 — Wonder
    "Wonder":        8, "Curiosity":    8,
    "Thoughtfulness":8, "Reflection":   8,

    # 9 — Desire
    "Desire":        9,

    # 10 — Surprise
    "Surprise":     10,

    # 11 — Anxiety
    "Anxiety":      11,

    # 12 — Love
    "Love":         12, "Affection":   12,
    "Warmth":       12, "Empathy":     12,
    "Compassion":   12, "Reassurance": 12,
    "Encouragement":12,

    # 13 — Happy
    "Happy":        13, "Happiness":   13,
    "Contentment":  13, "Satisfaction":13,
    "Relief":       13, "Confidence":  13,

    # 14 — Joy
    "Joy":          14, "Amusement":   14,

    # 15 — Interest
    "Interest":     15, "Attention":   15,
    "Acceptance":   15,
    # 16 — Neutral (riaggiunto per istruzione del prof)
    "Neutral":      16, "Explanation": 16,
    "Seriousness":  16, "Patience":    16,
    "Politeness":   16, "Understanding":16,
}
TYPO_FIX = {"Sadnesss": "Sadness"}

def clean_emotion(e):
    return TYPO_FIX.get(e, e)

# Clips i quali vidéo non sono disponibili (15 manquanti su 100)
SKIP_CLIPS = {
    "015","016","017","018","019","020","023","076"
}

# Colletta toute le emozioni fine-grained
all_emotions = set()
for clip in sorted(os.listdir(DATASET_PATH)):
    clip_path = os.path.join(DATASET_PATH, clip)
    if not os.path.isdir(clip_path):
        continue
    if clip in SKIP_CLIPS:          # ← SKIP applicato anche qui
        continue
    json_files = [f for f in os.listdir(clip_path)
                  if "Frames_annotation" in f]
    if not json_files:
        continue
    with open(os.path.join(clip_path, json_files[0])) as f:
        shots = json.load(f)
    for shot in shots:
        all_emotions.add(clean_emotion(shot["Emotion"]))

EMOTION_MAP = {e: i for i, e in enumerate(sorted(all_emotions))}
print(f"Emozioni trovate ({len(EMOTION_MAP)}): {EMOTION_MAP}")

records = []
clips   = sorted(os.listdir(DATASET_PATH))

for clip in clips:
    clip_path = os.path.join(DATASET_PATH, clip)
    if not os.path.isdir(clip_path):
        continue

    # ← SKIP correttamente applicato adesso
    if clip in SKIP_CLIPS:
        print(f"[SKIP] {clip} — video non disponibili")
        continue

    json_files = [f for f in os.listdir(clip_path)
                  if "Frames_annotation" in f]
    if not json_files:
        continue

    with open(os.path.join(clip_path, json_files[0])) as f:
        shots = json.load(f)

    for shot in shots:
        frame_from = int(shot["Frame From"].split("_")[-1])
        frame_to   = int(shot["Frame to"].split("_")[-1])
        emotion    = clean_emotion(shot["Emotion"])
        basic      = clean_emotion(shot.get("Basic_Emotion", ""))
        sentiment  = shot.get("Sentiment", "")

        for fn in range(frame_from, frame_to + 1):
            fname      = f"frame_det_00_{fn:06d}.jpg"
            frame_path = os.path.join(FRAMES_PATH, clip, fname)
            if not os.path.exists(frame_path):
                continue
            records.append({
                "clip_id":       clip,
                "frame_path":    frame_path,
                "emotion":       emotion,
                "emotion_idx":   EMOTION_MAP.get(emotion, -1),
                "basic_emotion": basic,
                "basic_idx":     BASIC_MAP.get(basic, -1),
                "fine_idx":      FINE_MAP.get(emotion, -1),
                "sentiment":     sentiment,
                "sentiment_idx": SENTIMENT_MAP.get(sentiment, -1),
            })

df = pd.DataFrame(records)

# ── Split 80/20 basé sur les clip_ids ──────────────────────────────
random.seed(42)
clip_ids = df["clip_id"].unique().tolist()
random.shuffle(clip_ids)
n_train   = int(len(clip_ids) * 0.80)
train_ids = set(clip_ids[:n_train])
test_ids  = set(clip_ids[n_train:])

# ── CSV sentiment (tous les clips, sentinel_idx != -1) ─────────────
df_sentiment = df[df["sentiment_idx"] != -1]
df_sentiment[df_sentiment["clip_id"].isin(train_ids)].to_csv(
    os.path.join(SPLITS_PATH, "train.csv"), index=False)
df_sentiment[df_sentiment["clip_id"].isin(test_ids)].to_csv(
    os.path.join(SPLITS_PATH, "test.csv"), index=False)
print(f"Sentiment — Train: {len(df_sentiment[df_sentiment['clip_id'].isin(train_ids)])} frame")
print(f"Sentiment — Test:  {len(df_sentiment[df_sentiment['clip_id'].isin(test_ids)])} frame")

# ── CSV basic (exclut Neutral → basic_idx=-1) ───────────────────────
df_basic = df[df["basic_idx"] != -1]
df_basic[df_basic["clip_id"].isin(train_ids)].to_csv(
    os.path.join(SPLITS_PATH, "train_basic.csv"), index=False)
df_basic[df_basic["clip_id"].isin(test_ids)].to_csv(
    os.path.join(SPLITS_PATH, "test_basic.csv"), index=False)
print(f"Basic     — Train: {len(df_basic[df_basic['clip_id'].isin(train_ids)])} frame")
print(f"Basic     — Test:  {len(df_basic[df_basic['clip_id'].isin(test_ids)])} frame")

# ── CSV fine (17 classes canoniques IMEmo) ───────────────────────────
df_fine = df[df["fine_idx"] != -1].copy()
df_fine["emotion_idx"] = df_fine["fine_idx"]
df_fine[df_fine["clip_id"].isin(train_ids)].to_csv(
    os.path.join(SPLITS_PATH, "train_fine.csv"), index=False)
df_fine[df_fine["clip_id"].isin(test_ids)].to_csv(
    os.path.join(SPLITS_PATH, "test_fine.csv"), index=False)
print(f"Fine — Train: {len(df_fine[df_fine['clip_id'].isin(train_ids)])} frame")
print(f"Fine — Test:  {len(df_fine[df_fine['clip_id'].isin(test_ids)])} frame")

# ── Emotion map ─────────────────────────────────────────────────────
with open(os.path.join(SPLITS_PATH, "emotion_map.json"), "w") as f:
    json.dump(EMOTION_MAP, f, indent=2)

print(f"\nSalvati in {SPLITS_PATH}/")
print("emotion_map.json salvato.")
