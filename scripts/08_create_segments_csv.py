"""
Crea CSV a livello di segmento (non di frame).
Ogni riga = un shot/segmento con i suoi indici di frame.
"""
import os, json, random
import pandas as pd

DATASET_PATH = "data/raw/IMEmo"
SPLITS_PATH  = "data/splits"

SENTIMENT_MAP = {"Positive": 0, "Negative": 1, "Neutral": 2}

BASIC_MAP = {
    "Anger":    0, "Angry":     0,
    "Disgust":  1,
    "Fear":     2,
    "Happy":    3, "Happiness": 3,
    "Sadness":  4, "Sad":       4,
    "Surprise": 5,
    "Neutral":  6,
}

FINE_MAP = {
    "Sadness":       0, "Hurt":          0,
    "Disappointment":0, "Regret":        0, "Pain":          0,
    "Shame":         1, "Embarrassment": 1, "Shyness":       1,
    "Fear":          2,
    "Distress":      3, "Frustration":   3,
    "Confusion":     3, "Concern":       3,
    "Anger":         4, "Defensiveness": 4, "Angry":         4,
    "Contempt":      5,
    "Rage":          6,
    "Disgust":       7,
    "Wonder":        8, "Curiosity":     8,
    "Thoughtfulness":8, "Reflection":    8,
    "Desire":        9,
    "Surprise":     10,
    "Anxiety":      11,
    "Love":         12, "Affection":    12, "Warmth":       12,
    "Empathy":      12, "Compassion":   12, "Reassurance":  12,
    "Encouragement":12,
    "Happy":        13, "Happiness":    13, "Contentment":  13,
    "Satisfaction": 13, "Relief":       13, "Confidence":   13,
    "Joy":          14, "Amusement":    14,
    "Interest":     15, "Attention":    15, "Acceptance":   15,
    "Neutral":      16, "Explanation":  16, "Seriousness":  16,
    "Patience":     16, "Politeness":   16, "Understanding":16,
}

TYPO_FIX = {"Sadnesss": "Sadness"}

SKIP_CLIPS = {
    "015","016","017","018","019","020","023","076"
}

records = []
for clip in sorted(os.listdir(DATASET_PATH)):
    clip_path = os.path.join(DATASET_PATH, clip)
    if not os.path.isdir(clip_path) or clip in SKIP_CLIPS:
        continue
    json_files = [f for f in os.listdir(clip_path)
                  if "Frames_annotation" in f]
    if not json_files:
        continue
    with open(os.path.join(clip_path, json_files[0])) as f:
        shots = json.load(f)

    for i, shot in enumerate(shots):
        raw_emotion = shot.get("Emotion", "")
        emotion     = TYPO_FIX.get(raw_emotion, raw_emotion)
        raw_basic   = shot.get("Basic_Emotion", "")
        basic       = TYPO_FIX.get(raw_basic, raw_basic)
        sentiment   = shot.get("Sentiment", "")

        records.append({
            "clip_id":       clip,
            "segment_id":    i,
            "frame_from":    int(shot["Frame From"].split("_")[-1]),
            "frame_to":      int(shot["Frame to"].split("_")[-1]),
            "sentiment_idx": SENTIMENT_MAP.get(sentiment, -1),
            "basic_idx":     BASIC_MAP.get(basic, -1),
            "fine_idx":      FINE_MAP.get(emotion, -1),
        })

df = pd.DataFrame(records)
print(f"Totale segmenti: {len(df)}")

random.seed(42)
clip_ids = df["clip_id"].unique().tolist()
random.shuffle(clip_ids)
n_train   = int(len(clip_ids) * 0.80)
train_ids = set(clip_ids[:n_train])
test_ids  = set(clip_ids[n_train:])

for suffix, ids in [("train", train_ids), ("test", test_ids)]:
    sub = df[df["clip_id"].isin(ids)]
    sub.to_csv(f"{SPLITS_PATH}/{suffix}_segments.csv", index=False)
    print(f"{suffix}: {len(sub)} segmenti ({len(ids)} clip)")

print("CSV segmenti salvati.")
