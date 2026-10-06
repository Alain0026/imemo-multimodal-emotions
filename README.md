IMEmo — Multimodal Emotion Recognition (DAN fine-tuned + HuBERT + LSTM)

Bachelor's thesis project (Laurea Triennale in Ingegneria Informatica, Università degli Studi di Firenze), developed at the MICC — Media Integration and Communication Center.

Analisi multimodale per il riconoscimento di emozioni durante l'interazione Author: Alain Martial Mbogning · Supervisor: Prof. Stefano Berretti · Co-supervisor: Claudio Ferrari

The project recognizes the emotion of the speaker in dialogue scenes from commercial films, combining the face (visual channel) and the voice (audio channel) over time. It starts from the IMEmo framework (CVPR Workshops 2025), rebuilds its original pipeline on the available data and proposes two changes to the encoders:

Component	Original pipeline	This work
Visual encoder	DAN (frozen)	DAN fine-tuned
Audio encoder	Wav2Vec 2.0	HuBERT
Temporal module	LSTM	LSTM
Architecture

For each segment of a clip:

Visual encoder — DAN (ResNet-18 pre-trained on MS-Celeb-1M + 4 attention heads) turns the faces of the segment into a 512-dim vector, projected to 128-dim.
Audio encoder — HuBERT turns the speech of the segment into a 768-dim vector, projected to 128-dim.
Fusion — the two 128-dim vectors are summed.
Temporal module — an LSTM (hidden size 256) receives the fused vector concatenated with an embedding of the previous segment's emotion.
Classifier — predicts the emotion at three levels: Sentiment, Basic, Fine-grained.
Selective fine-tuning of DAN
Layers	Status	Learning rate
ResNet layers 1–2 (generic features)	Frozen	—
ResNet layers 3–4	Trained	1e-4
4 attention heads	Trained	1e-4
BN + FC	Trained	1e-4
LSTM + classifier	Trained	1e-3
Dataset

The dataset used in this work is distributed in this repository as annotations + video references, following the usual practice for datasets built from YouTube videos: the clips are excerpts of copyrighted commercial films, so the video files themselves are not redistributed. The videos are downloaded directly from YouTube with the provided script.

Overview
100 original IMEmo clips, of which 92 are still available on YouTube
1,069 segments, split 80/20 at clip level (852 train / 217 test): all segments of a clip belong to the same set
Three annotation levels (a Neutral class was added to Basic and Fine-grained):
Level	Classes
Sentiment	3 (positive, negative, neutral)
Basic	7 (6 basic emotions + Neutral)
Fine-grained	17 (16 specific emotions + Neutral)

The label distribution is reported in audit.txt.

What is included

Original IMEmo annotations — one folder per clip in data/raw/IMEmo/ (100 clips, e.g. 052/):

File	Content
<ID>.txt	YouTube URL of the source video and start/end time of the clip
<ID>_Video_annotation.json	Characters in the scene: age, gender, social relation, action
<ID>_Frames_annotation.json	Frame-level annotations
<ID>_Age_character.txt	Short textual description of the characters

Splits and labels used in this work — in data/splits/:

File	Content
train.csv, test.csv	Train/test split (Sentiment level)
train_basic.csv, test_basic.csv	Train/test split, Basic level (7 classes)
train_fine.csv, test_fine.csv	Train/test split, Fine-grained level (17 classes)
train_segments.csv, test_segments.csv	Segment-level data used by the multimodal models
emotion_map.json	Mapping between Fine-grained emotion names and class indices

The video files (.mp4) and all derived data (frames, faces, audio, features) are not included.

How to rebuild the dataset

01_download_videos.py reads the <ID>.txt file of each clip, downloads the source video from YouTube with yt-dlp and cuts the clip between the start and end times:

bash
python scripts/01_download_videos.py   # downloads the clips still available on YouTube (92 of 100)
python scripts/02_extract_frames.py    # then follow the pipeline below (steps 2 → 8)

8 of the 100 original videos are no longer available on YouTube.

### Recovered videos (MICC)

The **92 clips recovered for this work** (`.mp4`, one per clip folder) are stored on the **MICC GPU servers** (Università degli Studi di Firenze). They are kept there because YouTube videos can be removed at any time: this copy guarantees that the experiments remain reproducible even if more source videos disappear.

These files are not publicly redistributed. Members of the MICC lab can ask the supervisor (Prof. Stefano Berretti) for access.

Results (92 clips, same split for all models)

All architectures use the LSTM temporal module.

Model	Sentiment	Basic	Fine
Original pipeline (reproduced)			
DAN frozen (facial-only)	34.10%	9.22%	3.23%
Wav2Vec 2.0 (speech-only)	41.94%	18.89%	4.61%
DAN frozen + Wav2Vec 2.0 + LSTM	66.36%	40.55%	25.81%
Proposed models			
DAN fine-tuned (facial-only)	71.22%	40.00%	26.83%
HuBERT (speech-only)	47.93%	19.35%	4.15%
DAN-FT + HuBERT + LSTM	71.79%	41.03%	27.18%

The proposed model improves on the reproduced original pipeline on all three levels (+5.43, +0.48, +1.37 points). Most of the gain comes from fine-tuning the visual encoder.

Repository structure
.
├── data/
│   ├── raw/IMEmo/            # original IMEmo annotations + YouTube references (100 clips)
│   └── splits/               # train/test splits, segments, emotion map
├── scripts/                  # data pipeline (01 → 08) and training scripts
├── src/
│   ├── training/             # trainer and metrics
│   └── utils/                # visualization utilities
├── assets/fonts/             # fonts used for the PDF reports
├── config.yaml               # paths and hyper-parameters
├── dan_modifications.patch   # changes applied to the original DAN code
├── audit.txt                 # label distribution of the dataset
└── requirements.txt
Installation

Tested with Python 3.12 on Linux with 2× NVIDIA RTX 2080 Ti.

bash
git clone git@github.com:<username>/imemo-multimodal-emotion.git
cd imemo-multimodal-emotion
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
DAN (third-party code + patch)

DAN is not copied into this repository. Clone it at the exact version used in this work and apply the patch:

bash
git clone https://github.com/yaoing/DAN
cd DAN
git checkout 3872eefef1441231342b1a798719f6e69cdbc1fd
git apply ../dan_modifications.patch
cd ..

The patch modifies networks/dan.py and rafdb.py.


The pre-trained ResNet-18 weights (MS-Celeb-1M) must be downloaded as described in the DAN repository and placed in models/resnet18_msceleb.pth.

Pipeline

The pipeline starts from the annotations in data/raw/IMEmo/. Paths and parameters are set in config.yaml.

Step	Script	Purpose
1	01_download_videos.py	Download the clips (yt-dlp)
2	02_extract_frames.py	Extract frames from the videos
3	03_create_labels_csv.py	Build the label CSV
4	04_detect_faces.py	Face detection and cropping (MTCNN)
5	05_extract_audio.py	Extract the audio track
6	06_extract_hubert_features.py / 06_extract_wav2vec_features.py	Audio features
7	07_extract_visual_features.py / 07_extract_visual_features_finetuned.py	Visual features (DAN frozen / fine-tuned)
8	08_create_segments_csv.py	Build the segment-level CSV

Training:

Script	Experiment
train_dan_finetuning.py	Fine-tuning of DAN
train_multimodal_original.py	Original pipeline: DAN frozen + Wav2Vec 2.0 + LSTM
train_multimodal_finetuned.py	Proposed model: DAN-FT + HuBERT + LSTM

bash
python scripts/01_download_videos.py
# ...
python scripts/train_multimodal_finetuned.py

01_download_videos.py may require a YouTube cookies.txt file. It is personal and is never committed (see .gitignore).

References
IMEmo — CVPR Workshops 2025. 
DAN — Z. Wen et al., Distract Your Attention: Multi-head Cross Attention Network for Facial Expression Recognition, arXiv:2109.07270. Code: https://github.com/yaoing/DAN
HuBERT — W.-N. Hsu et al., HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units, 2021.
Wav2Vec 2.0 — A. Baevski et al., wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations, NeurIPS 2020.
Acknowledgements

Thanks to Prof. Stefano Berretti and Claudio Ferrari for their supervision, and to the MICC for providing the GPU servers.
