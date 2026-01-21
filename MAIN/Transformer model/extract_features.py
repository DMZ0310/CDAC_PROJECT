import os
import torch
import librosa
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV = "transcripts.csv"
AUDIO_DIR = r"C:\Users\Dennismz\Desktop\CDAC_PROJECT\dataset for SER\CREMA-D\AudioWAV"
OUT_DIR = "features"
os.makedirs(OUT_DIR, exist_ok=True)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
model.eval()

df = pd.read_csv(CSV)

for _, row in tqdm(df.iterrows(), total=len(df)):
    wav_path = os.path.join(AUDIO_DIR, row["file"])
    save_path = os.path.join(OUT_DIR, row["file"].replace(".wav", ".pt"))

    if os.path.exists(save_path):
        continue

    y, _ = librosa.load(wav_path, sr=16000)
    inputs = processor(y, sampling_rate=16000, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        features = model(**inputs).last_hidden_state.squeeze(0).cpu()

    torch.save(features, save_path)

print("✅ Feature extraction completed")
