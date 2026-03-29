import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import pandas as pd
import os
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

print("Loading YAMNet model...")
model = hub.load("https://tfhub.dev/google/yamnet/1")

# Load class names
class_map_path = model.class_map_path().numpy().decode("utf-8")
class_names = pd.read_csv(class_map_path)


def predict_audio(file_path):
    waveform, sr = librosa.load(file_path, sr=16000, mono=True)

    scores, embeddings, spectrogram = model(waveform)
    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)

    # ✅ Top 5 logic
    top_indices = np.argsort(mean_scores)[-5:][::-1]

    results = []
    for i in top_indices:
        results.append({
            "label": class_names.iloc[i]['display_name'],
            "confidence": round(float(mean_scores[i]), 3)
        })

    return results


# 🔥 API endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        predictions = predict_audio(file_path)
    except Exception as e:
        return {"error": str(e)}

    # cleanup
    if os.path.exists(file_path):
        os.remove(file_path)

    return {
        "predictions": predictions
    }


# Optional test route
@app.get("/")
def home():
    return {"message": "YAMNet API is running"}
