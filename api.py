import base64
import io
import pickle
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from tensorflow.keras.models import load_model, Model
import uvicorn

# ─── CONFIG ────────────────────────────────────────────────────────────────────
cnn_name             = "MobileNetV2"
h5_model_path        = f"{cnn_name}_steno_model.h5"
pkl_path             = f"{cnn_name}_embeddings_and_indices.pkl"
VALIDATION_THRESHOLD = 0.93  # 93%

equivalents = {
    # your full equivalence map...
}

def is_equivalent(expected: str, predicted: str) -> bool:
    return (
        predicted == expected
        or (expected in equivalents and predicted in equivalents[expected])
    )

# ─── IMAGE PREPROCESSING ───────────────────────────────────────────────────────
def composite_on_white(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    img = img.convert("RGBA")
    bg  = Image.new("RGB", img.size, (255,255,255))
    bg.paste(img, mask=img.split()[3])
    return bg

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = composite_on_white(img)
    img = img.resize((224,224), Image.LANCZOS)
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr,0)

def load_from_base64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    return preprocess_image(img)

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm>1e-10 else vec

# ─── FASTAPI SETUP ──────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"],    allow_headers=["*"],
)

# ─── LOAD MODEL & RECORDS ───────────────────────────────────────────────────────
model     = load_model(h5_model_path, compile=False)
emb_model = Model(inputs=model.inputs,
                  outputs=model.get_layer("embedding_layer").output)

with open(pkl_path,"rb") as f:
    records = pickle.load(f)["records"]

# ─── REQUEST & RESPONSE MODELS ─────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    image: str
    expected_word: str

class PredictionResponse(BaseModel):
    correctness: str       # "Correct" or "Incorrect"
    expected_word: str
    detected_word: str
    accuracy: float

# ─── PREDICT ENDPOINT ──────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    try:
        # 1) preprocess + embed
        x       = load_from_base64(payload.image)
        emb_q   = emb_model.predict(x, verbose=0)[0]
        emb_q_n = l2_normalize(emb_q)

        # 2) compute similarities
        sims = [(float(np.dot(emb_q_n, rec["emb"])), rec["label"])
                for rec in records]
        sims.sort(key=lambda x: x[0], reverse=True)

        # 3) look for expected_word in top-5 above threshold
        top5      = sims[:5]
        exp_lower = payload.expected_word.lower()
        exp_score = next(
            (score for score, lbl in top5
             if lbl.lower()==exp_lower and score>=VALIDATION_THRESHOLD),
            None
        )

        if exp_score is not None:
            # expected was confidently detected
            detected = payload.expected_word
            accuracy = exp_score
            correct  = True
        else:
            # fallback to raw Top-1
            best_score, best_label = sims[0]
            detected  = best_label
            accuracy  = best_score
            correct   = (best_label.lower()==exp_lower
                         and best_score>=VALIDATION_THRESHOLD)

        return {
            "correctness":   "Correct" if correct else "Incorrect",
            "expected_word": payload.expected_word,
            "detected_word": detected,
            "accuracy":      round(accuracy,4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── HEALTH CHECK ──────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status":"alive"}

# ─── RUN ────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","10000")))
