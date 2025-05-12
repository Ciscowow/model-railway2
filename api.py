import base64
import io
import os
import pickle
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from tensorflow.keras.models import load_model, Model
import uvicorn

# ─── CONFIG ────────────────────────────────────────────────────────────────────
cnn_name      = "MobileNetV2"
h5_model_path = f"{cnn_name}_steno_model.h5"
pkl_path      = f"{cnn_name}_embeddings_and_indices.pkl"
MIN_KNN_SIM   = 0.95
TOP_K         = 5

# ─── FASTAPI SETUP ──────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── LOAD MODEL & PROTOTYPES ────────────────────────────────────────────────────
model = load_model(h5_model_path, compile=False)
emb_model = Model(inputs=model.inputs, outputs=model.get_layer("embedding_layer").output)

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

class_indices = data["class_indices"]
records       = data["records"]

equivalents = {
    "a": ["a", "an"], "an": ["a", "an"],
    "are": ["are", "our", "hour"], "our": ["are", "our", "hour"], "hour": ["are", "our", "hour"],
    "at": ["at", "it"], "it": ["at", "it"],
    "be": ["be", "by"], "by": ["be", "by"],
    "correspond": ["correspond", "correspondence"], "correspondence": ["correspond", "correspondence"],
    "ever": ["ever", "every"], "every": ["ever", "every"],
    "important": ["important", "importance"], "importance": ["important", "importance"],
    "in": ["in", "not"], "not": ["in", "not"],
    "is": ["is", "his"], "his": ["is", "his"],
    "publish": ["publish", "publication"], "publication": ["publish", "publication"],
    "satisfy": ["satisfy", "satisfactory"], "satisfactory": ["satisfy", "satisfactory"],
    "their": ["their", "there"], "there": ["their", "there"],
    "thing": ["thing", "think"], "think": ["thing", "think"],
    "well": ["well", "will"], "will": ["well", "will"],
    "won": ["won", "one"], "one": ["won", "one"],
    "you": ["you", "your"], "your": ["you", "your"],
}

def is_equivalent(expected: str, predicted: str) -> bool:
    return (
        predicted == expected
        or (expected in equivalents and predicted in equivalents[expected])
    )

# ─── IMAGE PREPROCESSING ───────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, 0)

def load_from_base64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    return preprocess_image(img)

def load_from_path(path: str) -> np.ndarray:
    img = Image.open(path)
    return preprocess_image(img)

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec

# ─── REQUEST MODEL ─────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    image: str
    expected_word: str

# ─── SINGLE-IMAGE ENDPOINT ─────────────────────────────────────────────────────
@app.post("/predict")
def predict(payload: PredictionRequest):
    try:
        x = load_from_base64(payload.image)
        emb_q = emb_model.predict(x, verbose=0)[0]
        emb_q_n = l2_normalize(emb_q)

        sims = [(float(np.dot(emb_q_n, rec["emb"])), rec["label"]) for rec in records]
        sims.sort(key=lambda t: t[0], reverse=True)

        top_matches = [{"word": lbl, "score": round(score, 4)} for score, lbl in sims[:TOP_K]]
        best_score, best_label = sims[0]

        return {
            "expected_word": payload.expected_word,
            "predicted_word": best_label,
            "similarity_score": round(best_score, 4),
            "top_5": top_matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── BATCH-IMAGE ENDPOINT ──────────────────────────────────────────────────────
@app.get("/batch_predict")
def batch_predict(images_dir: str = Query(..., description="Directory of PNG images")):
    if not os.path.isdir(images_dir):
        raise HTTPException(status_code=404, detail="Directory not found")

    total = correct1 = correct_k = 0

    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(".png"):
            continue

        total += 1
        expected = os.path.splitext(fname)[0].lower()
        path = os.path.join(images_dir, fname)

        x = load_from_path(path)
        emb_q = emb_model.predict(x, verbose=0)[0]
        emb_q_n = l2_normalize(emb_q)

        raw = [(float(np.dot(emb_q_n, rec["emb"])), rec["label"]) for rec in records]
        # keep best sim per label
        best_map = {}
        for sim, lbl in raw:
            if lbl not in best_map or sim > best_map[lbl]:
                best_map[lbl] = sim
        sims = sorted(best_map.items(), key=lambda x: x[1], reverse=True)

        best_lbl, best_sim = sims[0]
        top_labels = [lbl for lbl, _ in sims[:TOP_K]]

        ok1 = (best_sim >= MIN_KNN_SIM) and is_equivalent(expected, best_lbl)
        okK = any(is_equivalent(expected, lbl) for lbl in top_labels)

        correct1 += int(ok1)
        correct_k += int(okK)

    return {
        "status":       "completed",
        "total_images": total,
        "top1_correct": correct1,
        "topK_correct": correct_k
    }

# ─── HEALTH CHECK ──────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "alive"}

# ─── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
