import base64
import io
import os
import pickle
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model, Model
import uvicorn

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model and Prototypes ────────────────────────────────────────────────
cnn_name     = "MobileNetV2"
h5_model_path = f"{cnn_name}_steno_model.h5"
pkl_path     = f"{cnn_name}_embeddings_and_indices.pkl"
MIN_KNN_SIM  = 0.93

model     = load_model(h5_model_path, compile=False)
emb_model = Model(inputs=model.inputs, outputs=model.get_layer("embedding_layer").output)

with open(pkl_path, "rb") as f:
    data = pickle.load(f)
records = data["records"]

# ─── Equivalence Handling (unchanged) ─────────────────────────────────────────
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
    return (expected == predicted) or (predicted in equivalents.get(expected, []))


# ─── Request / Response Models ────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    image: str           # base64-encoded PNG/JPG
    expected_word: str

class PredictionResponse(BaseModel):
    expected_word: str
    predicted_word: str
    similarity_score: float  # cosine similarity [0–1]


class BatchRequest(BaseModel):
    items: List[PredictionRequest]

class BatchResponse(BaseModel):
    results: List[PredictionResponse]


# ─── Utility Functions ─────────────────────────────────────────────────────────
def preprocess_base64(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("L")
    img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)
    img = ImageOps.crop(img)
    img = ImageOps.pad(img, (224, 224), method=Image.LANCZOS, color=0)
    img = img.convert("RGB")
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, 0)

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
def predict_one(req: PredictionRequest):
    try:
        x = preprocess_base64(req.image)
        emb = emb_model.predict(x, verbose=0)[0]
        emb_n = l2_normalize(emb)

        # compute cosine scores
        sims = [(float(np.dot(emb_n, rec["emb"])), rec["label"]) for rec in records]
        sims.sort(key=lambda t: t[0], reverse=True)

        score, label = sims[0]
        return PredictionResponse(
            expected_word=req.expected_word,
            predicted_word=label,
            similarity_score=round(score, 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    results = []
    for item in batch.items:
        try:
            x = preprocess_base64(item.image)
            emb = emb_model.predict(x, verbose=0)[0]
            emb_n = l2_normalize(emb)

            sims = [(float(np.dot(emb_n, rec["emb"])), rec["label"]) for rec in records]
            sims.sort(key=lambda t: t[0], reverse=True)

            score, label = sims[0]
            results.append(PredictionResponse(
                expected_word=item.expected_word,
                predicted_word=label,
                similarity_score=round(score, 4)
            ))
        except Exception as e:
            # on error, record a dummy failure
            results.append(PredictionResponse(
                expected_word=item.expected_word,
                predicted_word="❌ error",
                similarity_score=0.0
            ))

    return BatchResponse(results=results)


@app.get("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "alive"}


# ─── Run Locally ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
