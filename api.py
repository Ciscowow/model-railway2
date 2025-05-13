import base64
import io
import pickle

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from tensorflow.keras.models import load_model, Model
import uvicorn

# ─── CONFIG ────────────────────────────────────────────────────────────────────
cnn_name      = "MobileNetV2"
h5_model_path = f"{cnn_name}_steno_model.h5"
pkl_path      = f"{cnn_name}_embeddings_and_indices.pkl"

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

# ─── IMAGE PREPROCESSING ────────────────────────────────────────────────────────
def composite_on_white(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    img = img.convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = composite_on_white(img)
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, 0)

def load_from_base64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    return preprocess_image(img)

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec

# ─── FASTAPI SETUP ──────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── LOAD MODEL & RECORDS ───────────────────────────────────────────────────────
model = load_model(h5_model_path, compile=False)
emb_model = Model(inputs=model.inputs, outputs=model.get_layer("embedding_layer").output)

with open(pkl_path, "rb") as f:
    records = pickle.load(f)["records"]

# ─── REQUEST MODEL ──────────────────────────────────────────────────────────────
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

        # find best match
        best_score = -1.0
        best_label = None
        for rec in records:
            score = float(np.dot(emb_q_n, rec["emb"]))
            if score > best_score:
                best_score, best_label = score, rec["label"]

        correctness = is_equivalent(payload.expected_word.lower(), best_label.lower())

        return {
            "predicted_word": best_label,
            "accuracy": round(best_score, 4),
            "correctness": correctness
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── HEALTH CHECK ──────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "alive"}

# ─── RUN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
