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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cnn_name     = "MobileNetV2"
h5_model_path = f"{cnn_name}_steno_model.h5"
pkl_path     = f"{cnn_name}_embeddings_and_indices.pkl"

model     = load_model(h5_model_path, compile=False)
emb_model = Model(inputs=model.inputs, outputs=model.get_layer("embedding_layer").output)

with open(pkl_path, "rb") as f:
    data = pickle.load(f)
records = data["records"]

equivalents = {
    "a": ["a", "an"], "an": ["a", "an"],
    # ... rest of your map ...
}

def is_equivalent(expected: str, predicted: str) -> bool:
    return (expected == predicted) or (predicted in equivalents.get(expected, []))

class PredictionRequest(BaseModel):
    image: str
    expected_word: str

class PredictionResponse(BaseModel):
    expected_word: str
    predicted_word: str
    similarity_score: float

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

@app.post("/predict", response_model=PredictionResponse)
def predict_one(req: PredictionRequest):
    try:
        x    = preprocess_base64(req.image)
        emb  = emb_model.predict(x, verbose=0)[0]
        embn = l2_normalize(emb)

        sims = [(float(np.dot(embn, r["emb"])), r["label"]) for r in records]
        sims.sort(key=lambda t: t[0], reverse=True)

        score, label = sims[0]
        return PredictionResponse(
            expected_word=req.expected_word,
            predicted_word=label,
            similarity_score=round(score, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "alive"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
