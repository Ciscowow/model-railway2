import base64
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps
import numpy as np
import pickle
from tensorflow.keras.models import load_model, Model
import uvicorn
import os

# Reduce TensorFlow logging noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = FastAPI()

# Enable CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and embeddings
cnn_name = "MobileNetV2"
h5_model_path = f"{cnn_name}_steno_model.h5"
pkl_path = f"{cnn_name}_embeddings_and_indices.pkl"
MIN_KNN_SIM = 0.95

model = load_model(h5_model_path, compile=False)
emb_model = Model(inputs=model.inputs, outputs=model.get_layer("embedding_layer").output)

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

class_indices = data["class_indices"]
records = data["records"]

# API schema
class PredictionRequest(BaseModel):
    image: str
    expected_word: str

# === Preprocess React Native input ===
def preprocess_base64(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert("L")

    image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)
    image = ImageOps.crop(image)
    image = ImageOps.pad(image, (224, 224), method=Image.LANCZOS, color=0)
    image = image.convert("RGB")

    # Save for debugging (optional)
    image.save("last_processed.png")

    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec

@app.post("/predict")
def predict(payload: PredictionRequest):
    try:
        x = preprocess_base64(payload.image)
        emb_q = emb_model.predict(x, verbose=0)[0]
        emb_q_n = l2_normalize(emb_q)

        sims = [(float(np.dot(emb_q_n, rec["emb"])), rec["label"]) for rec in records]
        sims.sort(key=lambda t: t[0], reverse=True)

        best_score, best_label = sims[0]

        return {
            "expected_word": payload.expected_word,
            "predicted_word": best_label,
            "similarity_score": round(best_score, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check for uptime robots
@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "alive"}

# Local run
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
