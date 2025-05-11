# api.py

import base64
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.models import load_model, Model
import uvicorn

app = FastAPI()

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and data
cnn_name = "MobileNetV2"
h5_model_path = f"{cnn_name}_steno_model.h5"
pkl_path = f"{cnn_name}_embeddings_and_indices.pkl"
MIN_KNN_SIM = 0.93

model = load_model(h5_model_path)
emb_model = Model(inputs=model.inputs, outputs=model.get_layer("embedding_layer").output)

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

class_indices = data["class_indices"]
records = data["records"]
all_labels = list(class_indices.keys())

# Word equivalences (synonyms or variants)
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

class PredictionRequest(BaseModel):
    image: str  # base64 string
    expected_word: str

def preprocess_base64(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((224, 224), resample=Image.LANCZOS)
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec

def is_equivalent(expected, predicted):
    expected = expected.lower().strip()
    predicted = predicted.lower().strip()
    return predicted == expected or predicted in equivalents.get(expected, [])

@app.post("/predict")
def predict(payload: PredictionRequest):
    try:
        # Preprocess image
        x = preprocess_base64(payload.image)
        emb_q = emb_model.predict(x, verbose=0)[0]
        emb_q_n = l2_normalize(emb_q)

        # Find most similar
        sims = [(float(np.dot(emb_q_n, rec["emb"])), rec["label"]) for rec in records]
        sims.sort(key=lambda t: t[0], reverse=True)

        best_score, best_label = sims[0]
        matched = is_equivalent(payload.expected_word, best_label)

        return {
            "expected_word": payload.expected_word,
            "predicted_word": best_label,
            "similarity_score": round(best_score, 4),
            "match": matched
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
