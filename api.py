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

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and embedding layer
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
all_labels = list(class_indices.keys())

# Word equivalence map
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
    image: str
    expected_word: str

# ⬇️ Enhanced preprocessing for canvas input
def preprocess_base64(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert("L")  # Convert to grayscale

    # Save input for inspection (optional)
    with open("debug_input.png", "wb") as f:
        f.write(base64.b64decode(base64_str))

    image = ImageOps.invert(image)  # Make stroke white on black
    image = ImageOps.pad(image, (224, 224), method=Image.LANCZOS, color=0)  # Center pad to 224x224
    image = image.convert("RGB")
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
        x = preprocess_base64(payload.image)
        emb_q = emb_model.predict(x, verbose=0)[0]
        emb_q_n = l2_normalize(emb_q)

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

# ✅ Health check for uptime monitors
@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "alive"}

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
