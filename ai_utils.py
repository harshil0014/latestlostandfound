# ai_utils.py
# Tiny helper so our app can "look" and "read"

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# ---- Load the CLIP model only once ----
_model = None


def _load_model():
    """Make sure the CLIP model is ready."""
    global _model
    if _model is None:
        # This downloads (~400 MB) the first time and then caches it.
        _model = SentenceTransformer("clip-ViT-B-32")


# ---- Public helpers ----
def get_image_vec(image_path: str) -> np.ndarray:
    """Turn a photo file into a 512-number vector."""
    _load_model()
    img = Image.open(image_path).convert("RGB")
    # encode returns a list; we take [0] to unwrap
    vec = _model.encode([img], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec


def get_text_vec(text: str) -> np.ndarray:
    """Turn a sentence into a 512-number vector."""
    _load_model()
    vec = _model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """0 = totally different, 1 = twins."""
    # a·b  /  (|a| * |b|)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
