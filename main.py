

import io
import os
import warnings
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

warnings.filterwarnings("ignore")


# Config

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "./face_auth_model.pth")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                                   
# Model architecture  — must match face_auth_training.ipynb exactly


class FaceEmbedder(nn.Module):
    """InceptionResnetV1 backbone + projection head (LayerNorm, safe for batch=1)."""

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.backbone = InceptionResnetV1(pretrained=None)
        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        emb = self.projection(emb)
        return nn.functional.normalize(emb, p=2, dim=1)



# Face detector  — MTCNN if cached locally, Haar cascade fallback


_MTCNN_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "checkpoints")
_MTCNN_CACHED = all(
    os.path.exists(os.path.join(_MTCNN_CACHE_DIR, w))
    for w in ["pnet.pt", "rnet.pt", "onet.pt"]
)

_haar_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _detect_faces_haar(pil_image: Image.Image) -> List[List[int]]:
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _haar_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return []
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return [[x, y, x + w, y + h] for x, y, w, h in faces]


def _detect_face(pil_image: Image.Image, mtcnn=None) -> Optional[List[int]]:
    """Return [x1, y1, x2, y2] of the most prominent face, or None."""
    if mtcnn is not None:
        try:
            boxes, _ = mtcnn.detect(pil_image.convert("RGB"))
            if boxes is not None and len(boxes) > 0:
                return [int(v) for v in boxes[0]]
        except Exception:
            pass
    boxes = _detect_faces_haar(pil_image)
    return boxes[0] if boxes else None



# Global inference state


_model: Optional[FaceEmbedder] = None
_mtcnn = None
_threshold: float = 0.6
_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])



# Model loading


def _load_model(checkpoint_path: str) -> None:
    """Load weights from checkpoint into global state. Raises FileNotFoundError."""
    global _model, _threshold, _mtcnn

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    embedding_dim = checkpoint.get("config", {}).get("embedding_dim", 512)

    model = FaceEmbedder(embedding_dim=embedding_dim).to(DEVICE)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing:
        print(f"  [warn] Missing keys   : {missing[:5]}")
    if unexpected:
        print(f"  [warn] Unexpected keys: {unexpected[:5]}")
    model.eval()

    _model = model
    _threshold = checkpoint.get("threshold", 0.6)

    if _MTCNN_CACHED:
        from facenet_pytorch import MTCNN as _MTCNN_cls
        _mtcnn = _MTCNN_cls(keep_all=False, device=DEVICE, min_face_size=20, post_process=False)
        detector = "MTCNN (cached)"
    else:
        _mtcnn = None
        detector = "OpenCV Haar (offline fallback)"

    print(f"[OK] Model loaded from '{checkpoint_path}'")
    print(f"     epoch={checkpoint.get('epoch','?')}  loss={checkpoint.get('loss','?')}  "
          f"threshold={_threshold:.2f}  device={DEVICE}  detector={detector}")


def _ensure_model() -> None:
    """Lazy-load model on first request. Returns HTTP 503 if checkpoint missing."""
    if _model is not None:
        return
    try:
        _load_model(CHECKPOINT_PATH)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model checkpoint not found: '{CHECKPOINT_PATH}'. "
                "Run face_auth_training.ipynb to generate face_auth_model.pth, "
                "place it next to main.py, then retry."
            ),
        )



# Inference


def _embed(pil_image: Image.Image) -> Tuple[Optional[torch.Tensor], Optional[List[int]]]:
    img_rgb = pil_image.convert("RGB")
    bbox = _detect_face(img_rgb, _mtcnn)
    if bbox is None:
        return None, None

    x1, y1, x2, y2 = bbox
    w, h = img_rgb.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None, None

    tensor = _transform(img_rgb.crop((x1, y1, x2, y2))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = _model(tensor).squeeze(0)
    return embedding, [x1, y1, x2, y2]


def predict(
    image1: Image.Image,
    image2: Image.Image,
    threshold: Optional[float] = None,
) -> Dict:
    thresh = threshold if threshold is not None else _threshold
    emb1, bbox1 = _embed(image1)
    emb2, bbox2 = _embed(image2)

    if emb1 is None or emb2 is None:
        missing = (["image1"] if emb1 is None else []) + (["image2"] if emb2 is None else [])
        return {
            "verification_result": f"face not detected in {', '.join(missing)}",
            "similarity_score": None,
            "bounding_boxes": {"image1": bbox1, "image2": bbox2},
            "threshold_used": thresh,
        }

    similarity = nn.functional.cosine_similarity(
        emb1.unsqueeze(0), emb2.unsqueeze(0)
    ).item()

    return {
        "verification_result": "same person" if similarity >= thresh else "different person",
        "similarity_score": round(similarity, 4),
        "bounding_boxes": {"image1": bbox1, "image2": bbox2},
        "threshold_used": thresh,
    }



# FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Try loading at startup — skip gracefully if checkpoint not ready yet
    if os.path.exists(CHECKPOINT_PATH):
        try:
            _load_model(CHECKPOINT_PATH)
        except Exception as exc:
            print(f"[warn] Could not load model at startup: {exc}")
    else:
        print(f"[warn] '{CHECKPOINT_PATH}' not found — server starting without model.")
        print("       Generate the checkpoint with face_auth_training.ipynb,")
        print("       place it next to main.py, then POST to /verify.")
    yield


app = FastAPI(
    title="Face Authentication API",
    description="Verify whether two face images belong to the same person.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


@app.get("/health", tags=["Utilities"], summary="Liveness and model status")
async def health():
    return {
        "status": "ok",
        "model": "loaded" if _model is not None else "not loaded",
        "checkpoint_found": os.path.exists(CHECKPOINT_PATH),
        "detector": "MTCNN" if _mtcnn is not None else "Haar (offline fallback)",
        "threshold": _threshold,
        "device": str(DEVICE),
    }


@app.post("/verify", tags=["Face Verification"], summary="Compare two face images")
async def verify(
    image1: UploadFile = File(..., description="First face image (JPEG/PNG/WEBP)"),
    image2: UploadFile = File(..., description="Second face image (JPEG/PNG/WEBP)"),
    threshold: Optional[float] = Query(
        default=None, ge=0.0, le=1.0,
        description="Cosine similarity threshold override (0-1)"
    ),
):
    # Lazy-load model if not yet loaded
    _ensure_model()

    # Validate content types
    for upload, label in [(image1, "image1"), (image2, "image2")]:
        if upload.content_type not in _ALLOWED_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"{label}: unsupported type '{upload.content_type}'. "
                       f"Accepted: jpeg, jpg, png, webp",
            )

    # Decode images
    try:
        pil1 = Image.open(io.BytesIO(await image1.read()))
        pil2 = Image.open(io.BytesIO(await image2.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode images: {exc}")

    # Run inference
    try:
        result = predict(pil1, pil2, threshold=threshold)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(content=result)



# Entry point


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, log_level="info")
