import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import easyocr
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "image-rag")
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "./images")).resolve()

VECTOR_MODEL_NAME = os.getenv("VECTOR_MODEL_NAME", "intfloat/multilingual-e5-small")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = SentenceTransformer(VECTOR_MODEL_NAME)

_ocr_reader = None
_caption_pipe = None


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["ru", "en"], gpu=False)
    return _ocr_reader


def get_caption_pipe():
    global _caption_pipe
    if _caption_pipe is None:
        _caption_pipe = pipeline(
            "image-to-text",
            model=os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-base"),
        )
    return _caption_pipe


def ensure_collection() -> None:
    collections = qdrant.get_collections().collections
    if QDRANT_COLLECTION not in [c.name for c in collections]:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def embed_text(text: str) -> List[float]:
    return embedding_model.encode(text, normalize_embeddings=True).tolist()


def extract_ocr(image_path: str) -> str:
    reader = get_ocr_reader()
    result = reader.readtext(image_path)
    return " ".join([item[1] for item in result]).strip()


def generate_caption(image_path: str) -> str:
    pipe = get_caption_pipe()
    result = pipe(image_path)
    if not result:
        return ""
    return result[0].get("generated_text", "").strip()


def analyze_image_local(image_path: str) -> str:
    ocr_text = extract_ocr(image_path)
    caption = generate_caption(image_path)
    return f"OCR:\n{ocr_text}\n\nDESCRIPTION:\n{caption}".strip()


def add_point_to_qdrant(
    *,
    image_name: str,
    image_path: str,
    content: str,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> str:
    ensure_collection()
    point_id = str(uuid.uuid4())
    payload: Dict[str, Any] = {
        "image_name": image_name,
        "image_path": str(Path(image_path).resolve()),
        "content": content,
    }
    if extra_payload:
        payload.update(extra_payload)

    point = PointStruct(
        id=point_id,
        vector=embed_text(content),
        payload=payload,
    )
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=[point])
    return point_id


def search_best_image(query: str, limit: int = 1) -> List[Dict[str, Any]]:
    ensure_collection()
    q = embed_text(query)
    results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=q,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    points = getattr(results, "points", []) or []
    out: List[Dict[str, Any]] = []
    for p in points:
        out.append(
            {
                "id": str(p.id),
                "score": float(getattr(p, "score", 0.0) or 0.0),
                "payload": p.payload or {},
            }
        )
    return out


def normalize_local_image_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((IMAGES_DIR / p).resolve())


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
