import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from google import genai

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)

from sentence_transformers import SentenceTransformer

# =========================
# ENV
# =========================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "image-rag"
)

# =========================
# GEMINI
# =========================

client = genai.Client(
    api_key=GOOGLE_API_KEY
)

# =========================
# EMBEDDINGS
# =========================

embedding_model = SentenceTransformer(
    "intfloat/multilingual-e5-small"
)

VECTOR_SIZE = 384

# =========================
# QDRANT
# =========================

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# =========================
# COLLECTION
# =========================

def create_collection():

    collections = qdrant.get_collections().collections

    names = [
        c.name
        for c in collections
    ]

    if QDRANT_COLLECTION not in names:

        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"[OK] Collection created: "
            f"{QDRANT_COLLECTION}"
        )

# =========================
# GEMINI ANALYSIS
# =========================

def analyze_image(image_path):

    image = Image.open(image_path)

    prompt = """
Analyze this image.

Return:

1. OCR text
2. Detailed visual description

Format:

OCR:
...

DESCRIPTION:
...
"""

    last_error = None

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    prompt,
                    image
                ]
            )

            return response.text

        except Exception as e:

            last_error = e

            print(
                f"[RETRY {attempt+1}] "
                f"{e}"
            )

            time.sleep(5)

    raise last_error

# =========================
# EMBEDDING
# =========================

def create_embedding(text):

    vector = embedding_model.encode(
        text,
        normalize_embeddings=True
    )

    return vector.tolist()

# =========================
# PROCESS IMAGE
# =========================

def process_image(image_path):

    combined_text = analyze_image(
        image_path
    )

    vector = create_embedding(
        combined_text
    )

    payload = {
        "image_name":
            Path(image_path).name,

        "content":
            combined_text,
    }

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload=payload,
    )

    return point

# =========================
# INGEST ONE IMAGE
# =========================

def ingest_single_image(
    image_path
):

    try:

        create_collection()

        point = process_image(
            image_path
        )

        qdrant.upsert(
            collection_name=
                QDRANT_COLLECTION,
            points=[point]
        )

        print(
            f"[OK] Uploaded "
            f"{Path(image_path).name}"
        )

        return True

    except Exception as e:

        print(
            f"[ERROR] "
            f"{e}"
        )

        return False