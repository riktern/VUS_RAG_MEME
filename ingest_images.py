import os
import uuid
import argparse
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image

from google import genai

from dotenv import load_dotenv


load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)

from sentence_transformers import SentenceTransformer


# =========================
# LOAD ENV
# =========================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "image-rag")


# =========================
# GEMINI CLIENT
# =========================

client = genai.Client(api_key=GOOGLE_API_KEY)

for model in client.models.list():
    print(model.name)

# =========================
# EMBEDDING MODEL
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


def create_collection():
    collections = qdrant.get_collections().collections
    names = [c.name for c in collections]

    if QDRANT_COLLECTION not in names:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"[OK] Collection created: {QDRANT_COLLECTION}")
    else:
        print(f"[OK] Collection already exists: {QDRANT_COLLECTION}")


# =========================
# IMAGE ANALYSIS
# =========================

def analyze_image(image_path: str):
    """
    Gemini Vision:
    - OCR
    - visual description
    """

    image = Image.open(image_path)

    prompt = """
    Analyze this image carefully.

    Return:
    1. All text visible in the image (OCR)
    2. Detailed visual description

    Response format:

    OCR:
    ...

    DESCRIPTION:
    ...
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, image]
    )

    return response.text


# =========================
# EMBEDDING
# =========================

def create_embedding(text: str):
    embedding = embedding_model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()


# =========================
# PROCESS SINGLE IMAGE
# =========================

def process_image(image_path: str):

    try:
        # ---- Gemini Analysis ----
        combined_text = analyze_image(image_path)

        if not combined_text:
            print(f"[SKIP] Empty response: {image_path}")
            return None

        # ---- Embedding ----
        vector = create_embedding(combined_text)

        # ---- Payload ----
        payload = {
            "image_name": Path(image_path).name,
            "image_path": str(Path(image_path).absolute()),
            "content": combined_text,
        }

        # ---- Qdrant Point ----
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload,
        )

        return point

    except Exception as e:
        print(f"[SKIP] {Path(image_path).name}: {e}")
        return None


# =========================
# INGEST DIRECTORY
# =========================

def ingest_images(images_dir: str):

    create_collection()

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    ]

    image_paths = []

    for ext in image_extensions:
        image_paths.extend(
            Path(images_dir).glob(f"*{ext}")
        )

    if len(image_paths) == 0:
        print("No images found.")
        return

    points = []

    for image_path in tqdm(
        image_paths,
        desc="Processing images"
    ):

        point = process_image(str(image_path))

        if point:
            points.append(point)
            print(f"[OK] processed {image_path.name}")

    if len(points) == 0:
        print("No points were created.")
        return

    # ---- Upload ----
    qdrant.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )

    print(f"\n[OK] Uploaded {len(points)} images to Qdrant")


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Path to images folder"
    )

    args = parser.parse_args()

    ingest_images(args.images_dir)