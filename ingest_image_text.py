import argparse
from pathlib import Path

from image_rag_core import add_point_to_qdrant


def ingest_one(image_path: str, description: str) -> None:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    if not description.strip():
        raise ValueError("Description is empty")

    add_point_to_qdrant(
        image_name=p.name,
        image_path=str(p.resolve()),
        content=description.strip(),
        extra_payload={"source": "user_text_only"},
    )
    print(f"[OK] Added: {p.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--description", required=True)
    args = parser.parse_args()
    ingest_one(args.image_path, args.description)
