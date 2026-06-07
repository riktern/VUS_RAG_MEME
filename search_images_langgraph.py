import os
import argparse
from typing import TypedDict, List, Dict, Any

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, END

# =========================
# ENV
# =========================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "image-rag")
IMAGES_DIR = os.getenv("IMAGES_DIR", "images")

# =========================
# EMBEDDING MODEL
# =========================

embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")

# =========================
# QDRANT
# =========================

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# =========================
# GRAPH STATE
# =========================

class SearchState(TypedDict, total=False):
    query: str
    query_vector: list
    results: List[Dict[str, Any]]

# =========================
# EMBEDDING NODE
# =========================

def embedding_node(state: SearchState):
    query = state["query"]

    vector = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    return {
        "query_vector": vector
    }

# =========================
# QDRANT SEARCH NODE
# =========================

def qdrant_search_node(state: SearchState):
    vector = state["query_vector"]

    results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=3,
        with_payload=True
    )

    points = results.points if results and results.points else []

    formatted = []

    for point in points:
        payload = point.payload or {}

        image_name = payload.get("image_name", "")
        content = payload.get("content", "")

        formatted.append({
            "image_name": image_name,
            "score": float(point.score) if point.score is not None else 0.0,
            "content": content,
        })

    return {
        "results": formatted
    }

# =========================
# RESULT NODE
# =========================

def result_node(state: SearchState):
    print("\n====================")
    print("SEARCH RESULTS")
    print("====================")
    print(f"\nQuery:\n{state['query']}")

    results = state.get("results", [])

    if not results:
        print("\nNo images found.")
        print("\n====================\n")
        return state

    for i, item in enumerate(results, start=1):
        print(f"\n#{i}")
        print(f"Image name: {item['image_name']}")
        print(f"Score: {round(item['score'], 4)}")

    print("\n====================\n")
    return state

# =========================
# BUILD GRAPH
# =========================

graph = StateGraph(SearchState)

graph.add_node("embedding_node", embedding_node)
graph.add_node("qdrant_search_node", qdrant_search_node)
graph.add_node("result_node", result_node)

graph.set_entry_point("embedding_node")
graph.add_edge("embedding_node", "qdrant_search_node")
graph.add_edge("qdrant_search_node", "result_node")
graph.add_edge("result_node", END)

app = graph.compile()

def search_image(query: str, top_k: int = 3):
    """
    Returns top-3 search results from Qdrant.
    Each item contains:
    - image_name
    - score
    - full_path
    """

    result = app.invoke({
        "query": query,
        "top_k": top_k
    })

    results = result.get("results", [])

    final_results = []
    for item in results:
        image_name = item["image_name"]
        full_path = os.path.join(IMAGES_DIR, image_name)

        final_results.append({
            "image_name": image_name,
            "image_path": full_path,
            "score": item["score"],
        })

    return final_results

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query"
    )

    args = parser.parse_args()

    results = search_image(args.query)

    if not results:
        print("No images found.")
    else:
        for i, item in enumerate(results, start=1):
            print(f"#{i}: {item['image_path']} | score={round(item['score'], 4)}")