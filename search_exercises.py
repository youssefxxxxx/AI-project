# search_exercises.py
import os
import logging
from sentence_transformers import SentenceTransformer
import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 4  # Always fetch 4 exercises

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client_chroma = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "./chroma_db_exercises"))
collection = client_chroma.get_or_create_collection(name="exercises_embeddings")

def search_exercises(query):
    if not query:
        raise ValueError("Query cannot be empty.")

    query_embedding = model.encode([query])[0]
    logger.info(f"Query embedding (truncated): {query_embedding[:5]}")

    # We will always fetch 4 exercises
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=DEFAULT_TOP_K,
        include=["embeddings", "metadatas"]
    )

    final_results = []
    exercise_embeddings = []

    if results and results.get("metadatas") and results["metadatas"][0]:
        for i, metadata in enumerate(results["metadatas"][0]):
            metadata["id"] = metadata.get("id", f"exercise_{i}")
            final_results.append(metadata)
            exercise_embeddings.append(results["embeddings"][0][i])
    else:
        logger.info("No results from Chroma.")

    logger.info("Final results:")
    for res in final_results:
        logger.info(res)

    return final_results, exercise_embeddings, query_embedding
