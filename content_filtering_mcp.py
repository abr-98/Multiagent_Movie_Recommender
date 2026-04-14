import chromadb
import os
import pandas as pd 
from typing import Any
import re

from sentence_transformers import CrossEncoder, util
from fastmcp import FastMCP
from pydantic import BaseModel, Field
import ssl
import certifi
import sys
import logging

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '60')

ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

mcp = FastMCP("Content_Filtering")

cross_encoder = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("content_mcp")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(os.path.join(LOG_DIR, "content_mcp.log"), encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)


def _log(message: str) -> None:
    logger.info(message)
    print(message, file=sys.stderr)


def get_cross_encoder():
    global cross_encoder
    if cross_encoder is not None:
        return cross_encoder

    try:
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', trust_remote_code=True)
        return cross_encoder
    except Exception as exc:
        # Keep serving recommendations even when model download/network access times out.
        _log(f"Warning: CrossEncoder unavailable ({exc}). Falling back to distance-based ranking.")
        return None



persist_dir_cache = "Chroma_cache"

os.makedirs(persist_dir_cache, exist_ok=True)

chroma_db_path = 'ChromaDBData'

client = chromadb.PersistentClient(path=chroma_db_path)

client_cache = chromadb.PersistentClient(path=persist_dir_cache)

class RecommendationRequest(BaseModel):
    query: str = Field(..., description="The user's query for movie recommendations.")

class RecommendationResponse(BaseModel):
    ranked_results: list[str] = Field(..., description="A list of recommended movie titles.")


def _extract_titles(rows: list[dict[str, Any]]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for row in rows:
        title = ""
        metadata = row.get("Metadatas")
        if isinstance(metadata, dict):
            for key in ("title", "movie_title", "movie_name", "name"):
                value = metadata.get(key)
                if value:
                    title = str(value).strip()
                    break
        if not title:
            doc = row.get("Documents")
            if isinstance(doc, str) and doc.strip():
                title = doc.strip().split("\n")[0][:120]
        if not title:
            movie_id = row.get("IDs")
            if movie_id:
                title = str(movie_id).strip()
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def _empty_query_result() -> dict[str, list[list[Any]]]:
    return {
        'ids': [[]],
        'documents': [[]],
        'metadatas': [[]],
        'distances': [[]],
    }


def _safe_chroma_query(collection: Any, query: str, n_results: int) -> dict[str, Any] | None:
    try:
        return collection.query(query_texts=[query], n_results=n_results)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}".lower()
        if "readtimeout" in msg or "timed out" in msg:
            _log(f"Warning: Chroma query timed out: {exc}")
            return None
        raise


def _fallback_lexical_rank(collection: Any, query: str, n_results: int = 5) -> pd.DataFrame:
    try:
        payload = collection.get(include=['documents', 'metadatas'])
    except Exception as exc:
        _log(f"Warning: fallback retrieval failed: {exc}")
        return pd.DataFrame(columns=['IDs', 'Documents', 'Distances', 'Metadatas'])

    ids = payload.get('ids') or []
    docs = payload.get('documents') or []
    metas = payload.get('metadatas') or []

    if not ids or not docs:
        return pd.DataFrame(columns=['IDs', 'Documents', 'Distances', 'Metadatas'])

    if not metas:
        metas = [None] * len(ids)

    query_tokens = set(re.findall(r"\w+", query.lower()))
    scored_rows = []

    for doc_id, doc_text, metadata in zip(ids, docs, metas):
        text = str(doc_text or "")
        doc_tokens = set(re.findall(r"\w+", text.lower()))
        overlap = len(query_tokens & doc_tokens)
        distance = 1.0 / (1 + overlap)
        scored_rows.append((distance, doc_id, text, metadata))

    scored_rows.sort(key=lambda row: row[0])
    top_rows = scored_rows[:n_results]

    return pd.DataFrame({
        'IDs': [row[1] for row in top_rows],
        'Documents': [row[2] for row in top_rows],
        'Distances': [row[0] for row in top_rows],
        'Metadatas': [row[3] for row in top_rows],
    })



@mcp.tool()
def get_content_based_recommendation(request: RecommendationRequest) -> RecommendationResponse:
    
    """
    _summary_
    name: "Get Content-Based Movie Recommendations"
    description: "This function takes the user's query and returns a ranked list of movie recommendations based on content filtering techniques."

    Args:
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's query for movie recommendations."
                }
            },
            "required": ["query"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "ranked_results": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "A list containing the ranked movie recommendations."
                }
            },
            "required": ["ranked_results"]
        }
    """
    query = request.query
    _log(f"content_mcp request query={query}")

    collection = client.get_or_create_collection(name='Fashion_Collection')
    cache_collection_name = 'Cache_new'
    cache_collection = client_cache.get_or_create_collection(name=cache_collection_name)

    threshold = 0.2

    ids = []
    documents = []
    distances = []
    metadatas = []
    results_df = pd.DataFrame()

    cache_results = _safe_chroma_query(cache_collection, query, n_results=2)
    if cache_results is None:
        cache_results = _empty_query_result()

    # Check if the distance is greater than the threshold, if so, return results from the main collection
    if cache_results['distances'][0] == [] or cache_results['distances'][0][0] > threshold:
        # Query the collection against the user query and return the results
        results = _safe_chroma_query(collection, query, n_results=5)

        if results is None:
            _log("Main collection query timed out. Using lexical fallback.")
            results_df = _fallback_lexical_rank(collection, query, n_results=5)
        else:

            # Store the query in cache_collection as a document with respect to ChromaDB for future reference
            # Store retrieved text, ids, distances, and metadatas in cache_collection as metadatas, so they can be fetched easily if a query indeed matches to a query in cache
            Keys = []
            Values = []

            for key, val in results.items():
                if val is None:
                    continue
                for i in range(len(val[0])):  # Iterate over the actual length of val
                    Keys.append(str(key) + str(i))
                    if len(val[0]) > i:  # Check if the current index exists in val
                        Values.append(str(val[0][i]))

            try:
                cache_collection.add(
                    documents=[query],
                    ids=[query],
                    metadatas=dict(zip(Keys, Values))
                )
            except Exception as exc:
                _log(f"Warning: cache write skipped: {exc}")

            # Print message indicating the results are found in the main collection
            _log("Not found in cache. Found in the main collection.")

            # Construct a DataFrame from the query results
            result_dict = {'Metadatas': results['metadatas'][0], 'Documents': results['documents'][0], 'Distances': results['distances'][0], "IDs": results["ids"][0]}
            results_df = pd.DataFrame.from_dict(result_dict)


    # If the distance is less than the threshold, return results from the cache
    elif cache_results['distances'][0][0] <= threshold and cache_results['ids'] and cache_results['ids'][0]:
        cache_result_dict = cache_results['metadatas'][0][0]

        # Loop through each inner list and then through the dictionary
        for key, value in cache_result_dict.items():
            if 'ids' in key:
                ids.append(value)
            elif 'documents' in key:
                documents.append(value)
            elif 'distances' in key:
                distances.append(value)
            elif 'metadatas' in key:
                metadatas.append(value)

        # Print message indicating the results are found in the cache
        _log("Found in cache!")

        # Create a DataFrame from the cached results
        results_df = pd.DataFrame({
            'IDs': ids,
            'Documents': documents,
            'Distances': distances,
            'Metadatas': metadatas
        })
    else:
        # Print message indicating no valid results found in cache
        _log("No valid results found in cache!")

    if results_df.empty:
        return RecommendationResponse(ranked_results=[])
    
    encoder = get_cross_encoder()

    if encoder is not None:
        cross_inputs = [[query, response] for response in results_df['Documents']]
        cross_rerank_scores = encoder.predict(cross_inputs)
        results_df['Reranked_scores'] = cross_rerank_scores
        rank = results_df.sort_values(by='Reranked_scores', ascending=False)
    else:
        fallback_scores = -pd.to_numeric(results_df['Distances'], errors='coerce').fillna(0.0)
        results_df['Reranked_scores'] = fallback_scores
        rank = results_df.sort_values(by='Reranked_scores', ascending=False)

    ranked_rows = [{str(k): v for k, v in row.items()} for row in rank.head(5).to_dict(orient='records')]
    movie_titles = _extract_titles(ranked_rows)
    _log(f"content_mcp response titles={movie_titles}")

    return RecommendationResponse(ranked_results=movie_titles)


if __name__ == "__main__":
    mcp.run(transport="stdio")