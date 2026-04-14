import chromadb
import os
import pandas as pd 
from typing import Any
import re
from sentence_transformers import CrossEncoder, util
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import requests
import sys
import logging

chroma_db_path = 'plotDB_2'

client = chromadb.PersistentClient(path=chroma_db_path)

persist_dir_cache = "plot_cache"

os.makedirs(persist_dir_cache, exist_ok=True)

client_cache = chromadb.PersistentClient(path=persist_dir_cache)

mcp = FastMCP("Plot_Based_Filtering")

cross_encoder = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("plot_mcp")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(os.path.join(LOG_DIR, "plot_mcp.log"), encoding="utf-8")
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
      cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
      return cross_encoder
  except Exception as exc:
      _log(f"Warning: CrossEncoder unavailable ({exc}). Falling back to distance-based ranking.")
      return None


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

def get_recommendation(query):

    collection = client.get_or_create_collection(name='Movie_plot')
    cache_collection_name = 'Cache_2'
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
    _log(str(results_df))

    if results_df.empty:
        return []

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

    return [{str(k): v for k, v in row.items()} for row in rank.head(5).to_dict(orient='records')]

#### filter creator

def clean_text(desc):
  modified_text = desc.replace("<ul>", "").replace("</ul>", "").replace("<li>", "").replace("</li>", ",").replace("<br>", ".")
  final = re.sub(r'<[^>]*>', '', modified_text).replace(". .",".")
  return final

def truncate_text(text, max_words=200):
    return " ".join(text.split()[:max_words])


HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def get_movie_plot(movie_name):
    base_url = "https://en.wikipedia.org/w/api.php"

    # Step 1: Search correct page
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": movie_name + " film",
        "format": "json"
    }

    res = requests.get(base_url, params=search_params, headers=HEADERS)
    data = res.json()

    results = data.get("query", {}).get("search", [])
    if not results:
        return "Movie not found"

    page_title = results[0]["title"]

    # Step 2: Get sections
    section_params = {
        "action": "parse",
        "page": page_title,
        "prop": "sections",
        "format": "json"
    }

    res = requests.get(base_url, params=section_params, headers=HEADERS)
    data = res.json()

    sections = data.get("parse", {}).get("sections", [])

    # Step 3: Find plot section index
    plot_index = None
    for sec in sections:
        title = sec["line"].lower()
        if any(k in title for k in ["plot", "synopsis", "premise"]):
            plot_index = sec["index"]
            break

    if not plot_index:
        return "Plot section not found"

    # Step 4: Fetch that section content
    content_params = {
        "action": "parse",
        "page": page_title,
        "prop": "text",
        "section": plot_index,
        "format": "json"
    }

    res = requests.get(base_url, params=content_params, headers=HEADERS)
    data = res.json()

    html = data["parse"]["text"]["*"]

    # Step 5: Clean HTML → text
    soup = BeautifulSoup(html, "html.parser")

    paragraphs = [p.get_text() for p in soup.find_all("p")]

    return "\n".join(paragraphs).strip()

def create_query(movie_name):

  movie_plot = get_movie_plot(movie_name)
  short_plot = truncate_text(clean_text(movie_plot))

  query = f"""

  Find movie plots similar to:

  {short_plot}

  """

  return query

class RecommendationRequest(BaseModel):
    movie_name: str = Field(..., description="The user's query for movie recommendations.")

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

@mcp.tool()
def get_plot_based_recommendation(movie_name_request: RecommendationRequest) -> RecommendationResponse:
  
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
                  "items": {"type": "string"},
                  "description": "A DataFrame containing the ranked movie recommendations."
              }
          },
          "required": ["ranked_results"]
      }
    """
   movie_name = movie_name_request.movie_name
   _log(f"plot_mcp request movie_name={movie_name}")
   query = create_query(movie_name)
   ranked_rows = get_recommendation(query)
   movie_titles = _extract_titles(ranked_rows)
   _log(f"plot_mcp response titles={movie_titles}")
   return RecommendationResponse(ranked_results=movie_titles)


if __name__ == "__main__":
  #print(get_plot_based_recommendation(RecommendationRequest(movie_name="Inception")).ranked_results)
  mcp.run(transport="stdio")