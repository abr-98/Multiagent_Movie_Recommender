import pickle as pkl
import json
import pandas as pd
import numpy as np
import ast
from fastmcp import FastMCP
from warnings import filterwarnings
from pydantic import BaseModel, Field
import sys
import os
import logging


filterwarnings("ignore")


mcp = FastMCP("Collaborative_Filtering")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("collaborative_mcp")
if not logger.handlers:
  logger.setLevel(logging.INFO)
  _handler = logging.FileHandler(os.path.join(LOG_DIR, "collaborative_mcp.log"), encoding="utf-8")
  _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
  logger.addHandler(_handler)


def _log(message: str) -> None:
  logger.info(message)
  print(message, file=sys.stderr)


def _parse_movie_list(value):
  """Parse a movie list stored as python-literal string, JSON string, or native list."""
  if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
    return [str(x) for x in list(value) if pd.notna(x)]

  if value is None or (isinstance(value, float) and pd.isna(value)):
    return []

  text = str(value).strip()
  if not text or text.lower() == "nan":
    return []

  try:
    parsed = ast.literal_eval(text)
    if isinstance(parsed, (list, tuple, set, np.ndarray, pd.Series)):
      return [str(x) for x in list(parsed) if pd.notna(x)]
    return [str(parsed)]
  except Exception:
    # Fallback: treat as a plain single movie title.
    return [text]

with open("filtering/label_encoder.pkl", "rb") as f:
  label_encoder = pkl.load(f)

def normalize(name):
    try:
      return name.lower().strip()
    except AttributeError:
      return ""

def update_user_vector(user_vector, user_movies, movie_to_id, movieId_to_col):
    for movie in user_movies:
        if movie.lower() in movie_to_id:
            mid = movie_to_id[movie.lower()]
            col_idx = movieId_to_col[mid]
            user_vector[col_idx] = 1
        else:
          _log(f"Movie not found: {movie}")
    
    return user_vector

def get_user_inclinations(choices):
  
  df_cold_starter = pd.read_csv('filtering/Cold_starter.csv')

  with open("filtering/movie_mapper_orient.json", "r") as f:
    movie_mapper = json.load(f)
  
  _log(str(len(movie_mapper)))

  user_matrix = np.zeros(len(movie_mapper), dtype=np.int8)

  movie_to_id = {normalize(name): int(mid) for mid, name in list(movie_mapper.items())}

  movieId_to_col = {int(mid): i for i, mid in enumerate(movie_mapper.keys())}

  user_vector = update_user_vector(user_matrix, choices, movie_to_id, movieId_to_col)

  return user_vector


def get_user_genre_inclination(user_inp):

  genres = label_encoder.classes_

  user_genre_vector = np.zeros(len(genres), dtype=np.int8)

  for genre in user_inp:
    if genre in genres:
      idx = np.where(genres == genre)[0][0]
      user_genre_vector[idx] = 1

  return user_genre_vector


def get_recommendation_genre(user_genre_vector):

  pca_genre = pkl.load(open("filtering/pca.pkl", "rb"))

  user_genre_vector = pca_genre.transform(user_genre_vector.reshape(1, -1))

  users_list = pkl.load(open("filtering/user_list.pkl", "rb"))

  nn_algo = pkl.load(open("filtering/neighbor_finder.pkl", "rb"))

  similar_users = nn_algo.kneighbors(user_genre_vector, n_neighbors=3, return_distance=False)

  similar_users = similar_users.flatten()

  user_pref_movies = pd.read_csv("filtering/user_top_pref.csv")

  #user_pref_movies.set_index("userId", inplace=True)

  user_pref_movies = user_pref_movies.loc[similar_users]
  #print(user_pref_movies)
  user_pref_movies = user_pref_movies["movie_list"].tolist()
  #print(user_pref_movies)
  recom = []
  for idx, movie_list_str in enumerate(user_pref_movies):
    parsed_movies = _parse_movie_list(movie_list_str)
    if not parsed_movies:
      _log(f"No parseable genre-based movie list at position {idx}")
      continue
    recom.extend(parsed_movies)

  recom = set(recom)

  return recom

  
def get_recommendation_movies(user_vector):

  svd = pkl.load(open("filtering/SVD.pkl", "rb"))

  user_vector = svd.transform(user_vector.reshape(1, -1))

  nn_algo = pkl.load(open("filtering/neighbor_finder_movie.pkl", "rb"))

  with open("filtering/movie_mapper_orient.json") as f:
    movie_mapper = json.load(f)

  
  similar_users = nn_algo.kneighbors(user_vector, n_neighbors=10, return_distance=False)

  similar_users = similar_users.flatten()

  recom = []

  df_user_pref = pd.read_csv("filtering/user_top_pref.csv")
  df_user_pref.set_index("userId", inplace=True)

  for user in similar_users:
    if user not in df_user_pref.index:
      _log(f"User id {user} not found in user_top_pref index")
      continue

    user_pref_row = df_user_pref.loc[user]

    # Handle both Series and DataFrame cases safely.
    if isinstance(user_pref_row, pd.DataFrame):
      if "movie_list" in user_pref_row.columns and not user_pref_row.empty:
        movie_list_str = user_pref_row["movie_list"].iloc[0]
      elif not user_pref_row.empty:
        movie_list_str = user_pref_row.iloc[0, 0]
      else:
        continue
    else:
      if "movie_list" in user_pref_row.index:
        movie_list_str = user_pref_row["movie_list"]
      elif len(user_pref_row.index) > 0:
        movie_list_str = user_pref_row.iloc[0]
      else:
        continue

    parsed_movies = _parse_movie_list(movie_list_str)
    if not parsed_movies:
      _log(f"No parseable movie list for user {user}")
      continue
    recom.extend(parsed_movies)

  recom = set(recom)

  return recom

class UserInput(BaseModel):
    movies: list[str] = Field(..., description="List of movies the user has watched")
    genres: list[str] = Field(..., description="List of genres the user prefers")

class RecommendationOutput(BaseModel):
  recommendations: list[str] = Field(..., description="List of recommended movies")


@mcp.tool()
def get_collaborative_based_recommendation(user_input: UserInput) -> RecommendationOutput:
  
  """
  _summary_
  name: "Get Final Recommendation from Collaborative Filtering"
  description: "This function takes the user's movie and genre preferences and returns a set of recommended movies based on collaborative filtering techniques."

  Args:
      "input_schema": {
          "type": "object",
          "properties": {
              "movies": {
                  "type": "array",
                  "items": {"type": "string"},
                  "description": "List of movies the user has watched"
              },
              "genres": {
                  "type": "array",
                  "items": {"type": "string"},
                  "description": "List of genres the user prefers"
              }
          },
          "required": ["movies", "genres"]
      },
      "output_schema": {
          "type": "set",
          "properties": {
              "recommendations": {
                  "type": "set",
                  "items": {"type": "string"},
                  "description": "Set of recommended movies"
              }
          },
          "required": ["recommendations"]
      }"""

  
  _log(f"collaborative_mcp request movies={user_input.movies} genres={user_input.genres}")
  user_vector = get_user_inclinations(user_input.movies)
  user_genre_vector = get_user_genre_inclination(user_input.genres)

  recom_movies = get_recommendation_movies(user_vector)
  recom_genres = get_recommendation_genre(user_genre_vector)

  final_recom = recom_movies.union(recom_genres)
  titles = sorted([str(m) for m in final_recom])
  _log(f"collaborative_mcp response titles={titles}")

  return RecommendationOutput(recommendations=titles)


if __name__ == "__main__":
  mcp.run(transport="stdio")