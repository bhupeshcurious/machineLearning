import os
import json
import math
import random
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import ClientError

# =====================================================================
# 0. GLOBAL EXPERIMENTAL CONFIGURATION
# =====================================================================
DATA_DIR = "C:/Users/lenovo/Desktop/github/machineLearning/data/ml-1m"           # Paths to extracted movies.dat, ratings.dat
NUM_NEGATIVE_CANDIDATES = 19     # 1 Positive Target + 19 Negatives = Pool of 20
EVAL_USER_SAMPLE = 10            # Number of users to test (adjust based on API limits)
TOP_K = 5                        # Evaluation cutoff rank
MODEL_NAME = "gemini-2.0-flash"   # Core reasoning engine

# FIX 1: The api_key was a hardcoded literal string instead of an env var lookup.
# os.environ.get("AQ.Ab8RN6...") would always return None (no env var is named
# like an API key). Read the correct env variable name instead.
client = genai.Client(api_key=os.environ.get("AIzaSyBaBIehiASi1LxYjTowro_1PyLnghfUIP0"))

# Define a strict, reproducible Pydantic output schema for the LLM ranking
class RecommenderOutput(BaseModel):
    ranked_titles: List[str]
    academic_reasoning: str


# =====================================================================
# 1. ACADEMIC DATA PARSING & PREPARATION
# =====================================================================
def load_and_clean_movielens() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses MovieLens 1M .dat files, handling double-colon separation
    and encoding formats typical of academic raw datasets.
    """
    print("[INFO] Parsing MovieLens raw files...")

    movies = pd.read_csv(
        os.path.join(DATA_DIR, 'movies.dat'),
        sep='::',
        engine='python',
        names=['MovieID', 'Title', 'Genres'],
        encoding='latin-1'
    )

    ratings = pd.read_csv(
        os.path.join(DATA_DIR, 'ratings.dat'),
        sep='::',
        engine='python',
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
        encoding='latin-1'
    )

    return movies, ratings


def generate_user_sequences(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Dict[int, Dict]:
    """
    Transforms interactions into user-specific timelines.
    Implements a strict temporal separation to prevent data leakage.
    """
    user_sequences = {}
    movie_id_to_title = dict(zip(movies_df['MovieID'], movies_df['Title']))
    all_movie_titles = list(movies_df['Title'].unique())

    # Group interactions by user
    grouped = ratings_df.groupby('UserID')

    for user_id, group in grouped:
        # Sort chronologically by interaction timestamp
        sorted_group = group.sort_values(by='Timestamp')

        # Filter for high-quality interactions (Positive Feedback Loop: Rating >= 4)
        positive_interactions = sorted_group[sorted_group['Rating'] >= 4]

        if len(positive_interactions) < 6:
            continue  # Exclude sparse profiles to maintain evaluation data quality

        titles_sequence = [movie_id_to_title[mid] for mid in positive_interactions['MovieID'] if mid in movie_id_to_title]

        # Temporal Split Protocol: Leave-One-Out Validation
        # Train Context = History sequence up to item N-1
        # Test Ground Truth = Item N (The hidden target)
        history_seq = titles_sequence[:-1]
        ground_truth_item = titles_sequence[-1]

        # Create negative candidate items (unseen by the user)
        seen_set = set(titles_sequence)
        negative_pool = [title for title in all_movie_titles if title not in seen_set]

        user_sequences[user_id] = {
            "history": history_seq,
            "ground_truth": ground_truth_item,
            "negative_pool": negative_pool
        }

    return user_sequences


# =====================================================================
# 2. EVALUATION METRICS MATHEMATICAL LOGIC
# =====================================================================
def evaluate_ranking_metrics(predictions: List[str], target: str, k: int) -> Dict[str, float]:
    """
    Computes mathematical ranking tracking metrics for validation.
    """
    truncated_preds = predictions[:k]

    hr = 0.0
    mrr = 0.0
    ndcg = 0.0

    if target in truncated_preds:
        hr = 1.0
        rank_idx = truncated_preds.index(target)  # 0-indexed position

        # MRR calculation
        mrr = 1.0 / (rank_idx + 1)

        # NDCG calculation via binary relevance discount formula
        ndcg = 1.0 / math.log2(rank_idx + 2)

    return {"HR@K": hr, "MRR@K": mrr, "NDCG@K": ndcg}


# =====================================================================
# 3. GENERATIVE INFERENCE ENGINE (GEMINI API)
# =====================================================================
def run_llm_reranker(history: List[str], candidates: List[str]) -> List[str]:
    """
    Executes context-aware zero-shot ranking using structured JSON output schemas.
    """
    history_str = "\n".join([f"- {item}" for item in history[-15:]])  # Windowing: last 15 items
    candidates_str = ", ".join(candidates)

    # FIX 3: Removed the duplicate "Evaluation" word in the section heading.
    prompt = f"""
    Context-Aware Recommender Evaluation System.

    User Historical Profile (Chronological viewing sequence of highly rated films):
    {history_str}

    Candidate Evaluation Pool (Contains 1 true hidden preferred movie and several random negatives):
    [{candidates_str}]

    Task Instructions:
    1. Analyze the hidden patterns, genres, and thematic tropes inside the User Historical Profile.
    2. Rank all items present in the Candidate Evaluation Pool from most likely to least likely to be watched next by the user.
    3. Return your final ranked preferences. You are STRICTLY forbidden from altering titles or introducing titles not explicitly present in the Candidate Evaluation Pool.
    """

    # Retry with exponential backoff to handle 429 rate limit errors
    max_retries = 6
    backoff = 10  # initial wait in seconds

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RecommenderOutput,
                    temperature=0.0,
                ),
            )
            data = json.loads(response.text)
            return data.get("ranked_titles", [])

        except ClientError as e:
            if e.status_code == 429:
                wait = backoff * (2 ** attempt)  # 10s, 20s, 40s, 80s ...
                print(f"[WARN] Rate limited (429). Waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
            else:
                print(f"[ERROR] API error: {e}")
                return []
        except Exception as e:
            print(f"[ERROR] Failed to parse output schema: {e}")
            return []

    print("[ERROR] Max retries exceeded. Skipping this user.")
    return []


# =====================================================================
# 4. EXECUTION MATRIX & BENCHMARK LOOP
# =====================================================================
def main():
    # FIX 2: load_and_clean_movielens() returns (movies, ratings) but the original
    # code unpacked them as (movies_df, ratings_df) and then passed them swapped
    # to generate_user_sequences(ratings_df, movies_df) — meaning movies were
    # treated as ratings and vice versa. Unpacking order now matches the return order.
    movies_df, ratings_df = load_and_clean_movielens()
    user_profiles = generate_user_sequences(ratings_df, movies_df)

    # FIX 4: Guard against fewer qualifying users than EVAL_USER_SAMPLE.
    available_users = list(user_profiles.keys())
    test_users = available_users[:min(EVAL_USER_SAMPLE, len(available_users))]
    all_user_metrics = []

    print(f"\n[INFO] Starting experimental benchmark over {len(test_users)} sampled user entities...")

    for i, uid in enumerate(test_users):
        profile = user_profiles[uid]
        history = profile["history"]
        target = profile["ground_truth"]
        negatives = profile["negative_pool"]

        # Sample negative instances and construct standard ranking pool
        sampled_negatives = random.sample(negatives, NUM_NEGATIVE_CANDIDATES)
        candidate_pool = sampled_negatives + [target]
        random.shuffle(candidate_pool)  # Strip positioning cues out of the pool

        print(f" Processing Evaluation Sample {i+1}/{len(test_users)} (User ID: {uid}) | History Dimensions: {len(history)}")

        # Perform Generative Ranking
        ranked_predictions = run_llm_reranker(history, candidate_pool)

        # Sanity Check: Handle structural API omission issues gracefully
        clean_predictions = [item for item in ranked_predictions if item in candidate_pool]
        missing = [item for item in candidate_pool if item not in clean_predictions]
        final_predictions = clean_predictions + missing  # Append edge-case missing items to the bottom

        # Polite delay between requests to stay under per-minute rate limits
        time.sleep(3)

        # Calculate Quantitative Metrics
        metrics = evaluate_ranking_metrics(final_predictions, target, k=TOP_K)
        all_user_metrics.append(metrics)

    # 5. Compile and Output Macro Performance Metrics
    summary_df = pd.DataFrame(all_user_metrics)
    print("\n" + "="*50)
    print(f"EXPERIMENTAL BENCHMARK RESULTS (K={TOP_K}, Candidates={NUM_NEGATIVE_CANDIDATES+1})")
    print("="*50)
    print(summary_df.mean().to_string())
    print("="*50)

if __name__ == "__main__":
    main()