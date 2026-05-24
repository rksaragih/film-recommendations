import pandas as pd
import pickle
import numpy as np

# =========================================
# LOAD DATASET & MODEL
# =========================================

DATASET_PATH = "../Dataset/"
MODEL_PATH = "../models/"

movies = pd.read_csv(DATASET_PATH + "clean_movies.csv")

similarity = pickle.load(open(MODEL_PATH + "similarity.pkl", "rb"))
cf_avg_normalized = pickle.load(open(MODEL_PATH + "cf_predictions.pkl", "rb"))
movie_id_to_idx = pickle.load(open(MODEL_PATH + "movie_id_to_idx.pkl", "rb"))

# =========================================
# RECOMMENDATION FUNCTION
# =========================================

def recommend(movie_title, top_n=5, cbf_weight=0.5, cf_weight=0.5):
    """
    Hybrid recommendation menggunakan weighted average:
    final_score = (cbf_weight * cbf_score) + (cf_weight * cf_score)

    cbf_weight : bobot Content-Based Filtering (kemiripan konten)
    cf_weight  : bobot Collaborative Filtering (popularitas dari rating)
    total bobot harus = 1.0
    """

    # Cek apakah film tersedia
    if movie_title not in movies['title'].values:
        print(f"Film '{movie_title}' tidak ditemukan!")
        return []

    # Ambil index film di CBF matrix
    movie_idx = movies[movies['title'] == movie_title].index[0]
    movie_id = movies.iloc[movie_idx]['id']

    # ---- CBF Score ----
    cbf_scores = list(enumerate(similarity[movie_idx]))
    cbf_scores = [(i, score) for i, score in cbf_scores if i != movie_idx]

    # ---- CF Score ----
    # Ambil normalized CF score untuk tiap film
    # Kalau film tidak ada di cf_predictions (tidak punya rating), score = 0
    recommendations = []

    for idx, cbf_score in cbf_scores:
        candidate_id = movies.iloc[idx]['id']
        cf_score = cf_avg_normalized.get(candidate_id, 0.0)
        final_score = (cbf_weight * cbf_score) + (cf_weight * cf_score)

        recommendations.append({
            'title': movies.iloc[idx]['title'],
            'cbf_score': round(cbf_score, 4),
            'cf_score': round(float(cf_score), 4),
            'final_score': round(final_score, 4)
        })

    # Urutkan berdasarkan final_score tertinggi
    recommendations = sorted(recommendations, key=lambda x: x['final_score'], reverse=True)

    return recommendations[:top_n]

# =========================================
# EVALUATION FUNCTION
# =========================================

def evaluate_similarity(movie_title, top_n=5, cbf_weight=0.5, cf_weight=0.5):

    recommendations = recommend(movie_title, top_n, cbf_weight, cf_weight)

    if not recommendations:
        return

    print("=" * 55)
    print(f"REKOMENDASI FILM UNTUK: {movie_title}")
    print(f"(CBF weight: {cbf_weight} | CF weight: {cf_weight})")
    print("=" * 55)

    final_scores = []
    cbf_scores = []
    cf_scores = []

    for idx, movie in enumerate(recommendations, start=1):
        print(f"{idx}. {movie['title']}")
        print(f"   CBF Score   : {movie['cbf_score']}")
        print(f"   CF Score    : {movie['cf_score']}")
        print(f"   Final Score : {movie['final_score']}")

        final_scores.append(movie['final_score'])
        cbf_scores.append(movie['cbf_score'])
        cf_scores.append(movie['cf_score'])

    print("\n" + "=" * 55)
    print("HASIL EVALUASI MODEL")
    print("=" * 55)
    print(f"{'':25} {'CBF':>8} {'CF':>8} {'Final':>8}")
    print(f"{'Average Score':25} {np.mean(cbf_scores):>8.4f} {np.mean(cf_scores):>8.4f} {np.mean(final_scores):>8.4f}")
    print(f"{'Highest Score':25} {np.max(cbf_scores):>8.4f} {np.max(cf_scores):>8.4f} {np.max(final_scores):>8.4f}")
    print(f"{'Lowest Score':25} {np.min(cbf_scores):>8.4f} {np.min(cf_scores):>8.4f} {np.min(final_scores):>8.4f}")

# =========================================
# MAIN PROGRAM
# =========================================

if __name__ == "__main__":

    movie_name = "Toy Story"

    evaluate_similarity(movie_name, top_n=5, cbf_weight=0.5, cf_weight=0.5)