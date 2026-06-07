import ast
import pickle
import numpy as np
import pandas as pd

# =========================================
# LOAD DATASET & MODEL
# =========================================

DATASET_PATH = "../Dataset/"
MODEL_PATH   = "../models/"

movies = pd.read_csv(DATASET_PATH + "clean_movies.csv")
ratings = pd.read_csv(DATASET_PATH + "clean_ratings.csv")

# Parse kolom genres (tersimpan sebagai string list)
movies['genres'] = movies['genres'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)

# Load komponen SVD
U = pickle.load(open(MODEL_PATH + "svd_U.pkl", "rb"))
sigma = pickle.load(open(MODEL_PATH + "svd_sigma.pkl", "rb"))
Vt = pickle.load(open(MODEL_PATH + "svd_Vt.pkl", "rb"))
user_ratings_mean = pickle.load(open(MODEL_PATH + "user_ratings_mean.pkl", "rb"))
user_id_to_idx = pickle.load(open(MODEL_PATH + "user_id_to_idx.pkl", "rb"))
cf_movie_ids = pickle.load(open(MODEL_PATH + "cf_movie_ids.pkl", "rb"))

# Load CF avg (fallback cold start)
cf_avg_norm = pickle.load(open(MODEL_PATH + "cf_predictions.pkl", "rb"))

movie_avg_rating = (
    ratings.groupby('movieId')['rating']
    .mean()
    .rename('avg_rating')
)
movie_rating_counts = ratings.groupby('movieId')['rating'].count()

# =========================================
# HELPER: PREDIKSI RATING SEMUA FILM UNTUK SATU USER
# =========================================

def get_personal_cf_scores(user_id):
    """
    Hitung prediksi rating user untuk semua film via SVD.
    Return: Series {movie_id: predicted_rating} dalam skala ASLI (0-5),
            SEBELUM dinormalisasi — agar bisa dibandingkan dengan actual rating.
    """
    user_idx = user_id_to_idx[user_id]
    sigma_diag = np.diag(sigma)

    user_pred = np.dot(np.dot(U[user_idx], sigma_diag), Vt)
    user_pred += user_ratings_mean[user_idx]

    return pd.Series(user_pred, index=cf_movie_ids)


# =========================================
# EVALUASI SKENARIO 1: USER LAMA
# =========================================

def evaluate_personal(user_id, top_n=10):
    """
    Evaluasi rekomendasi untuk user yang sudah punya riwayat rating.

    Bagian A — Deskriptif:
        Tampilkan Top-N film yang direkomendasikan (belum pernah ditonton).

    Bagian B — Metrik (MAE & RMSE):
        Bandingkan predicted rating vs actual rating pada film
        yang SUDAH pernah ditonton, sebagai sanity check model SVD.
    """

    print("=" * 60)
    print(f"EVALUASI USER ID: {user_id}")
    print("=" * 60)

    # Cek apakah user dikenal
    if user_id not in user_id_to_idx:
        print(f"User {user_id} tidak ditemukan di data training.")
        return

    user_ratings = ratings[ratings['userId'] == user_id]
    if user_ratings.empty:
        print(f"User {user_id} tidak memiliki riwayat rating.")
        return

    # Ambil prediksi rating (skala asli, belum dinormalisasi)
    cf_scores_raw = get_personal_cf_scores(user_id)

    # Normalisasi 0-1 untuk keperluan ranking rekomendasi
    mn, mx = cf_scores_raw.min(), cf_scores_raw.max()
    cf_scores_norm = (cf_scores_raw - mn) / (mx - mn) if mx > mn else cf_scores_raw * 0

    watched_ids = set(user_ratings['movieId'].values)

    # ── BAGIAN A: DESKRIPTIF — Top-N Rekomendasi ──
    print(f"\n{'─'*60}")
    print(f"  A. TOP-{top_n} REKOMENDASI PERSONAL")
    print(f"     (film yang belum pernah ditonton, skor tertinggi)")
    print(f"{'─'*60}")

    recommendations = []
    for _, row in movies.iterrows():
        movie_id = row['id']
        if movie_id in watched_ids:
            continue
        score = float(cf_scores_norm.get(movie_id, 0.0))
        recommendations.append({
            'title':  row['title'],
            'genres': ', '.join(row['genres'][:3]) if row['genres'] else '-',
            'score':  score,
        })

    recommendations.sort(key=lambda x: x['score'], reverse=True)
    top_recs = recommendations[:top_n]

    print(f"  {'#':<4} {'Judul':<40} {'Genre':<25} {'Score':>6}")
    print(f"  {'─'*4} {'─'*40} {'─'*25} {'─'*6}")
    for i, rec in enumerate(top_recs, 1):
        print(f"  {i:<4} {rec['title'][:39]:<40} {rec['genres'][:24]:<25} {rec['score']:>6.4f}")

    # ── BAGIAN B: METRIK — MAE & RMSE ──
    print(f"\n{'─'*60}")
    print(f"  B. AKURASI PREDIKSI SVD (Sanity Check)")
    print(f"     (film yang sudah ditonton — predicted vs actual)")
    print(f"{'─'*60}")

    actuals = []
    predicteds = []
    details = []

    for _, row in user_ratings.iterrows():
        movie_id = row['movieId']
        actual = row['rating']
        predicted = float(cf_scores_raw.get(movie_id, np.nan))

        if np.isnan(predicted):
            continue

        movie_title = movies[movies['id'] == movie_id]['title'].values
        title = movie_title[0] if len(movie_title) > 0 else f"ID:{movie_id}"

        actuals.append(actual)
        predicteds.append(predicted)
        details.append({
            'title': title,
            'actual': actual,
            'predicted': predicted,
            'error': abs(actual - predicted),
        })

    if not actuals:
        print("  Tidak ada data untuk dihitung.")
        return

    # Hitung MAE dan RMSE
    errors = np.array([d['error'] for d in details])
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    # Tampilkan detail per film
    details.sort(key=lambda x: x['error'])
    print(f"\n  {'Judul':<38} {'Actual':>7} {'Prediksi':>9} {'Selisih':>8}")
    print(f"  {'─'*38} {'─'*7} {'─'*9} {'─'*8}")
    for d in details:
        print(f"  {d['title'][:37]:<38} {d['actual']:>7.1f} {d['predicted']:>9.2f} {d['error']:>8.2f}")

    # Tampilkan ringkasan metrik
    print(f"\n  {'─'*60}")
    print(f"  {'Total film dievaluasi':<35} {len(actuals):>6} film")
    print(f"  {'MAE  (rata-rata selisih)':<35} {mae:>6.4f}")
    print(f"  {'RMSE (penalti error besar)':<35} {rmse:>6.4f}")
    print(f"  {'─'*60}")
    print(f"  Interpretasi MAE:")
    if mae < 0.5:
        print(f"  ✓ Sangat baik — prediksi meleset < 0.5 bintang")
    elif mae < 1.0:
        print(f"  ~ Cukup baik — prediksi meleset < 1 bintang")
    else:
        print(f"  ✗ Perlu perhatian — prediksi meleset > 1 bintang")
    print()


# =========================================
# EVALUASI SKENARIO 2: USER BARU (COLD START)
# =========================================

def evaluate_cold_start(selected_genres, top_n=10):
    """
    Evaluasi rekomendasi cold start berdasarkan genre pilihan.

    Bagian A — Deskriptif:
        Tampilkan Top-N film dengan avg rating tertinggi sesuai genre.

    Bagian B — Metrik:
        Distribusi avg rating dari hasil rekomendasi
        (min, max, mean) sebagai indikator kualitas kandidat.
    """

    print("=" * 60)
    print(f"EVALUASI COLD START")
    print(f"Genre dipilih: {', '.join(selected_genres)}")
    print("=" * 60)

    selected_set = set(selected_genres)

    candidates = []
    for _, row in movies.iterrows():
        movie_genres = set(row['genres'])
        if not movie_genres.intersection(selected_set):
            continue

        avg_rating = float(movie_avg_rating.get(row['id'], 0.0))
        count = int(movie_rating_counts.get(row['id'], 0))
        if avg_rating == 0.0 or count < 50:
            continue

        candidates.append({
            'title': row['title'],
            'genres': ', '.join(row['genres'][:3]) if row['genres'] else '-',
            'avg_rating': avg_rating,
        })

    candidates.sort(key=lambda x: x['avg_rating'], reverse=True)
    top_recs = candidates[:top_n]

    # ── BAGIAN A: DESKRIPTIF ──
    print(f"\n{'─'*60}")
    print(f"  A. TOP-{top_n} REKOMENDASI COLD START")
    print(f"{'─'*60}")
    print(f"  {'#':<4} {'Judul':<40} {'Genre':<20} {'Avg ★':>6}")
    print(f"  {'─'*4} {'─'*40} {'─'*20} {'─'*6}")
    for i, rec in enumerate(top_recs, 1):
        print(f"  {i:<4} {rec['title'][:39]:<40} {rec['genres'][:19]:<20} {rec['avg_rating']:>6.2f}")

    # ── BAGIAN B: METRIK DISTRIBUSI ──
    print(f"\n{'─'*60}")
    print(f"  B. DISTRIBUSI RATING HASIL REKOMENDASI")
    print(f"{'─'*60}")

    all_ratings  = [r['avg_rating'] for r in top_recs]
    print(f"  {'Total kandidat film':<35} {len(candidates):>6} film")
    print(f"  {'Film ditampilkan (top-N)':<35} {len(top_recs):>6} film")
    print(f"  {'Avg rating tertinggi':<35} {max(all_ratings):>6.2f}")
    print(f"  {'Avg rating terendah':<35} {min(all_ratings):>6.2f}")
    print(f"  {'Rata-rata avg rating':<35} {np.mean(all_ratings):>6.2f}")
    print(f"  {'─'*60}")
    print()


# =========================================
# MAIN
# =========================================

if __name__ == "__main__":

    # ── Uji Skenario 1: user lama ──
    evaluate_personal(user_id=1, top_n=10)

    print("\n" + "=" * 60 + "\n")

    # ── Uji Skenario 2: user baru ──
    evaluate_cold_start(selected_genres=["Action", "Comedy"], top_n=10)