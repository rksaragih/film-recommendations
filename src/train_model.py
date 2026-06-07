import pandas as pd
import pickle
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

# =========================================
# LOAD DATASET
# =========================================

DATASET_PATH = "../Dataset/"
MODEL_PATH = "../models/"

movies = pd.read_csv(DATASET_PATH + "clean_movies.csv")
ratings = pd.read_csv(DATASET_PATH + "clean_ratings.csv")

movies = movies[['id', 'title', 'genres']]

# =========================================
# COLLABORATIVE FILTERING (SVD)
# =========================================
# SVD (Singular Value Decomposition) adalah Matrix Factorization —
# memecah user-item matrix menjadi faktor laten yang merepresentasikan
# preferensi user dan karakteristik film secara tersembunyi.

# Bangun mapping userId dan movieId ke index matrix
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()
user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

rows = ratings['userId'].map(user_id_to_idx).to_numpy()
cols = ratings['movieId'].map(movie_id_to_idx).to_numpy()
data = ratings['rating'].to_numpy()

n_users = len(user_ids)
n_movies = len(movie_ids)
print(f"Building sparse user-item matrix: {n_users} users x {n_movies} movies")

# Gunakan sparse matrix agar efisien di memori
sparse_ratings = csr_matrix((data, (rows, cols)), shape=(n_users, n_movies))

# ── Normalisasi: mean-centering per user ──
# Kurangi rata-rata rating tiap user untuk menghilangkan user bias
# (user yang royal beri bintang 5 vs user yang pelit beri bintang rendah)
user_rating_counts = np.asarray(sparse_ratings.getnnz(axis=1)).ravel()
user_rating_sums = np.asarray(sparse_ratings.sum(axis=1)).ravel()
user_ratings_mean = np.zeros(n_users, dtype=np.float64)
nonzero_users = user_rating_counts > 0
user_ratings_mean[nonzero_users] = (
    user_rating_sums[nonzero_users] / user_rating_counts[nonzero_users]
)

demeaned_values = data - user_ratings_mean[rows]
sparse_demeaned = csr_matrix((demeaned_values, (rows, cols)), shape=(n_users, n_movies))

# ── SVD Decomposition ──
# Pecah matrix menjadi U × diag(sigma) × Vt dengan k=50 latent factors
# U     → profil laten setiap user     (n_users × k)
# sigma → bobot setiap dimensi laten   (k,)
# Vt    → profil laten setiap film     (k × n_movies)
k = min(50, min(n_users, n_movies) - 1)
U, sigma, Vt = svds(sparse_demeaned, k=k)

print(f"SVD selesai — U: {U.shape}, sigma: {sigma.shape}, Vt: {Vt.shape}")

# ── CF score rata-rata (fallback untuk user tanpa riwayat rating) ──
# Hitung prediksi dari "user rata-rata" untuk semua film,
# dipakai sebagai fallback jika user tidak dikenal sistem
sigma_diag = np.diag(sigma)
avg_pred_latent = np.dot(np.dot(np.mean(U, axis=0), sigma_diag), Vt)
cf_avg_scores = pd.Series(
    avg_pred_latent + user_ratings_mean.mean(),
    index=movie_ids
)
cf_avg_normalized = (
    (cf_avg_scores - cf_avg_scores.min()) /
    (cf_avg_scores.max() - cf_avg_scores.min())
)

print(f"CF avg score length: {len(cf_avg_normalized)}")

# =========================================
# SIMPAN MODEL
# =========================================

# DataFrame film (untuk lookup judul & genre)
pickle.dump(movies, open(MODEL_PATH + "movies.pkl", "wb"))

# CF score rata-rata (fallback)
pickle.dump(cf_avg_normalized, open(MODEL_PATH + "cf_predictions.pkl", "wb"))

# Komponen SVD — dipakai untuk prediksi personal per user di app.py
pickle.dump(U, open(MODEL_PATH + "svd_U.pkl", "wb"))
pickle.dump(sigma, open(MODEL_PATH + "svd_sigma.pkl", "wb"))
pickle.dump(Vt, open(MODEL_PATH + "svd_Vt.pkl", "wb"))
pickle.dump(user_ratings_mean, open(MODEL_PATH + "user_ratings_mean.pkl", "wb"))
pickle.dump(user_id_to_idx, open(MODEL_PATH + "user_id_to_idx.pkl", "wb"))
pickle.dump(movie_id_to_idx, open(MODEL_PATH + "movie_id_to_idx_cf.pkl", "wb"))
pickle.dump(movie_ids, open(MODEL_PATH + "cf_movie_ids.pkl", "wb"))

print("\nModel berhasil disimpan!")
print(f"  - movies.pkl")
print(f"  - cf_predictions.pkl (CF avg, fallback)")
print(f"  - svd_U.pkl (shape: {U.shape})")
print(f"  - svd_sigma.pkl (shape: {sigma.shape})")
print(f"  - svd_Vt.pkl (shape: {Vt.shape})")
print(f"  - user_ratings_mean.pkl (shape: {user_ratings_mean.shape})")
print(f"  - user_id_to_idx.pkl ({n_users} users)")
print(f"  - movie_id_to_idx_cf.pkl ({n_movies} movies)")
print(f"  - cf_movie_ids.pkl ({len(movie_ids)} movie IDs)")