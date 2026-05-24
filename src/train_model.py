import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

# =========================================
# LOAD DATASET
# =========================================

DATASET_PATH = "../Dataset/"
MODEL_PATH = "../models/"

movies = pd.read_csv(DATASET_PATH + "clean_movies.csv")
ratings = pd.read_csv(DATASET_PATH + "clean_ratings.csv")

movies = movies[['id', 'title', 'tags']]

# =========================================
# CONTENT-BASED FILTERING (TF-IDF)
# =========================================
# Sama seperti sebelumnya — TF-IDF + cosine similarity antar film

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    sublinear_tf=True
)

tfidf_matrix = tfidf.fit_transform(movies['tags'])

# linear_kernel efisien untuk TF-IDF yang sudah ternormalisasi
cbf_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

print(f"CBF matrix shape: {cbf_similarity.shape}")

# =========================================
# COLLABORATIVE FILTERING (SVD)
# =========================================
# SVD (Singular Value Decomposition) adalah Matrix Factorization —
# memecah user-item matrix menjadi faktor laten yang merepresentasikan
# preferensi user dan karakteristik film secara tersembunyi.

# Build sparse user-item matrix to avoid constructing a huge dense matrix
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

sparse_ratings = csr_matrix((data, (rows, cols)), shape=(n_users, n_movies))

# Normalisasi: kurangi rata-rata rating tiap user untuk menghilangkan bias user
user_rating_counts = np.asarray(sparse_ratings.getnnz(axis=1)).ravel()
user_rating_sums = np.asarray(sparse_ratings.sum(axis=1)).ravel()
user_ratings_mean = np.zeros(n_users, dtype=np.float64)
nonzero_users = user_rating_counts > 0
user_ratings_mean[nonzero_users] = (
    user_rating_sums[nonzero_users] / user_rating_counts[nonzero_users]
)

# Demean hanya nilai rating yang ada
demeaned_values = data - user_ratings_mean[rows]
sparse_demeaned = csr_matrix((demeaned_values, (rows, cols)), shape=(n_users, n_movies))

# Decompose matrix dengan SVD
k = min(50, min(n_users, n_movies) - 1)
U, sigma, Vt = svds(sparse_demeaned, k=k)

# Hitung rata-rata prediksi CF per film tanpa membuat matrix lengkap
sigma_diag = np.diag(sigma)
avg_pred_latent = np.dot(np.dot(np.mean(U, axis=0), sigma_diag), Vt)
cf_avg_scores = pd.Series(
    avg_pred_latent + user_ratings_mean.mean(),
    index=movie_ids
)
cf_avg_normalized = (cf_avg_scores - cf_avg_scores.min()) / (cf_avg_scores.max() - cf_avg_scores.min())

print(f"CF normalized score length: {len(cf_avg_normalized)}")

# =========================================
# SIMPAN MODEL
# =========================================

# Simpan movies DataFrame
pickle.dump(movies, open(MODEL_PATH + "movies.pkl", "wb"))

# Simpan CBF similarity matrix
pickle.dump(cbf_similarity, open(MODEL_PATH + "similarity.pkl", "wb"))

# Simpan CF normalized scores per movie
pickle.dump(cf_avg_normalized, open(MODEL_PATH + "cf_predictions.pkl", "wb"))

# Simpan mapping id -> index untuk lookup cepat
# Dipakai saat hybrid: cari index film berdasarkan tmdbId
movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies['id'].values)}
pickle.dump(movie_id_to_idx, open(MODEL_PATH + "movie_id_to_idx.pkl", "wb"))

# Simpan TF-IDF vectorizer dan matrix untuk keyword search
pickle.dump(tfidf, open(MODEL_PATH + "tfidf_vectorizer.pkl", "wb"))
pickle.dump(tfidf_matrix, open(MODEL_PATH + "tfidf_matrix.pkl", "wb"))

print("\nModel berhasil disimpan!")
print(f"  - movies.pkl")
print(f"  - similarity.pkl            (CBF, shape: {cbf_similarity.shape})")
print(f"  - cf_predictions.pkl        (CF, shape: {cf_avg_normalized.shape})")
print(f"  - movie_id_to_idx.pkl")
print(f"  - tfidf_vectorizer.pkl      (untuk keyword search)")
print(f"  - tfidf_matrix.pkl          (untuk keyword search)")