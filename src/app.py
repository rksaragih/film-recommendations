import ast
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for

# =========================================
# LOAD MODEL & DATA
# =========================================

DATASET_PATH = "../Dataset/"
MODEL_PATH = "../models/"

movies = pd.read_csv(DATASET_PATH + "clean_movies.csv")
ratings = pd.read_csv(DATASET_PATH + "clean_ratings.csv")

# Kolom genres tersimpan sebagai string "['Action', 'Drama']" → parse balik ke list
movies['genres'] = movies['genres'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)

# Komponen SVD untuk rekomendasi personal
U = pickle.load(open(MODEL_PATH + "svd_U.pkl", "rb"))
sigma = pickle.load(open(MODEL_PATH + "svd_sigma.pkl", "rb"))
Vt = pickle.load(open(MODEL_PATH + "svd_Vt.pkl", "rb"))
user_ratings_mean = pickle.load(open(MODEL_PATH + "user_ratings_mean.pkl", "rb"))
user_id_to_idx = pickle.load(open(MODEL_PATH + "user_id_to_idx.pkl", "rb"))
cf_movie_ids = pickle.load(open(MODEL_PATH + "cf_movie_ids.pkl", "rb"))

app = Flask(__name__)
app.secret_key = 'secret_key'

# =========================================
# PRECOMPUTE — dihitung sekali saat startup
# =========================================

# Daftar genre unik untuk halaman pick_genres
ALL_GENRES = sorted({
    genre
    for genres in movies['genres']
    for genre in genres
    if genre
})

# Rata-rata rating per film (untuk cold start)
movie_avg_rating = ratings.groupby('movieId')['rating'].mean()

# Jumlah penilai per film (untuk filter minimum 50 penilai)
movie_rating_counts = ratings.groupby('movieId')['rating'].count()

# =========================================
# HELPER: PREDIKSI RATING PERSONAL (SVD)
# =========================================

def get_personal_cf_scores(user_id):
    """
    Hitung prediksi rating user untuk semua film via SVD.

    Formula:
        predicted = U[user_idx] × diag(sigma) × Vt + user_ratings_mean[user_idx]

    Return: Series {movie_id: predicted_score} ternormalisasi 0-1.
    """
    user_idx = user_id_to_idx[user_id]
    sigma_diag = np.diag(sigma)

    user_pred = np.dot(np.dot(U[user_idx], sigma_diag), Vt)
    user_pred += user_ratings_mean[user_idx]

    cf_scores = pd.Series(user_pred, index=cf_movie_ids)

    mn, mx = cf_scores.min(), cf_scores.max()
    if mx > mn:
        cf_scores = (cf_scores - mn) / (mx - mn)
    else:
        cf_scores[:] = 0.0

    return cf_scores

# =========================================
# REKOMENDASI: USER LAMA
# =========================================

def recommend_for_user(user_id, top_n=10):
    """
    Rekomendasi personal berbasis SVD.

    Langkah:
    1. Hitung prediksi rating user untuk semua film
    2. Buang film yang sudah pernah ditonton (sudah diberi rating)
    3. Kembalikan Top-N film dengan prediksi tertinggi
    """
    watched_ids = set(ratings[ratings['userId'] == user_id]['movieId'].values)
    cf_scores = get_personal_cf_scores(user_id)

    recommendations = []
    for _, row in movies.iterrows():
        movie_id = row['id']
        if movie_id in watched_ids:
            continue

        score = float(cf_scores.get(movie_id, 0.0))
        recommendations.append({
            'title': row['title'],
            'genres': row['genres'],
            'score': round(score, 4),
        })

    recommendations.sort(key=lambda x: x['score'], reverse=True)
    return recommendations[:top_n]

# =========================================
# REKOMENDASI: USER BARU (cold start)
# =========================================

def recommend_by_genres(selected_genres, top_n=10):
    """
    Rekomendasi untuk user baru yang belum punya riwayat rating.

    Langkah:
    1. Filter film yang memiliki setidaknya satu genre yang dipilih
    2. Filter film dengan minimal 50 penilai (hindari film obscure)
    3. Urutkan berdasarkan rata-rata rating tertinggi
    4. Kembalikan Top-N
    """
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
            'genres': row['genres'],
            'avg_rating': round(avg_rating, 2),
            'score': round(avg_rating, 4),
        })

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:top_n]

# =========================================
# HELPER: RIWAYAT RATING USER
# =========================================

def get_user_history(user_id, top_n=5):
    """Film yang pernah diberi rating user, diurutkan dari rating tertinggi."""
    user_ratings = ratings[ratings['userId'] == user_id].copy()
    if user_ratings.empty:
        return []

    user_ratings = user_ratings.sort_values('rating', ascending=False).head(top_n)
    history = []
    for _, row in user_ratings.iterrows():
        movie_row = movies[movies['id'] == row['movieId']]
        if not movie_row.empty:
            history.append({
                'title': movie_row.iloc[0]['title'],
                'rating': row['rating'],
            })
    return history

# =========================================
# ROUTES
# =========================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Login sederhana: user memasukkan userId numerik.
    Setelah login:
      - Ada riwayat rating → halaman utama (rekomendasi personal SVD)
      - Tidak ada riwayat → halaman pilih genre (cold start)
    """
    error = None
    if request.method == 'POST':
        try:
            user_id = int(request.form.get('user_id', '').strip())
            session['user_id'] = user_id

            has_history = not ratings[ratings['userId'] == user_id].empty
            if not has_history:
                return redirect(url_for('pick_genres'))

            return redirect(url_for('index'))
        except ValueError:
            error = "User ID harus berupa angka."

    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/pick-genres', methods=['GET', 'POST'])
def pick_genres():
    """
    Halaman cold start untuk user baru.
    User memilih 1 atau lebih genre favorit.
    Pilihan disimpan di session untuk dipakai di halaman utama.
    """
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        selected = request.form.getlist('genres')
        if selected:
            session['selected_genres'] = selected
            return redirect(url_for('index'))

    return render_template('pick_genres.html', all_genres=ALL_GENRES)


@app.route('/')
def index():
    """
    Halaman utama — menampilkan rekomendasi.

    Alur:
    - Belum login → halaman kosong + prompt login
    - Login, user lama → rekomendasi personal via SVD
    - Login, user baru → rekomendasi via genre pilihan + avg rating
    """
    user_id = session.get('user_id', None)

    if user_id is None:
        return render_template('index.html',
                               recommendations=None,
                               user_id=None,
                               user_history=[],
                               mode=None)

    has_history = not ratings[ratings['userId'] == user_id].empty

    if has_history:
        recommendations = recommend_for_user(user_id, top_n=10)
        user_history = get_user_history(user_id)
        mode = 'personal'
    else:
        selected_genres = session.get('selected_genres', [])
        if not selected_genres:
            return redirect(url_for('pick_genres'))

        recommendations = recommend_by_genres(selected_genres, top_n=10)
        user_history = []
        mode = 'cold_start'

    return render_template('index.html',
                           recommendations=recommendations,
                           user_id=user_id,
                           user_history=user_history,
                           mode=mode)


if __name__ == '__main__':
    app.run(debug=True)