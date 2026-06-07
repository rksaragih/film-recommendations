# 🎬 Matchinema — Sistem Rekomendasi Film Personal

> Sistem rekomendasi film terpersonalisasi berbasis **Collaborative Filtering** menggunakan **Singular Value Decomposition (SVD)**, dilengkapi strategi penanganan user baru berbasis preferensi genre.

---

## 📌 Deskripsi

Proyek ini membangun sistem rekomendasi film yang **terpersonalisasi per user** — setiap user mendapatkan rekomendasi yang berbeda berdasarkan pola rating mereka sendiri dan kesamaan selera dengan user lain.

Sistem menggunakan pendekatan **Collaborative Filtering berbasis Matrix Factorization (SVD)** sebagai metode utama, dengan strategi berbeda tergantung kondisi user:

| Kondisi User | Metode | Penjelasan |
|---|---|---|
| **User lama** (punya riwayat rating) | Collaborative Filtering — SVD | Prediksi rating per user berdasarkan pola rating user-user lain yang seleranya serupa, lalu tampilkan film yang belum pernah ditonton |
| **User baru** (belum punya riwayat rating) | Filter genre + avg rating | User memilih genre favorit, sistem menampilkan film dengan rata-rata rating tertinggi dari pengguna lain (min. 50 penilai) |

Dataset yang digunakan:
- **TMDB (The Movie Database)** — metadata film: `movies_metadata.csv`, `credits.csv`, `keywords.csv`
- **MovieLens** — data interaksi pengguna: `ratings.csv`, `links.csv`

---

## 📁 Struktur Folder

```
film-recommendation/
├── Dataset/
│   ├── movies_metadata.csv     # Metadata film (TMDB)
│   ├── credits.csv             # Data cast & crew (TMDB)
│   ├── keywords.csv            # Keyword film (TMDB)
│   ├── ratings.csv             # Rating pengguna (MovieLens)
│   ├── links.csv               # Mapping movieId ↔ tmdbId
│   ├── clean_movies.csv        # Output preprocessing (id, title, genres, tags)
│   └── clean_ratings.csv       # Rating yang sudah dibersihkan
├── models/
│   ├── cf_predictions.pkl      # CF score rata-rata per film (fallback)
│   ├── svd_U.pkl               # Matriks user latent factors (n_users × k)
│   ├── svd_sigma.pkl           # Singular values SVD (k,)
│   ├── svd_Vt.pkl              # Matriks item latent factors (k × n_movies)
│   ├── user_ratings_mean.pkl   # Rata-rata rating per user
│   ├── user_id_to_idx.pkl      # Mapping userId → index SVD
│   ├── movie_id_to_idx_cf.pkl  # Mapping movieId → index SVD
│   └── cf_movie_ids.pkl        # Urutan movie ID di Vt
├── insight/
│   ├── 1_top_genre.png         # Visualisasi top 10 genre
│   ├── 2_distribusi_rating.png # Visualisasi distribusi rating
│   ├── 3_top_film_dirating.png # Visualisasi top 10 film terbanyak dirating
│   ├── 4_film_per_tahun.png    # Visualisasi jumlah film per tahun rilis
│   └── 5_top_user_aktif.png    # Visualisasi top 10 user paling aktif
├── src/
│   ├── preprocessing.py        # Pembersihan & transformasi data
│   ├── train_model.py          # Training Collaborative Filtering (SVD)
│   ├── evaluate_model.py       # Evaluasi rekomendasi personal & user baru
│   ├── app.py                  # Flask web app
│   └── spark_analysis.py       # Analisis & visualisasi data dengan Spark
├── templates/
│   ├── index.html              # Halaman utama (rekomendasi)
│   ├── login.html              # Halaman login
│   └── pick_genres.html        # Halaman pilih genre (user baru)
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalasi

### 1. Clone repository

```bash
git clone https://github.com/username/film-recommendation.git
cd film-recommendation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

| Library | Kegunaan |
|---|---|
| `pandas`, `numpy` | Manipulasi data |
| `scikit-learn` | Preprocessing teks |
| `scipy` | SVD sparse matrix decomposition |
| `nltk` | Stemming teks |
| `flask` | Web framework |
| `pyspark`, `matplotlib`, `seaborn` | Analisis & visualisasi data |

### 3. Download NLTK data

```python
import nltk
nltk.download('punkt')
```

### 4. Siapkan dataset

Letakkan semua file dataset di folder `Dataset/`. Dataset dapat diunduh dari:
- [Kaggle — The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

---

## 🚀 Cara Menjalankan

Jalankan secara **berurutan** dari folder `src/`:

### 1. Preprocessing

```bash
python preprocessing.py
```

Membersihkan dan menggabungkan dataset:
- Validasi & konversi ID film
- Merge `movies_metadata`, `credits`, `keywords`
- Parse kolom JSON: genres, cast, crew, keywords
- Bangun kolom `tags` (overview + genres + keywords + cast + director) → lowercase + stemming
- Bersihkan dan normalisasi data rating

Output:
- `Dataset/clean_movies.csv` — kolom: `id`, `title`, `genres`, `tags`
- `Dataset/clean_ratings.csv` — kolom: `userId`, `movieId`, `rating`

### 2. Training Model

```bash
python train_model.py
```

Melatih model Collaborative Filtering berbasis SVD:
- Bangun sparse user-item matrix dari data rating (`csr_matrix`)
- Mean-centering per user untuk menghilangkan bias rating
- SVD decomposition dengan `k=50` latent factors
- Simpan komponen `U`, `sigma`, `Vt` untuk prediksi personal per user
- Hitung CF score rata-rata sebagai fallback

Output: 8 file `.pkl` di folder `models/`

### 3. Evaluasi Model

```bash
python evaluate_model.py
```

Mengevaluasi dua skenario:

**Skenario 1 — User Lama:**
- Bagian A (Deskriptif): Top-N rekomendasi personal beserta skor prediksi
- Bagian B (Metrik): MAE & RMSE — bandingkan prediksi SVD vs rating asli pada film yang sudah ditonton sebagai *sanity check* model

**Skenario 2 — User Baru:**
- Bagian A (Deskriptif): Top-N film populer berdasarkan genre pilihan
- Bagian B (Metrik): Distribusi avg rating dari hasil rekomendasi

Untuk mengganti user atau genre yang dievaluasi, ubah bagian `__main__`:

```python
evaluate_personal(user_id=1, top_n=10)
evaluate_cold_start(selected_genres=["Action", "Comedy"], top_n=10)
```

### 4. Flask Web App

```bash
python app.py
```

Buka browser di `http://localhost:5000`.

### 5. (Opsional) Analisis Data dengan Spark

```bash
python spark_analysis.py
```

Menghasilkan 5 visualisasi yang tersimpan di folder `insight/`.

---

## 🔐 Alur Penggunaan Web App

```
Buka /
    │
    └─ Belum login → halaman kosong + tombol Login
            │
            ▼
        /login — input User ID (angka)
            │
            ├─ User punya riwayat rating
            │       └─ → Halaman utama
            │              Rekomendasi personal via SVD
            │              (film belum ditonton, prediksi rating tertinggi)
            │
            └─ User belum punya riwayat rating
                    └─ → /pick-genres
                           Pilih 1+ genre favorit
                                │
                                ▼
                           Halaman utama
                           Film populer sesuai genre
                           (avg rating tertinggi, min. 50 penilai)
```

**Catatan:** Tidak ada registrasi — siapapun bisa masuk dengan User ID angka apapun. Jika ID tidak dikenal atau belum punya riwayat rating, user otomatis diarahkan ke halaman pilih genre.

---

## 📊 Metodologi

### Collaborative Filtering — SVD (User Lama)

Collaborative Filtering bekerja dengan asumsi bahwa user yang memiliki pola rating serupa di masa lalu akan menyukai film yang sama di masa depan.

**1. Bangun sparse user-item matrix**

Data rating direpresentasikan sebagai matrix `(n_users × n_movies)` menggunakan `csr_matrix` untuk efisiensi memori, karena sebagian besar entri kosong (*sparse*).

**2. Mean-centering per user**

Rating setiap user dikurangi rata-rata ratingnya sendiri:
```
demeaned[u][i] = rating[u][i] - mean_rating[u]
```
Ini menghilangkan *user bias* — misalnya user yang selalu memberi bintang 5 vs user yang pelit memberi bintang tinggi.

**3. SVD Decomposition**

Matrix didekomposisi menjadi tiga komponen:
```
R ≈ U × diag(sigma) × Vt
```
- `U` → profil laten setiap user `(n_users × k)`
- `sigma` → bobot setiap dimensi laten `(k,)`
- `Vt` → profil laten setiap film `(k × n_movies)`

Dengan `k=50` latent factors, SVD menangkap preferensi tersembunyi seperti kecenderungan menyukai film bergenre gelap, film keluarga, atau film dengan tempo cepat — tanpa label eksplisit.

**4. Prediksi rating per user**

```
predicted[u] = U[u] × diag(sigma) × Vt + mean_rating[u]
```

Hasilnya adalah vektor prediksi rating user `u` untuk semua film, dikembalikan ke skala asli (0–5), lalu dinormalisasi ke rentang 0–1 untuk keperluan ranking.

**5. Filter film yang sudah ditonton**

Film yang sudah pernah diberi rating oleh user dikeluarkan dari daftar, sehingga rekomendasi hanya berisi film baru yang belum ditonton.

---

### Penanganan User Baru (Cold Start)

Karena SVD membutuhkan riwayat rating untuk membuat prediksi, user yang belum pernah memberi rating tidak bisa dilayani oleh model. Strategi yang digunakan:

1. User memilih satu atau lebih genre favorit
2. Sistem memfilter film yang memiliki minimal satu genre yang dipilih
3. Film diurutkan berdasarkan rata-rata rating dari semua pengguna lain
4. Hanya film yang sudah dinilai oleh **minimal 50 user** yang ditampilkan — untuk menghindari film obscure yang kebetulan mendapat rating sempurna dari sedikit penilai

---

## 📈 Dataset Summary

| Metrik | Nilai |
|---|---|
| Jumlah Film | ~45.608 |
| Jumlah User | ~270.896 |
| Jumlah Rating | ~25.981.403 |
| Sparsity | >99,99% |
| Rating Range | 0,5 – 5,0 |

---

## 🎯 Contoh Output Evaluasi

### Skenario 1 — User Lama (User ID: 1)

```
════════════════════════════════════════════════════════════
EVALUASI USER ID: 1
════════════════════════════════════════════════════════════

  A. TOP-10 REKOMENDASI PERSONAL
  #    Judul                                    Genre                      Score
  1    Natural Born Killers                     Crime, Thriller, Drama    0.8410
  2    Alien                                    Horror, Action, Thriller  0.8392
  3    Blade Runner                             Science Fiction, Drama    0.8388
  4    Men in Black                             Action, Adventure, Comedy 0.8369
  5    GoodFellas                               Drama, Crime              0.8363
  ...

  B. AKURASI PREDIKSI SVD (Sanity Check)
  Judul                           Actual  Prediksi  Selisih
  The Godfather                      5.0      4.99     0.01
  Fight Club                         4.0      4.15     0.15
  The Dark Knight                    4.0      4.18     0.18
  The Hobbit: An Unexpected Journey  0.5      4.24     3.74  ← outlier

  MAE  (rata-rata selisih)        0.6717
  RMSE (penalti error besar)      0.9643
  ~ Cukup baik — prediksi meleset < 1 bintang
```

### Skenario 2 — User Baru (Genre: Action, Comedy)

```
════════════════════════════════════════════════════════════
EVALUASI COLD START
Genre dipilih: Action, Comedy
════════════════════════════════════════════════════════════

  A. TOP-10 REKOMENDASI
  #    Judul                           Genre                  Avg ★
  1    Band of Brothers                Action, Drama, War      4.39
  2    Seven Samurai                   Action, Drama           4.26
  3    The Dark Knight                 Drama, Action, Crime    4.18
  4    Inception                       Action, Thriller        4.16
  5    The Matrix                      Action, Science Fiction 4.15
  ...

  B. DISTRIBUSI RATING
  Total kandidat film              6.337 film
  Avg rating tertinggi              4.39
  Avg rating terendah               4.15
  Rata-rata avg rating              4.20
```

---

## ⚠️ Catatan & Keterbatasan

**Bias menuju rata-rata** — SVD cenderung menarik prediksi ke arah rata-rata populasi. Film yang sangat disukai atau sangat dibenci oleh seorang user bisa meleset jauh dari prediksi (lihat contoh The Hobbit: actual 0.5, prediksi 4.24). Ini adalah karakteristik inheren collaborative filtering, bukan bug.

**Cold start murni rule-based** — Penanganan user baru tidak menggunakan machine learning, melainkan filter dan pengurutan sederhana berdasarkan rata-rata rating. Seiring user mulai memberi rating pada film, sistem secara otomatis beralih ke rekomendasi personal berbasis SVD.

**Minimum rating threshold** — Filter minimal 50 penilai pada cold start bertujuan menjaga kualitas rekomendasi agar tidak didominasi film yang terlalu obscure.