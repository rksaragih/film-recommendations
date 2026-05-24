# 🎬 Implementasi Hybrid Filtering untuk Sistem Rekomendasi Film Menggunakan TF-IDF dan Matrix Factorization

> Menggabungkan Content-Based Filtering dan Collaborative Filtering untuk rekomendasi film yang lebih akurat dan personal.

---

## 📌 Deskripsi

Proyek ini membangun sistem rekomendasi film menggunakan pendekatan **Hybrid Filtering** yang menggabungkan dua metode utama:

- **Content-Based Filtering (CBF)** — merekomendasikan film berdasarkan kemiripan konten (genre, keyword, cast, director, overview) menggunakan **TF-IDF + Cosine Similarity**.
- **Collaborative Filtering (CF)** — merekomendasikan film berdasarkan pola rating pengguna menggunakan **SVD (Singular Value Decomposition)** untuk matrix factorization.

Kedua skor digabungkan dengan **weighted average** untuk rekomendasi hybrid yang akurat dan personal.

**Fitur Terbaru**:
- 🔍 **Dual Search Mode**: 
  - **Judul Film**: Input judul film yang ada di dataset
  - **Keyword Search**: Input deskripsi/keyword untuk menemukan film mirip (bahkan yang tidak ada di dataset!)
- 🌐 **Multi-language Support**: Sistem otomatis menerjemahkan input dari Bahasa Indonesia ke Inggris

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
│   ├── ratings_small.csv       # Subset rating untuk testing
│   ├── links.csv               # Mapping movieId ke tmdbId
│   ├── clean_movies.csv        # Output preprocessing
│   └── clean_ratings.csv       # Rating yang sudah dibersihkan
├── models/
│   ├── movies.pkl                 # DataFrame film (CBF + metadata)
│   ├── similarity.pkl             # Matrix similarity CBF (TF-IDF cosine)
│   ├── cf_predictions.pkl         # CF score normalized per film (SVD)
│   ├── movie_id_to_idx.pkl        # Mapping movie ID ke index
│   ├── tfidf_vectorizer.pkl       # TF-IDF vectorizer (untuk keyword search)
│   └── tfidf_matrix.pkl           # TF-IDF matrix (untuk keyword search)
├── insight/
│   ├── 1_top_genre.png         # Visualisasi top 10 genre
│   ├── 2_distribusi_rating.png # Visualisasi distribusi rating
│   ├── 3_top_film_dirating.png # Visualisasi top 10 film dirating
│   ├── 4_film_per_tahun.png    # Visualisasi film per tahun rilis
│   └── 5_top_user_aktif.png    # Visualisasi top 10 user aktif
├── src/
│   ├── preprocessing.py        # Pembersihan & transformasi data
│   ├── train_model.py          # Training CBF (TF-IDF) + CF (SVD)
│   ├── evaluate_model.py       # Evaluasi & uji rekomendasi hybrid
│   ├── app.py                  # Flask web app untuk rekomendasi
│   ├── spark_analysis.py       # Analisis data dengan Spark + visualisasi
│   └── test.py                 # Testing & debugging
├── requirements.txt            # Dependency list
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

Dependencies:
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — TF-IDF vectorizer & metrics
- `scipy` — SVD sparse matrix factorization
- `nltk` — text preprocessing
- `pyspark` — distributed analysis & visualization
- `matplotlib`, `seaborn` — plotting
- `flask` — web framework (opsional, untuk `app.py`)

### 3. Download NLTK data (optional)

```python
import nltk
nltk.download('punkt')
```

### 4. Siapkan dataset

Letakkan semua file dataset di folder `Dataset/`:
- `movies_metadata.csv`, `credits.csv`, `keywords.csv` → dari [TMDB](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- `ratings.csv`, `links.csv` → dari [MovieLens](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

---

## 🚀 Cara Menjalankan

Jalankan secara berurutan dari folder `src/`:

### 1. Preprocessing

```bash
python preprocessing.py
```

Membersihkan dan menggabungkan dataset:
- Hapus duplicate, missing values
- Extract genre, keywords dari JSON
- Kombinasi metadata dengan ratings
- Output: `Dataset/clean_movies.csv`, `Dataset/clean_ratings.csv`

### 2. Training Model

```bash
python train_model.py
```

Melatih model hybrid:
- **CBF**: TF-IDF vectorization pada `tags` → cosine similarity matrix
- **CF**: SVD pada sparse user-item matrix dari ratings → normalized CF scores per film
- Output: `models/movies.pkl`, `models/similarity.pkl`, `models/cf_predictions.pkl`, `models/movie_id_to_idx.pkl`, `models/tfidf_vectorizer.pkl`, `models/tfidf_matrix.pkl`

**Note**: Dengan dataset penuh (~26M ratings, 45K+ films), training CF bisa memakan waktu beberapa menit.

### 3. Evaluasi Model

```bash
python evaluate_model.py
```

Menguji rekomendasi hybrid dan menampilkan:
- Top-N recommendations untuk film tertentu
- CBF score (content similarity)
- CF score (collaborative rating)
- Final hybrid score (weighted average)

### 4. (Optional) Analisis Data dengan Spark

```bash
python spark_analysis.py
```

Menghasilkan 5 visualisasi Spark:
1. Top 10 genre terpopuler
2. Distribusi rating pengguna
3. Top 10 film terbanyak dirating
4. Jumlah film per tahun rilis
5. Top 10 user paling aktif

Output tersimpan di folder `insight/` sebagai PNG.

### 5. (Optional) Flask Web App

```bash
python app.py
```

Menjalankan web interface untuk sistem rekomendasi di `http://localhost:5000`.

**Fitur Web App**:
- **Mode 1 - Cari Judul Film**: 
  - Input judul film exact yang ada di dataset
  - Autocomplete dari daftar film yang tersedia
  - Cocok jika tahu judul film secara pasti
  
- **Mode 2 - Cari dengan Keyword** (Fitur Baru):
  - Input deskripsi bebas (genre, aktor, tema, dll)
  - Support Bahasa Indonesia & Inggris (auto-translate)
  - Cocok untuk film baru atau yang belum ada di dataset
  - Contoh: "film action dengan aktor Tom Cruise", "sci-fi thriller", "animated family movie"

**Note**: Keyword search menggunakan TF-IDF + cosine similarity tanpa memerlukan film ada di database.

---

## 📊 Metodologi

### Content-Based Filtering (CBF)
1. **Text Preprocessing**: Ekstrak dan kombinasi genre, keywords, cast, director, overview menjadi `tags`
2. **TF-IDF Vectorization**: Convert tags menjadi vektor numerik dengan bobot term frequency-inverse document frequency
   - `max_features=5000`, `ngram_range=(1,2)`, `sublinear_tf=True`
3. **Similarity Computation**: Hitung cosine similarity antar film menggunakan `linear_kernel`
   - Output: Matrix similarity (45K+ × 45K+)

### Collaborative Filtering (CF)
1. **Sparse Matrix Construction**: Bangun user-item matrix dari ratings data (~26M entries)
   - Menggunakan `csr_matrix` untuk efisiensi memori
2. **Normalization**: Demean ratings per user untuk menghilangkan bias rating behavior
3. **SVD Decomposition**: Faktorisasi matriks dengan k=50 latent factors
   - Menggunakan `scipy.sparse.linalg.svds` untuk sparse SVD
4. **Score Aggregation**: Hitung mean predicted rating per film dan normalisasi ke [0,1]

### Hybrid Recommendation
- **Final Score** = `cbf_weight × cbf_score + cf_weight × cf_score`
- Default: `cbf_weight=0.5`, `cf_weight=0.5` (dapat dikonfigurasi)

### Keyword Search (Fitur Baru)
1. **Language Translation**: 
   - Input user diterjemahkan ke Bahasa Inggris menggunakan `deep-translator`
   - Mendukung input Bahasa Indonesia dan Inggris
   
2. **TF-IDF Vectorization**: 
   - Keyword user di-vectorize dengan TF-IDF vocabulary yang sama dengan training CBF
   - Tidak memerlukan film ada di database sebelumnya
   
3. **Similarity Computation**:
   - Hitung cosine similarity antara keyword vector dan semua film tags
   - Ranking berdasarkan similarity + CF scores
   
4. **Advantages**:
   - ✅ Dapat merekomendasikan film baru yang belum ada di dataset
   - ✅ Support natural language queries (deskripsi, genre, aktor, tema)
   - ✅ Lebih fleksibel dibanding exact title matching

---

## 📈 Dataset Summary

| Metrik | Nilai |
|---|---|
| Jumlah Film | ~45,608 |
| Jumlah User | ~270,896 |
| Jumlah Rating | ~25,981,403 |
| Sparsity | >99.99% |
| Rating Range | 0.5 - 5.0 |

---

## 🎯 Contoh Output

### Rekomendasi Hybrid

```
==========================================================
REKOMENDASI FILM UNTUK: Toy Story
(CBF weight: 0.5 | CF weight: 0.5)
==========================================================
1. Toy Story 2
   CBF Score   : 0.4474
   CF Score    : 0.7823
   Final Score : 0.6149

2. Toy Story 3
   CBF Score   : 0.4103
   CF Score    : 0.8156
   Final Score : 0.6130

3. Partysaurus Rex
   CBF Score   : 0.3113
   CF Score    : 0.6234
   Final Score : 0.4674

==================================================
                  CBF      CF     Final
Average Score   0.3559  0.7405  0.5482
Highest Score   0.4474  0.8156  0.6149
Lowest Score    0.3113  0.6234  0.4674
```

---

## 🛠️ Teknologi & Dependencies

| Library | Versi | Kegunaan |
|---|---|---|
| `pandas` | ≥2.0.0 | Data manipulation & cleaning |
| `scikit-learn` | ≥1.3.0 | TF-IDF vectorizer, metrics |
| `scipy` | ≥1.10.0 | SVD sparse matrix decomposition |
| `numpy` | latest | Numerical computations |
| `nltk` | ≥3.8.0 | Text preprocessing |
| `pyspark` | ≥3.5.0 | Distributed data analysis |
| `matplotlib` | ≥3.7.0 | Plotting utilities |
| `seaborn` | ≥0.12.0 | Statistical visualization |
| `flask` | (optional) | Web framework untuk app.py |
| `deep-translator` | ≥1.10.0 | Multi-language translation (Bahasa Indonesia → Inggris) |

---

## 👨‍💻 Informasi Proyek

- **Institusi**: Institut Pertanian Bogor (IPB University)
- **Mata Kuliah**: Big Data
- **Semester**: 6
- **Tahun**: 2026
