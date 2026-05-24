import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# =========================================
# SPARK SESSION
# =========================================

spark = SparkSession.builder \
    .appName("FilmRecommendationInsight") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# =========================================
# DATASET PATH
# =========================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_PATH = os.path.join(BASE_DIR, 'Dataset')
OUTPUT_DIR = os.path.join(BASE_DIR, 'insight')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================
# LOAD DATASET
# =========================================

movies_sdf = spark.read.csv(
    os.path.join(DATASET_PATH, 'movies_metadata.csv'),
    header=True, inferSchema=True
)

ratings_sdf = spark.read.csv(
    os.path.join(DATASET_PATH, 'ratings.csv'),
    header=True, inferSchema=True
)

links_sdf = spark.read.csv(
    os.path.join(DATASET_PATH, 'links.csv'),
    header=True, inferSchema=True
)

# Register sebagai temporary view agar bisa diquery dengan spark.sql()
movies_sdf.createOrReplaceTempView("movies")
ratings_sdf.createOrReplaceTempView("ratings")
links_sdf.createOrReplaceTempView("links")

# =========================================
# STYLE VISUALISASI
# =========================================

sns.set_theme(style="darkgrid")
PALETTE = "viridis"
FIG_SIZE = (10, 5)

def save_fig(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Disimpan: {path}")

# =========================================
# 1. TOP 10 GENRE TERPOPULER
# =========================================
# genres di movies_metadata berbentuk JSON string, misal:
# [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]
# Kita explode dengan split sederhana menggunakan regex di Spark SQL

print("\n[1/5] Top 10 Genre Terpopuler...")

genre_df = spark.sql("""
    SELECT genres FROM movies
    WHERE genres IS NOT NULL AND genres != '[]'
""").toPandas()

# Parse genre di pandas karena butuh ast.literal_eval
import ast

def extract_genres(val):
    try:
        items = ast.literal_eval(val)
        return [item['name'] for item in items if isinstance(item, dict)]
    except:
        return []

all_genres = []
for val in genre_df['genres']:
    all_genres.extend(extract_genres(val))

genre_counts = pd.Series(all_genres).value_counts().head(10).reset_index()
genre_counts.columns = ['genre', 'jumlah_film']

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.barplot(data=genre_counts, x='jumlah_film', y='genre', palette=PALETTE, ax=ax)
ax.set_title('Top 10 Genre Terpopuler', fontsize=14, fontweight='bold')
ax.set_xlabel('Jumlah Film')
ax.set_ylabel('Genre')
for i, v in enumerate(genre_counts['jumlah_film']):
    ax.text(v + 20, i, str(v), va='center', fontsize=9)
save_fig('1_top_genre.png')

# =========================================
# 2. DISTRIBUSI RATING
# =========================================

print("[2/5] Distribusi Rating...")

rating_dist = spark.sql("""
    SELECT rating, COUNT(*) AS jumlah
    FROM ratings
    GROUP BY rating
    ORDER BY rating
""").toPandas()

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.barplot(data=rating_dist, x='rating', y='jumlah', palette=PALETTE, ax=ax)
ax.set_title('Distribusi Rating Pengguna', fontsize=14, fontweight='bold')
ax.set_xlabel('Rating')
ax.set_ylabel('Jumlah Rating')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
save_fig('2_distribusi_rating.png')

# =========================================
# 3. TOP 10 FILM TERBANYAK DIRATING
# =========================================

print("[3/5] Top 10 Film Terbanyak Dirating...")

top_rated_movies = spark.sql("""
    SELECT l.tmdbId, COUNT(r.rating) AS jumlah_rating, AVG(r.rating) AS avg_rating
    FROM ratings r
    JOIN links l ON r.movieId = l.movieId
    WHERE l.tmdbId IS NOT NULL
    GROUP BY l.tmdbId
    ORDER BY jumlah_rating DESC
    LIMIT 10
""").toPandas()

top_rated_movies['tmdbId'] = top_rated_movies['tmdbId'].astype(int)

# Ambil judul film dari movies_metadata
movies_pd = spark.sql("""
    SELECT id, title FROM movies
    WHERE id IS NOT NULL
""").toPandas()

movies_pd['id'] = pd.to_numeric(movies_pd['id'], errors='coerce').dropna().astype(int)
top_rated_movies = top_rated_movies.merge(
    movies_pd, left_on='tmdbId', right_on='id', how='left'
)
top_rated_movies['title'] = top_rated_movies['title'].fillna('Unknown')

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.barplot(data=top_rated_movies, x='jumlah_rating', y='title', palette=PALETTE, ax=ax)
ax.set_title('Top 10 Film Terbanyak Dirating', fontsize=14, fontweight='bold')
ax.set_xlabel('Jumlah Rating')
ax.set_ylabel('Film')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
for i, v in enumerate(top_rated_movies['jumlah_rating']):
    ax.text(v + 100, i, f'{v:,}', va='center', fontsize=9)
save_fig('3_top_film_dirating.png')

# =========================================
# 4. JUMLAH FILM PER TAHUN RILIS
# =========================================

print("[4/5] Jumlah Film per Tahun Rilis...")

film_per_tahun = spark.sql("""
    SELECT YEAR(TRY_CAST(release_date AS DATE)) AS tahun, COUNT(*) AS jumlah_film
    FROM movies
    WHERE release_date IS NOT NULL
      AND release_date != ''
      AND TRY_CAST(release_date AS DATE) IS NOT NULL
      AND YEAR(TRY_CAST(release_date AS DATE)) BETWEEN 1950 AND 2017
    GROUP BY tahun
    ORDER BY tahun
""").toPandas()

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.lineplot(data=film_per_tahun, x='tahun', y='jumlah_film', color='steelblue', ax=ax)
ax.fill_between(film_per_tahun['tahun'], film_per_tahun['jumlah_film'], alpha=0.3, color='steelblue')
ax.set_title('Jumlah Film per Tahun Rilis (1950–2017)', fontsize=14, fontweight='bold')
ax.set_xlabel('Tahun')
ax.set_ylabel('Jumlah Film')
save_fig('4_film_per_tahun.png')

# =========================================
# 5. TOP 10 USER PALING AKTIF
# =========================================

print("[5/5] Top 10 User Paling Aktif...")

top_users = spark.sql("""
    SELECT userId, COUNT(*) AS jumlah_rating, AVG(rating) AS avg_rating
    FROM ratings
    GROUP BY userId
    ORDER BY jumlah_rating DESC
    LIMIT 10
""").toPandas()

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.barplot(data=top_users, x='jumlah_rating', y=top_users['userId'].astype(str), palette=PALETTE, ax=ax)
ax.set_title('Top 10 User Paling Aktif', fontsize=14, fontweight='bold')
ax.set_xlabel('Jumlah Rating Diberikan')
ax.set_ylabel('User ID')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
for i, row in enumerate(top_users.itertuples()):
    ax.text(row.jumlah_rating + 50, i, f'{row.jumlah_rating:,}\n(avg: {row.avg_rating:.1f})', va='center', fontsize=8)
save_fig('5_top_user_aktif.png')

# =========================================
# STOP SPARK SESSION
# =========================================

spark.stop()

print(f"\nSelesai! Semua visualisasi disimpan di folder: {OUTPUT_DIR}")