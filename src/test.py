import pandas as pd
import ast

DATASET_PATH = "../Dataset/"

movies = pd.read_csv(DATASET_PATH + "movies_metadata.csv", low_memory=False)
ratings = pd.read_csv(DATASET_PATH + "ratings.csv", low_memory=False)
credits = pd.read_csv(DATASET_PATH + "credits.csv", low_memory=False)
keywords = pd.read_csv(DATASET_PATH + "keywords.csv", low_memory=False)
links = pd.read_csv(DATASET_PATH + "links.csv", low_memory=False)
ratings_s = pd.read_csv(DATASET_PATH + "ratings_small.csv", low_memory=False)
links_s = pd.read_csv(DATASET_PATH + "links_small.csv", low_memory=False)


print(f"Jumlah Data Film: {len(movies)} Data serta {len(movies.columns)} Kolom: ")
print(f"Jumlah Data Rating: {len(ratings)} Data serta {len(ratings.columns)} Kolom: ")
print(f"Jumlah Data Kredit: {len(credits)} Data serta {len(credits.columns)} Kolom: ")
print(f"Jumlah Data Kata Kunci: {len(keywords)} Data serta {len(keywords.columns)} Kolom: ")
print(f"Jumlah Data Tautan: {len(links)} Data serta {len(links.columns)} Kolom: ")
print(f"Jumlah Data Rating Kecil: {len(ratings_s)} Data serta {len(ratings_s.columns)} Kolom: ")
print(f"Jumlah Data Tautan Kecil: {len(links_s)} Data serta {len(links_s.columns)} Kolom: ")

# clean_movies = pd.read_csv(DATASET_PATH + "clean_movies.csv", low_memory=False)
# clean_ratings = pd.read_csv(DATASET_PATH + "clean_ratings.csv", low_memory=False)

# n_movies = movies['id'].nunique()
# n_users = ratings['userId'].nunique()
# n_ratings = len(ratings)
# rating_min = ratings['rating'].min()
# rating_max = ratings['rating'].max()
# total_possible = n_users * n_movies
# n_empty = total_possible - n_ratings
# sparsity = (n_empty / total_possible) * 100

# n_movies_clean = clean_movies['id'].nunique()
# n_users_clean = clean_ratings['userId'].nunique()
# n_ratings_clean = len(clean_ratings)
# rating_min_clean = clean_ratings['rating'].min()
# rating_max_clean = clean_ratings['rating'].max()
# total_possible_clean = n_users_clean * n_movies_clean
# n_empty_clean = total_possible_clean - n_ratings_clean
# sparsity_clean = (n_empty_clean / total_possible_clean) * 100

# print(f"Jumlah Data Sebelum Cleaning:")
# print(f"Jumlah Film: {n_movies}")
# print(f"Jumlah Pengguna: {n_users}")    
# print(f"Jumlah Rating: {n_ratings}")
# print(f"Rentang Rating: {rating_min} to {rating_max}")
# print(f"Total Rating yang Mungkin: {total_possible}")
# print(f"Jumlah Rating Kosong: {n_empty}")
# print(f"Sparsity: {sparsity:.2f}%")

# print(f"Jumlah Data Sesudah Cleaning:")
# print(f"Jumlah Film: {n_movies_clean}")
# print(f"Jumlah Pengguna: {n_users_clean}")    
# print(f"Jumlah Rating: {n_ratings_clean}")
# print(f"Rentang Rating: {rating_min_clean} to {rating_max_clean}")
# print(f"Total Rating yang Mungkin: {total_possible_clean}")
# print(f"Jumlah Rating Kosong: {n_empty_clean}")
# print(f"Sparsity: {sparsity_clean:.2f}%")