import ast
import os
import re
import pandas as pd
from nltk.stem.porter import PorterStemmer

# =========================================
# DATASET PATH
# =========================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATASET_PATH = os.path.join(BASE_DIR, 'Dataset')

CREDITS_PATH = os.path.join(DATASET_PATH, 'credits.csv')
KEYWORDS_PATH = os.path.join(DATASET_PATH, 'keywords.csv')
MOVIES_PATH = os.path.join(DATASET_PATH, 'movies_metadata.csv')
RATINGS_PATH = os.path.join(DATASET_PATH, 'ratings.csv')
LINKS_PATH = os.path.join(DATASET_PATH, 'links.csv')

OUTPUT_PATH = os.path.join(DATASET_PATH, 'clean_movies.csv')
OUTPUT_RATINGS_PATH = os.path.join(DATASET_PATH, 'clean_ratings.csv')

# =========================================
# HELPER FUNCTIONS
# =========================================

def is_valid_int_id(value):
    if pd.isna(value):
        return False
    text = str(value).strip()
    if re.fullmatch(r'[0-9]+\.0', text):
        text = text[:-2]
    return bool(re.fullmatch(r'[0-9]+', text))


def safe_int_id(value):
    text = str(value).strip()
    return int(float(text))


def parse_json_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text == '':
        return []
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def get_names(data):
    if not isinstance(data, list):
        return []
    return [item.get('name', '').strip() for item in data if isinstance(item, dict) and item.get('name')]


def get_top_cast(data, top_n=5):
    names = get_names(data)
    return names[:top_n]


def get_director(data):
    if not isinstance(data, list):
        return ''
    for item in data:
        if isinstance(item, dict) and item.get('job') == 'Director':
            return item.get('name', '').strip()
    return ''


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def stem_text(text, stemmer):
    return ' '.join(stemmer.stem(word) for word in text.split() if word)


# =========================================
# LOAD DATASET
# =========================================

movies = pd.read_csv(MOVIES_PATH, low_memory=False)
movies.columns = movies.columns.str.strip()
movies = movies[movies['id'].apply(is_valid_int_id)].copy()
movies['id'] = movies['id'].apply(safe_int_id)

credits = pd.read_csv(CREDITS_PATH, low_memory=False)
credits = credits[credits['id'].apply(is_valid_int_id)].copy()
credits['id'] = credits['id'].apply(safe_int_id)

keywords = pd.read_csv(KEYWORDS_PATH, low_memory=False)
keywords.columns = keywords.columns.str.strip()
keywords = keywords[keywords['id'].apply(is_valid_int_id)].copy()
keywords['id'] = keywords['id'].apply(safe_int_id)

ratings = pd.read_csv(RATINGS_PATH)
links = pd.read_csv(LINKS_PATH)
links = links.dropna(subset=['tmdbId'])
links['tmdbId'] = links['tmdbId'].apply(safe_int_id)
links['movieId'] = links['movieId'].apply(safe_int_id)

# =========================================
# MERGE DATA
# =========================================

movies = movies[['id', 'title', 'overview', 'genres']].copy()
credits = credits[['id', 'cast', 'crew']].copy()
keywords = keywords[['id', 'keywords']].copy()

merged = movies.merge(credits, on='id', how='left')
merged = merged.merge(keywords, on='id', how='left')
merged = merged.drop_duplicates(subset=['id'])

# =========================================
# HANDLE MISSING VALUES
# =========================================

merged['overview'] = merged['overview'].fillna('')
merged['genres'] = merged['genres'].fillna('[]')
merged['keywords'] = merged['keywords'].fillna('[]')
merged['cast'] = merged['cast'].fillna('[]')
merged['crew'] = merged['crew'].fillna('[]')
merged['title'] = merged['title'].fillna('')

# =========================================
# TRANSFORM FEATURES
# =========================================

merged['genres'] = merged['genres'].apply(parse_json_list).apply(get_names)
merged['keywords'] = merged['keywords'].apply(parse_json_list).apply(get_names)
merged['cast'] = merged['cast'].apply(parse_json_list).apply(get_top_cast)
merged['director'] = merged['crew'].apply(parse_json_list).apply(get_director)

merged['tags'] = (
    merged['overview'] + ' ' +
    merged['genres'].apply(lambda x: ' '.join(x)) + ' ' +
    merged['keywords'].apply(lambda x: ' '.join(x)) + ' ' +
    merged['cast'].apply(lambda x: ' '.join(x)) + ' ' +
    merged['director']
)
merged['tags'] = merged['tags'].apply(clean_text)

stemmer = PorterStemmer()
merged['tags'] = merged['tags'].apply(lambda text: stem_text(text, stemmer))

# Keep only movies with some tag content
final_movies = merged[['id', 'title', 'genres', 'tags']].copy()
final_movies = final_movies[final_movies['tags'].str.strip() != '']

final_movies.to_csv(OUTPUT_PATH, index=False)

print(f'Preprocessing selesai! Total film: {len(final_movies)}')
print(final_movies.head())

# =========================================
# PREPROCESS RATINGS
# =========================================

ratings = ratings.merge(links[['movieId', 'tmdbId']], on='movieId', how='inner')

# Hanya simpan rating untuk film yang ada di clean_movies.csv
valid_tmdb_ids = set(final_movies['id'].values)
ratings = ratings[ratings['tmdbId'].isin(valid_tmdb_ids)].copy()
 
# Select dan rename kolom
clean_ratings = ratings[['userId', 'tmdbId', 'rating']].copy()
clean_ratings = clean_ratings.rename(columns={'tmdbId': 'movieId'})
 
# Kalau user menilai film yang sama lebih dari sekali, ambil rata-ratanya
clean_ratings = clean_ratings.groupby(['userId', 'movieId'], as_index=False)['rating'].mean()
 
clean_ratings.to_csv(OUTPUT_RATINGS_PATH, index=False)
 
print(f'\nPreprocessing ratings selesai! Total rating: {len(clean_ratings)}')
print(f'Total user unik : {clean_ratings["userId"].nunique()}')
print(f'Total film unik : {clean_ratings["movieId"].nunique()}')
print(clean_ratings.head())