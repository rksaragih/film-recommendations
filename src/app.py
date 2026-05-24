import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import linear_kernel
from deep_translator import GoogleTranslator

# =========================================
# LOAD MODEL
# =========================================

DATASET_PATH = "../Dataset/"
MODEL_PATH = "../models/"

movies = pd.read_csv(DATASET_PATH + "clean_movies.csv")
similarity = pickle.load(open(MODEL_PATH + "similarity.pkl", "rb"))
cf_avg_normalized = pickle.load(open(MODEL_PATH + "cf_predictions.pkl", "rb"))
tfidf_vectorizer = pickle.load(open(MODEL_PATH + "tfidf_vectorizer.pkl", "rb"))
tfidf_matrix = pickle.load(open(MODEL_PATH + "tfidf_matrix.pkl", "rb"))

app = Flask(__name__)

# =========================================
# TRANSLATOR
# =========================================

def translate_to_english(text):
    """Translate text to English if it's not already in English"""
    try:
        # Try to detect and translate
        translator = GoogleTranslator(source='auto', target='en')
        translated = translator.translate(text)
        return translated
    except:
        # If translation fails, return original text
        return text

# =========================================
# RECOMMENDATION FUNCTIONS
# =========================================

def recommend(movie_title, top_n=5, cbf_weight=0.5, cf_weight=0.5):
    """Recommend films based on exact movie title match"""
    if movie_title not in movies['title'].values:
        return None

    movie_idx = movies[movies['title'] == movie_title].index[0]
    cbf_scores = list(enumerate(similarity[movie_idx]))
    cbf_scores = [(i, score) for i, score in cbf_scores if i != movie_idx]

    recommendations = []
    for idx, cbf_score in cbf_scores:
        candidate_id = movies.iloc[idx]['id']
        cf_score = cf_avg_normalized.get(candidate_id, 0.0)
        final_score = (cbf_weight * cbf_score) + (cf_weight * float(cf_score))
        recommendations.append({
            'title': movies.iloc[idx]['title'],
            'cbf_score': round(cbf_score, 4),
            'cf_score': round(float(cf_score), 4),
            'final_score': round(final_score, 4)
        })

    recommendations = sorted(recommendations, key=lambda x: x['final_score'], reverse=True)
    return recommendations[:top_n]


def search_by_keyword(keyword, top_n=5, cbf_weight=0.5, cf_weight=0.5):
    """Search films based on keyword description (works for new/unknown films)"""
    # Translate keyword to English if needed
    keyword_en = translate_to_english(keyword)
    
    # Vectorize the keyword using the same TF-IDF vectorizer
    try:
        query_tfidf = tfidf_vectorizer.transform([keyword_en])
    except:
        return None
    
    # Compute similarity with all films
    similarities = linear_kernel(query_tfidf, tfidf_matrix).flatten()
    
    # Get top N indices
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    recommendations = []
    for rank, idx in enumerate(top_indices):
        cbf_score = float(similarities[idx])
        candidate_id = movies.iloc[idx]['id']
        cf_score = cf_avg_normalized.get(candidate_id, 0.0)
        final_score = (cbf_weight * cbf_score) + (cf_weight * float(cf_score))
        
        recommendations.append({
            'title': movies.iloc[idx]['title'],
            'cbf_score': round(cbf_score, 4),
            'cf_score': round(float(cf_score), 4),
            'final_score': round(final_score, 4)
        })
    
    return recommendations if recommendations[0]['cbf_score'] > 0 else None

# =========================================
# ROUTES
# =========================================

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    query = ''
    error = None
    search_mode = 'title'  # 'title' atau 'keyword'

    if request.method == 'POST':
        search_mode = request.form.get('search_mode', 'title').strip()
        
        if search_mode == 'title':
            query = request.form.get('movie_title', '').strip()
            if query:
                recommendations = recommend(query)
                if recommendations is None:
                    error = f"Film \"{query}\" tidak ditemukan di dataset. Coba gunakan pencarian keyword atau deskripsi."
        
        elif search_mode == 'keyword':
            query = request.form.get('keyword', '').strip()
            if query:
                recommendations = search_by_keyword(query)
                if recommendations is None:
                    error = f"Tidak ada hasil untuk keyword \"{query}\". Coba dengan keyword lain."

    movie_titles = sorted(movies['title'].dropna().unique().tolist())
    return render_template('index.html',
                           recommendations=recommendations,
                           query=query,
                           error=error,
                           search_mode=search_mode,
                           movie_titles=movie_titles)

if __name__ == '__main__':
    app.run(debug=True)