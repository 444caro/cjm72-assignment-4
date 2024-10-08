from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


# fetch dataset 
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# initialize vectorizer and LSA
stop_words = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=5000)
X = vectorizer.fit_transform(documents)

# perform LSA using TruncatedSVD
n_components = 100
svd_model = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd_model.fit_transform(X)

# function to process the query
def process_query(query):
    """
    Function to process the query
    Input: query (str)
    Output: query_vector (numpy array)
    """
    query_vector = vectorizer.transform([query])
    query_vector_reduced = svd_model.transform(query_vector)
    return query_vector_reduced

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    query_reduced = process_query(query)
    similarities = cosine_similarity(query_reduced, X_reduced)[0]
    # get the top 5 indices based on similarity scores
    top_indices = np.argsort(similarities)[-5:][::-1]
    top_similarities = similarities[top_indices]
    top_documents = [documents[i] for i in top_indices]
    return top_documents, top_similarities.tolist(), top_indices.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
