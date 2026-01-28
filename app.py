from flask import Flask, render_template, request, jsonify, send_from_directory
import spacy
import nltk
import PyPDF2
import bs4 as bs
import urllib.request
import re
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
from summarizer import Summarizer
import os

# --------------------------------------
# INITIAL SETUP
# --------------------------------------
nlp = spacy.load("en_core_web_sm")

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

lemmatizer = WordNetLemmatizer()

bert_model = Summarizer()

app = Flask(__name__, static_folder='static')

# --------------------------------------
# TF-IDF Functions (Keep your existing functions)
# --------------------------------------
def frequency_matrix(sentences):
    freq_matrix = {}
    stopWords = nlp.Defaults.stop_words

    for sent in sentences:
        freq_table = {}
        words = [word.text.lower() for word in sent if word.text.isalnum()]

        for word in words:
            word = lemmatizer.lemmatize(word)
            if word not in stopWords:
                freq_table[word] = freq_table.get(word, 0) + 1

        freq_matrix[sent[:15]] = freq_table

    return freq_matrix


def tf_matrix(freq_matrix):
    tf_matrix = {}
    for sent, freq_table in freq_matrix.items():
        total_words = len(freq_table)
        tf_matrix[sent] = {word: count / total_words for word, count in freq_table.items()}
    return tf_matrix


def sentences_per_words(freq_matrix):
    spw = {}
    for sent, f_table in freq_matrix.items():
        for word in f_table:
            spw[word] = spw.get(word, 0) + 1
    return spw


def idf_matrix(freq_matrix, spw, total_sentences):
    import math
    idf = {}
    for sent, f_table in freq_matrix.items():
        idf[sent] = {word: math.log10(total_sentences / float(spw[word])) for word in f_table}
    return idf


def tf_idf_matrix(tf_matrix_data, idf_matrix_data):
    tfidf = {}
    for (sent, f1), (_, f2) in zip(tf_matrix_data.items(), idf_matrix_data.items()):
        tfidf[sent] = {word: f1[word] * f2[word] for word in f1}
    return tfidf


def score_sentences(tfidf_matrix):
    scores = {}
    for sent, f_table in tfidf_matrix.items():
        if len(f_table) > 0:
            scores[sent] = sum(f_table.values()) / len(f_table)
    return scores


def tfidf_summary(sentences, scores, ratio=0.3):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    summary_len = max(1, round(len(sentences) * ratio))
    selected = set([s[0] for s in ranked[:summary_len]])

    result = []
    for sent in sentences:
        if sent[:15] in selected:
            result.append(sent.text)

    return " ".join(result)


# --------------------------------------
# TEXT RANK
# --------------------------------------
def textrank_summary(text, ratio=0.3):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    if len(sentences) < 2:
        return text

    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()

    sim_matrix = cosine_similarity(vectors)
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph)

    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary_len = max(1, round(len(sentences) * ratio))
    top_sentences = [sent for _, sent in ranked[:summary_len]]

    final = [s for s in sentences if s in top_sentences]
    return " ".join(final)


# --------------------------------------
# BERT SUMMARIZER
# --------------------------------------
def bert_summary(text, ratio=0.3):
    if len(text.split()) < 50:
        return text
    result = bert_model(text, ratio=ratio)
    return "".join(result).strip()


# --------------------------------------
# FLASK ROUTES
# --------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.json
        text = data["text"]
        algo = data["algorithm"]
        ratio = float(data["ratio"])

        if algo == "tfidf":
            doc = nlp(text)
            sentences = list(doc.sents)
            freq = frequency_matrix(sentences)
            tf = tf_matrix(freq)
            spw = sentences_per_words(freq)
            idf = idf_matrix(freq, spw, len(sentences))
            tfidf = tf_idf_matrix(tf, idf)
            scores = score_sentences(tfidf)
            summary = tfidf_summary(sentences, scores, ratio)

        elif algo == "textrank":
            summary = textrank_summary(text, ratio)

        elif algo == "bert":
            summary = bert_summary(text, ratio)
        else:
            return jsonify({"error": "Invalid algorithm selected"}), 400

        return jsonify({
            "summary": summary,
            "original_count": len(text.split()),
            "summary_count": len(summary.split())
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    # Create static folder if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    app.run(debug=True, host='0.0.0.0', port=5000)