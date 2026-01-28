# -------------------------------------------------------
# Text Summarizer – Updated with Percentage-Based Summary
# -------------------------------------------------------

import sys
import math
import bs4 as bs
import urllib.request
import re
import PyPDF2
import nltk
from nltk.stem import WordNetLemmatizer 
import spacy

# Download required NLTK data
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

# Initialize spaCy
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()

from summarizer import Summarizer
bert_model = Summarizer()


def bert_summary(text, ratio=0.3):
    if len(text.split()) < 50:
        return text  # BERT works best with long text

    result = bert_model(text, ratio=ratio)
    summarized_text = ''.join(result)
    return summarized_text.strip()

# -------------------------------------------------------
# Step 1: Reading Input Text
# -------------------------------------------------------

def file_text(filepath):
    with open(filepath, encoding="utf-8") as f:
        text = f.read().replace("\n", '')
        return text


def pdfReader(pdf_path):
    with open(pdf_path, 'rb') as pdfFileObject:
        pdfReader = PyPDF2.PdfReader(pdfFileObject)
        count = len(pdfReader.pages)

        print("\nTotal Pages in pdf =", count)

        choice = input("Do you want to read entire pdf ? [Y]/N : ")

        if choice.lower() == 'n':
            start_page = int(input("Enter start page number (index 0): "))
            end_page = int(input(f"Enter end page number (less than {count}): "))

            if start_page < 0 or start_page >= count:
                print("Invalid Start Page")
                sys.exit()

            if end_page < 0 or end_page >= count:
                print("Invalid End Page")
                sys.exit()

        else:
            start_page = 0
            end_page = count - 1

        content = ""
        for i in range(start_page, end_page + 1):
            content += pdfReader.pages[i].extract_text()

        return content


def wiki_text(url):
    scrap_data = urllib.request.urlopen(url)
    article = scrap_data.read()
    parsed = bs.BeautifulSoup(article, 'lxml')
    paragraphs = parsed.find_all('p')

    article_text = ""
    for p in paragraphs:
        article_text += p.text

    article_text = re.sub(r'\[[0-9]*\]', '', article_text)
    return article_text


# -------------------------------------------------------
# Step 2: Asking User for Input Method
# -------------------------------------------------------

input_text_type = int(input(
"Select one way of inputting your text:\n"
"1. Type Text (Copy-Paste)\n"
"2. Load from .txt file\n"
"3. Load from .pdf file\n"
"4. Load from Wikipedia URL\n\n"
))

if input_text_type == 1:
    text = input("Enter your text:\n\n")

elif input_text_type == 2:
    txt_path = input("Enter .txt file path: ")
    text = file_text(txt_path)

elif input_text_type == 3:
    file_path = input("Enter PDF file path: ")
    text = pdfReader(file_path)

elif input_text_type == 4:
    wiki_url = input("Enter Wikipedia URL: ")
    text = wiki_text(wiki_url)

else:
    print("Invalid input option.")
    sys.exit()


# -------------------------------------------------------
# Step 3: TF-IDF Utility Functions
# -------------------------------------------------------

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
        tf_table = {}
        total_words = len(freq_table)

        for word, count in freq_table.items():
            tf_table[word] = count / total_words

        tf_matrix[sent] = tf_table

    return tf_matrix


def sentences_per_words(freq_matrix):
    sent_per_words = {}

    for sent, f_table in freq_matrix.items():
        for word in f_table:
            sent_per_words[word] = sent_per_words.get(word, 0) + 1

    return sent_per_words


def idf_matrix(freq_matrix, sent_per_words, total_sentences):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table:
            idf_table[word] = math.log10(total_sentences / float(sent_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def tf_idf_matrix(tf_matrix_data, idf_matrix_data):
    tf_idf_matrix = {}

    for (sent, f_table1), (_, f_table2) in zip(tf_matrix_data.items(), idf_matrix_data.items()):
        tf_idf_table = {}

        for word in f_table1:
            tf_idf_table[word] = float(f_table1[word] * f_table2[word])

        tf_idf_matrix[sent] = tf_idf_table

    return tf_idf_matrix


def score_sentences(tf_idf_matrix):
    sentence_scores = {}

    for sent, f_table in tf_idf_matrix.items():
        total = sum(f_table.values())
        count = len(f_table)
        if count > 0:
            sentence_scores[sent] = total / count

    return sentence_scores


# -------------------------------------------------------
# NEW FUNCTION — Percentage-Based Summary
# -------------------------------------------------------

def percentage_summary(sentences, sentence_scores, ratio=0.3):
    ranked = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    summary_len = max(1, int(len(sentences) * ratio))
    selected_keys = set([s[0] for s in ranked[:summary_len]])

    summary = ""
    for sentence in sentences:
        if sentence[:15] in selected_keys:
            summary += " " + sentence.text

    return summary.strip()


# -------------------------------------------------------
# Step 4: Process TF-IDF and Create Summary
# -------------------------------------------------------

# Word count
original_words = [w for w in text.split() if w.isalnum()]
original_word_count = len(original_words)

# Convert text to spaCy doc
doc = nlp(text)
sentences = list(doc.sents)

# TF-IDF
freq_matrix_data = frequency_matrix(sentences)
tf_matrix_data = tf_matrix(freq_matrix_data)
sent_per_words = sentences_per_words(freq_matrix_data)
idf_matrix_data = idf_matrix(freq_matrix_data, sent_per_words, len(sentences))
tf_idf_data = tf_idf_matrix(tf_matrix_data, idf_matrix_data)
sentence_scores = score_sentences(tf_idf_data)

# Ask user for summary size
print("\nChoose Summary Percentage:")
print("1. 10%")
print("2. 20%")
print("3. 30% (Recommended)")
print("4. 50% (Large Summary)")

choice = int(input("\nEnter option (1-4): "))
ratio_map = {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.50}
ratio = ratio_map.get(choice, 0.30)

# Generate Summary
summary = percentage_summary(sentences, sentence_scores, ratio)

# -------------------------------------------------------
# Step 5: Output
# -------------------------------------------------------

print("\n********************* SUMMARY *********************\n")
print(summary)
print("\n***************************************************")

print("\nOriginal word count =", original_word_count)
print("Summary word count  =", len(summary.split()))
