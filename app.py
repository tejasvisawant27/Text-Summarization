import os
import nltk
import numpy as np
from flask import Flask, render_template, request
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    num_sentences = int(request.form['num_sentences'])
    summarized_text = summary(text, num_sentences)
    return render_template('result.html', summary=summarized_text)

def summary(text, num_sentences):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Create a matrix to hold the similarity scores
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    # Create a dictionary to hold the sentence indices
    sentence_indices = {}
    for i in range(len(sentences)):
        sentence_indices[i] = sentences[i]

    # Create a matrix of sentence embeddings using word embeddings
    word_embeddings = {}
    with open(os.path.join(app.root_path, 'glove.6B.50d.txt'), encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs

    sentence_embeddings = []
    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        embeddings = [word_embeddings.get(token, np.zeros((50,))) for token in tokens]
        sentence_embedding = np.mean(embeddings, axis=0)
        sentence_embeddings.append(sentence_embedding)

    # Calculate the similarity scores
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                similarity_matrix[i][j] = 1 - cosine_distance(sentence_embeddings[i], sentence_embeddings[j])

    # Use PageRank to calculate sentence importance scores
    damping_factor = 0.85
    iterations = 10
    scores = np.ones(len(sentences)) / len(sentences)
    for i in range(iterations):
        new_scores = np.ones(len(sentences)) * (1 - damping_factor) / len(sentences)
        for j in range(len(sentences)):
            for k in range(len(sentences)):
                if similarity_matrix[k][j] != 0:
                    new_scores[j] += damping_factor * (similarity_matrix[k][j] / np.sum(similarity_matrix[k]))

        scores = new_scores

    # Sort the sentence indices by importance score
    ranked_indices = np.argsort(-scores)

    # Build the summary string
    summary = ""
    for i in range(min(num_sentences, len(ranked_indices))):
        summary += sentence_indices[ranked_indices[i]] + " "

    return summary

if __name__ == '__main__':
    app.run(debug=True)



