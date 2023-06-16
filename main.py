import json

import numpy as np
import spacy

# Load the pre-trained word vectors
nlp = spacy.load('en_core_web_lg')


def find_closest_words(input_word, word_array, threshold):
    # Convert the input word to its vector representation
    input_vector = nlp(input_word).vector

    closest_words = []

    # Iterate over each word in the array
    for word in word_array:
        # Convert the word to its vector representation
        word_vector = word['vector']

        # Calculate the cosine similarity between the input word vector and the current word vector
        similarity = input_vector.dot(word_vector) / (np.linalg.norm(input_vector) * np.linalg.norm(word_vector))

        # Update the closest word if the current similarity is higher
        closest_words.append({"word": word['word'], "similarity": similarity})

    sorted_words_similarities = sorted(closest_words, key=lambda x: x['similarity'], reverse=True)

    return [word_similar['word'] for word_similar in sorted_words_similarities[:threshold]]


def create_embeddings(words: list) -> list:
    return [{'word': word, 'vector': nlp(word).vector} for word in words]


def get_gpt_words(phrase: str, word_arr: list) -> set:
    words_for_gpt = []
    for word in phrase.split(' '):
        closest_words = find_closest_words(word, word_arr, 30)
        words_for_gpt.extend(closest_words)

    return set(words_for_gpt)


def load_words(filepath: str) -> list[str]:
    with open(filepath, 'r') as f:
        docs = json.load(f)

    return [doc['captions'][-1]['transcription'] for doc in docs]


def main():
    # words = set(load_words('translated_sign2mint.json') + (load_words('translate_signsuisse.json')))
    with open('embeddings.json', 'r') as f:
        load_words_embeddings = list(json.load(f))

    # arr_embedding_words = create_embeddings(list(words))
    words_for_gpt = get_gpt_words('hello world', load_words_embeddings)



if __name__ == '__main__':
    main()
