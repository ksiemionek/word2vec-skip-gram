import numpy as np


def load_tokens(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    return text.split()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def nearest_neighbors(word, vocab, w2v, top_n):
    idx = vocab.word_to_idx[word]
    vec = w2v.W_in[idx]

    diffs = w2v.W_in - vec
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))

    top = np.argsort(dists)[1 : top_n + 1]
    print(f"Word: {word}")
    for i in top:
        print(f"- {vocab.idx_to_word[i]}")
