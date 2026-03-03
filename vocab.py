import numpy as np
from utils import load_tokens


class Vocabulary:
    def __init__(self, file_path):
        self.word_to_idx = {}
        self.idx_to_word = {}

        words = load_tokens(file_path)
        self.build(words)

        self.words_vectorized = np.array([self.word_to_idx[word] for word in words])

    def build(self, words):
        idx = 0
        for word in words:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1
