import numpy as np
from utils import sigmoid

class Word2Vec:
    def __init__(self, vocab_size, learning_rate, n_embedding, words_vectorized):
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.W_in = np.random.randn(vocab_size, n_embedding)
        self.W_out = np.zeros((vocab_size, n_embedding))

        freqs = np.bincount(words_vectorized) ** 0.75
        self.neg_probs = freqs / np.sum(freqs)

    def sample_negatives(self, target_idx, context_idx, n_negatives):
        negs = []
        while len(negs) < n_negatives:
            idx = np.random.choice(self.vocab_size, p=self.neg_probs)
            if idx != target_idx and idx != context_idx:
                negs.append(idx)
        return negs

