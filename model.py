import numpy as np
from utils import sigmoid


class Word2Vec:
    def __init__(self, vocab_size, learning_rate, n_embedding, words_vectorized):
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.W_in = np.random.randn(vocab_size, n_embedding) * 0.01
        self.W_out = np.zeros((vocab_size, n_embedding))

        freqs = np.bincount(words_vectorized) ** 0.75
        probs = freqs / np.sum(freqs)
        self.neg_table = self.neg_table = np.random.choice(vocab_size, size=10000000, p=probs)

    def sample_negatives(self, target_idx, context_idx, n_negatives):
        negs = []
        while len(negs) < n_negatives:
            idx = self.neg_table[np.random.randint(len(self.neg_table))]
            if idx != target_idx and idx != context_idx:
                negs.append(idx)
        return negs

    def train_step(self, target_idx, context_idx, neg_indices):
        target_v = self.W_in[target_idx]
        context_v = self.W_out[context_idx]
        negs_v = self.W_out[neg_indices]

        context_prob = sigmoid(np.dot(target_v, context_v))
        neg_prob = sigmoid(-(negs_v @ target_v))

        loss = -np.log(context_prob + 1e-9) - np.sum(np.log(neg_prob + 1e-9))

        context_err = context_prob - 1
        neg_err = 1 - neg_prob

        grad_target = context_err * context_v + neg_err @ negs_v
        grad_context = context_err * target_v
        grad_neg = np.outer(neg_err, target_v)

        self.W_in[target_idx] -= self.learning_rate * grad_target
        self.W_out[context_idx] -= self.learning_rate * grad_context
        self.W_out[neg_indices] -= self.learning_rate * grad_neg

        return loss

    def save_model(self, in_path, out_path):
        np.save(in_path, self.W_in)
        np.save(out_path, self.W_out)

    def load_model(self, in_path, out_path):
        self.W_in = np.load(in_path)
        self.W_out = np.load(out_path)
