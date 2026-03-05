import numpy as np
from numpy.typing import NDArray


def load_tokens(file_path: str) -> list[str]:
    with open(file_path, "r") as f:
        text = f.read()
    return text.split()


def sigmoid(x: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-x))


def nearest_neighbors(
    word: str, vocab: "Vocabulary", w2v: "Word2Vec", top_n: int
) -> None:
    idx = vocab.word_to_idx[word]
    vec = w2v.W_in[idx]

    diffs = w2v.W_in - vec
    dists = np.sqrt(np.sum(diffs**2, axis=1))

    top = np.argsort(dists)[1 : top_n + 1]
    print(f"Word: {word}")
    for i in top:
        print(f"- {vocab.idx_to_word[i]}")


def train_model(
    vocab: "Vocabulary",
    w2v: "Word2Vec",
    epochs: int,
    window_size: int,
    n_negatives: int,
) -> None:
    for epoch in range(epochs):
        total_loss = 0
        n_pairs = 0

        for target, context in vocab.get_pairs(window_size):
            negatives = w2v.sample_negatives(target, context, n_negatives)
            loss = w2v.train_step(target, context, negatives)
            total_loss += loss
            n_pairs += 1

            if n_pairs % 100000 == 0:
                print(f"Pair {n_pairs}, avg loss: {total_loss / 100000:.4f}")
                total_loss = 0
