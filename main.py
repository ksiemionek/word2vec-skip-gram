from vocab import Vocabulary
from model import Word2Vec
from config import *
from utils import nearest_neighbors, train_model


def main():
    vocab = Vocabulary(FILE_PATH, MIN_COUNT)
    w2v = Word2Vec(len(vocab), LEARNING_RATE, N_EMBEDDING, vocab.words_vectorized)

    if TRAIN:
        train_model(vocab, w2v, EPOCHS, WINDOW_SIZE, N_NEGATIVES)
        w2v.save_model("model/W_in", "model/W_out")
    else:
        w2v.load_model("model/W_in.npy", "model/W_out.npy")

    nearest_neighbors("jet", vocab, w2v, 5)
    nearest_neighbors("brains", vocab, w2v, 5)
    nearest_neighbors("king", vocab, w2v, 5)
    nearest_neighbors("queen", vocab, w2v, 5)
    nearest_neighbors("apple", vocab, w2v, 5)


if __name__ == "__main__":
    main()
