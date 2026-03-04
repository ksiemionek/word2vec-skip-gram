from vocab import Vocabulary
from model import Word2Vec
from config import *
from utils import nearest_neighbors, train_model


def main():
    # TRAIN THE MODEL
    # vocab = Vocabulary(FILE_PATH, MIN_COUNT)
    # w2v = Word2Vec(len(vocab), LEARNING_RATE, N_EMBEDDING, vocab.words_vectorized)
    # train_model(vocab, w2v, EPOCHS, WINDOW_SIZE, N_NEGATIVES)
    # w2v.save_model("W_in", "W_out")

    # LOAD THE TRAINED MODEL
    vocab = Vocabulary(FILE_PATH, MIN_COUNT)
    w2v = Word2Vec(len(vocab), LEARNING_RATE, N_EMBEDDING, vocab.words_vectorized)
    w2v.load_model("W_in.npy", "W_out.npy")

    nearest_neighbors("jet", vocab, w2v, 5)
    nearest_neighbors("brain", vocab, w2v, 5)
    nearest_neighbors("king", vocab, w2v, 5)
    nearest_neighbors("queen", vocab, w2v, 5)
    nearest_neighbors("apple", vocab, w2v, 5)


if __name__ == "__main__":
    main()
