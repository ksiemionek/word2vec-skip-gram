import numpy as np


def load_tokens(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    return text.split()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
