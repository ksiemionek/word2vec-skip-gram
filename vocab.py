import numpy as np
from utils import load_tokens


class Vocabulary:
    def __init__(self, file_path: str, min_count: int) -> None:
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.min_count = min_count

        words = load_tokens(file_path)
        self.build(words)

        self.words_vectorized = np.array(
            [self.word_to_idx[word] for word in words if word in self.word_to_idx]
        )

    def build(self, words: list[str]) -> None:
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1

        idx = 0
        for word in words:
            if word not in self.word_to_idx and counts[word] >= self.min_count:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1

    def get_pairs(self, window_size: int):
        for idx, target in enumerate(self.words_vectorized):
            start = max(0, idx - window_size)
            end = min(len(self.words_vectorized), idx + window_size + 1)
            for j in range(start, end):
                if j != idx:
                    yield target, self.words_vectorized[j]

    def __len__(self) -> int:
        return len(self.word_to_idx)
