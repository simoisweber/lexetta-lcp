from typing import Protocol
import numpy as np
from wordfreq import zipf_frequency

class Retriever(Protocol):
    def __call__(self, sample: dict, n: int) -> list:
        """
        Retrieves n items from the history based on *sample*

        Params:
            sample: a row dict (keys: task_id, annotator_id, corpus, sentence, token, complexity)
            n: number of items to return

        Returns:
            Returns a list of row dicts of length n
        """

class RandomRetriever:
    def __init__(self, history: list, seed: int = None):
        self.history = history
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample: dict, n: int) -> list:
        pool = [item for item in self.history if item["task_id"] != sample["task_id"]]
        indices = self.rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [pool[i] for i in indices]


class WordFrequencyRetriever:
    def __init__(self, history: list, lang: str = "en"):
        self.history = history
        self.lang = lang
        self._freqs = [zipf_frequency(item["token"], lang) for item in history]

    def __call__(self, sample: dict, n: int) -> list:
        query_freq = zipf_frequency(sample["token"], self.lang)
        pool = [(item, freq) for item, freq in zip(self.history, self._freqs) if item["task_id"] != sample["task_id"]]
        ranked = sorted(pool, key=lambda t: abs(t[1] - query_freq))
        return [item for item, _ in ranked[:min(n, len(ranked))]]

class CorpusRetriever:
    def __init__(self, history: list, seed: int = None):
        self.history = history
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample: dict, n: int) -> list:
        pool = [item for item in self.history if item["corpus"] == sample["corpus"] and item["task_id"] != sample["task_id"]]
        indices = self.rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [pool[i] for i in indices]
