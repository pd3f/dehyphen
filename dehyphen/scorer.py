from functools import lru_cache

import flair
from flair.embeddings import FlairEmbeddings

from .dehyphen import Scorer


class FlairScorer(Scorer):
    def __init__(
        self, lang="multi", cache=None, forward=True, backward=True, fast=False
    ):
        """choose some flair language models: en, es, de etc.
        https://github.com/flairNLP/flair/blob/8c09e62d9a5a3c227b9ca0fb9f214de9620d4ca0/flair/embeddings/token.py#L431

        https://github.com/flairNLP/flair/blob/fca44a2a0fed0be3d8a2ef6940a42fa22c625d31/resources/docs/embeddings/FLAIR_EMBEDDINGS.md
        """

        fast_suffix = "-fast" if fast else ""
        model_names = []
        if forward:
            model_names.append(f"{lang}-forward{fast_suffix}")
        if backward:
            model_names.append(f"{lang}-backward{fast_suffix}")

        self.lms = [FlairEmbeddings(x).lm for x in model_names]

        # change default location of flair models
        if not cache is None:
            flair.cache_root = cache

    @lru_cache(maxsize=1024)
    def score_text(self, text):
        return float(sum([lm.calculate_perplexity(text) for lm in self.lms]))

    def score(self, texts):
        assert min(map(len, texts)) > 1, "flair fails if there is only one character"
        return list(map(self.score_text, texts))
