from functools import lru_cache

from .dehyphen import Scorer
from .flair_lm import load_language_model


class FlairScorer(Scorer):
    def __init__(
        self,
        lang="multi",
        cache=None,
        forward=True,
        backward=True,
        fast=False,
    ):
        """Score with Flair's character language models: en, es, de etc., see `flair_lm.MODELS`.

        `cache`: directory for downloaded models, defaults to `$FLAIR_CACHE_ROOT` or `~/.flair`
        """

        fast_suffix = "-fast" if fast else ""
        model_names = []
        if forward:
            model_names.append(f"{lang}-forward{fast_suffix}")
        if backward:
            model_names.append(f"{lang}-backward{fast_suffix}")

        self.lms = [load_language_model(x, cache) for x in model_names]
        self.score_text = lru_cache(maxsize=1024)(self._score_text)

    def _score_text(self, text):
        return float(sum([lm.perplexity(text) for lm in self.lms]))

    def score(self, texts):
        if min(map(len, texts)) < 2:
            raise ValueError("every option needs at least two characters")
        return list(map(self.score_text, texts))
