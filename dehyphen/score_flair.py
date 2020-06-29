import flair
from flair.embeddings import FlairEmbeddings


def score(texts, lang="multi", cache=None):
    # choose some flair language models: en, es, de etc.
    # https://github.com/flairNLP/flair/blob/8c09e62d9a5a3c227b9ca0fb9f214de9620d4ca0/flair/embeddings/token.py#L431

    # change default location of flair models
    if not cache is None:
        flair.cache_root = cache

    model_names = (f"{lang}-forward", f"{lang}-backward")
    lms = [FlairEmbeddings(x).lm for x in model_names]
    results = map(lambda x: sum([lm.calculate_perplexity(x) for lm in lms]), texts)
    return [float(r) for r in results]

