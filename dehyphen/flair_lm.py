"""Flair's character language models, reduced to what scoring perplexity needs.

Model names and URLs follow Flair's `FlairEmbeddings` (MIT, Alan Akbik et al.),
without the CLEF-HIPE models: their files are gone and they expect lowercased text.
"""

import math
import os
import pickle
import shutil
import urllib.request
import uuid
from pathlib import Path

import torch
from torch import nn

HU = "https://flair.informatik.hu-berlin.de/resources/embeddings/flair/"

MODELS = {
    # multilingual
    "multi-forward": HU + "lm-jw300-forward-v0.1.pt",
    "multi-backward": HU + "lm-jw300-backward-v0.1.pt",
    "multi-v0-forward": HU + "lm-multi-forward-v0.1.pt",
    "multi-v0-backward": HU + "lm-multi-backward-v0.1.pt",
    "multi-v0-forward-fast": HU + "lm-multi-forward-fast-v0.1.pt",
    "multi-v0-backward-fast": HU + "lm-multi-backward-fast-v0.1.pt",
    "multi-forward-fast": HU + "lm-multi-forward-fast-v0.1.pt",
    "multi-backward-fast": HU + "lm-multi-backward-fast-v0.1.pt",
    # English
    "en-forward": HU + "news-forward-0.4.1.pt",
    "en-backward": HU + "news-backward-0.4.1.pt",
    "en-forward-fast": HU + "lm-news-english-forward-1024-v0.2rc.pt",
    "en-backward-fast": HU + "lm-news-english-backward-1024-v0.2rc.pt",
    "news-forward": HU + "news-forward-0.4.1.pt",
    "news-backward": HU + "news-backward-0.4.1.pt",
    "news-forward-fast": HU + "lm-news-english-forward-1024-v0.2rc.pt",
    "news-backward-fast": HU + "lm-news-english-backward-1024-v0.2rc.pt",
    "mix-forward": HU + "lm-mix-english-forward-v0.2rc.pt",
    "mix-backward": HU + "lm-mix-english-backward-v0.2rc.pt",
    # Amharic
    "am-forward": "http://ltdata1.informatik.uni-hamburg.de/amharic/models/flair/best-lm.pt",
    # Arabic
    "ar-forward": HU + "lm-ar-opus-large-forward-v0.1.pt",
    "ar-backward": HU + "lm-ar-opus-large-backward-v0.1.pt",
    # Bulgarian
    "bg-forward-fast": HU + "lm-bg-small-forward-v0.1.pt",
    "bg-backward-fast": HU + "lm-bg-small-backward-v0.1.pt",
    "bg-forward": HU + "lm-bg-opus-large-forward-v0.1.pt",
    "bg-backward": HU + "lm-bg-opus-large-backward-v0.1.pt",
    # Czech
    "cs-forward": HU + "lm-cs-opus-large-forward-v0.1.pt",
    "cs-backward": HU + "lm-cs-opus-large-backward-v0.1.pt",
    "cs-v0-forward": HU + "lm-cs-large-forward-v0.1.pt",
    "cs-v0-backward": HU + "lm-cs-large-backward-v0.1.pt",
    # Danish
    "da-forward": HU + "lm-da-opus-large-forward-v0.1.pt",
    "da-backward": HU + "lm-da-opus-large-backward-v0.1.pt",
    # German
    "de-forward": HU + "lm-mix-german-forward-v0.2rc.pt",
    "de-backward": HU + "lm-mix-german-backward-v0.2rc.pt",
    "de-historic-ha-forward": HU + "lm-historic-hamburger-anzeiger-forward-v0.1.pt",
    "de-historic-ha-backward": HU + "lm-historic-hamburger-anzeiger-backward-v0.1.pt",
    "de-historic-wz-forward": HU + "lm-historic-wiener-zeitung-forward-v0.1.pt",
    "de-historic-wz-backward": HU + "lm-historic-wiener-zeitung-backward-v0.1.pt",
    "de-historic-rw-forward": HU + "redewiedergabe_lm_forward.pt",
    "de-historic-rw-backward": HU + "redewiedergabe_lm_backward.pt",
    # Spanish
    "es-forward": HU + "lm-es-forward.pt",
    "es-backward": HU + "lm-es-backward.pt",
    "es-forward-fast": HU + "lm-es-forward-fast.pt",
    "es-backward-fast": HU + "lm-es-backward-fast.pt",
    "es-clinical-forward": HU + "es-clinical-forward.pt",
    "es-clinical-backward": HU + "es-clinical-backward.pt",
    # Basque
    "eu-forward": HU + "lm-eu-opus-large-forward-v0.2.pt",
    "eu-backward": HU + "lm-eu-opus-large-backward-v0.2.pt",
    "eu-v1-forward": HU + "lm-eu-opus-large-forward-v0.1.pt",
    "eu-v1-backward": HU + "lm-eu-opus-large-backward-v0.1.pt",
    "eu-v0-forward": HU + "lm-eu-large-forward-v0.1.pt",
    "eu-v0-backward": HU + "lm-eu-large-backward-v0.1.pt",
    # Persian
    "fa-forward": HU + "lm-fa-opus-large-forward-v0.1.pt",
    "fa-backward": HU + "lm-fa-opus-large-backward-v0.1.pt",
    # Finnish
    "fi-forward": HU + "lm-fi-opus-large-forward-v0.1.pt",
    "fi-backward": HU + "lm-fi-opus-large-backward-v0.1.pt",
    # French
    "fr-forward": HU + "lm-fr-charlm-forward.pt",
    "fr-backward": HU + "lm-fr-charlm-backward.pt",
    # Hebrew
    "he-forward": HU + "lm-he-opus-large-forward-v0.1.pt",
    "he-backward": HU + "lm-he-opus-large-backward-v0.1.pt",
    # Hindi
    "hi-forward": HU + "lm-hi-opus-large-forward-v0.1.pt",
    "hi-backward": HU + "lm-hi-opus-large-backward-v0.1.pt",
    # Croatian
    "hr-forward": HU + "lm-hr-opus-large-forward-v0.1.pt",
    "hr-backward": HU + "lm-hr-opus-large-backward-v0.1.pt",
    # Indonesian
    "id-forward": HU + "lm-id-opus-large-forward-v0.1.pt",
    "id-backward": HU + "lm-id-opus-large-backward-v0.1.pt",
    # Italian
    "it-forward": HU + "lm-it-opus-large-forward-v0.1.pt",
    "it-backward": HU + "lm-it-opus-large-backward-v0.1.pt",
    # Japanese
    "ja-forward": HU + "japanese-forward.pt",
    "ja-backward": HU + "japanese-backward.pt",
    # Malayalam
    "ml-forward": "https://raw.githubusercontent.com/qburst/models-repository/master/FlairMalayalamModels/ml-forward.pt",
    "ml-backward": "https://raw.githubusercontent.com/qburst/models-repository/master/FlairMalayalamModels/ml-backward.pt",
    # Dutch
    "nl-forward": HU + "lm-nl-opus-large-forward-v0.1.pt",
    "nl-backward": HU + "lm-nl-opus-large-backward-v0.1.pt",
    "nl-v0-forward": HU + "lm-nl-large-forward-v0.1.pt",
    "nl-v0-backward": HU + "lm-nl-large-backward-v0.1.pt",
    # Norwegian
    "no-forward": HU + "lm-no-opus-large-forward-v0.1.pt",
    "no-backward": HU + "lm-no-opus-large-backward-v0.1.pt",
    # Polish
    "pl-forward": HU + "lm-polish-forward-v0.2.pt",
    "pl-backward": HU + "lm-polish-backward-v0.2.pt",
    "pl-opus-forward": HU + "lm-pl-opus-large-forward-v0.1.pt",
    "pl-opus-backward": HU + "lm-pl-opus-large-backward-v0.1.pt",
    # Portuguese
    "pt-forward": HU + "lm-pt-forward.pt",
    "pt-backward": HU + "lm-pt-backward.pt",
    # PubMed
    "pubmed-forward": HU + "pubmed-forward.pt",
    "pubmed-backward": HU + "pubmed-backward.pt",
    "pubmed-2015-forward": HU + "pubmed-2015-fw-lm.pt",
    "pubmed-2015-backward": HU + "pubmed-2015-bw-lm.pt",
    # Slovenian
    "sl-forward": HU + "lm-sl-opus-large-forward-v0.1.pt",
    "sl-backward": HU + "lm-sl-opus-large-backward-v0.1.pt",
    "sl-v0-forward": HU + "lm-sl-large-forward-v0.1.pt",
    "sl-v0-backward": HU + "lm-sl-large-backward-v0.1.pt",
    # Swedish
    "sv-forward": HU + "lm-sv-opus-large-forward-v0.1.pt",
    "sv-backward": HU + "lm-sv-opus-large-backward-v0.1.pt",
    "sv-v0-forward": HU + "lm-sv-large-forward-v0.1.pt",
    "sv-v0-backward": HU + "lm-sv-large-backward-v0.1.pt",
    # Tamil
    "ta-forward": HU + "lm-ta-opus-large-forward-v0.1.pt",
    "ta-backward": HU + "lm-ta-opus-large-backward-v0.1.pt",
    # Ukrainian
    "uk-forward": "https://huggingface.co/dchaplinsky/flair-uk-forward/resolve/main/best-lm.pt",
    "uk-backward": "https://huggingface.co/dchaplinsky/flair-uk-backward/resolve/main/best-lm.pt",
}


class Dictionary:
    """Character-to-index mapping, unpickled from Flair's `flair.data.Dictionary`."""

    def get_idx_for_item(self, item):
        return self.item2idx.get(item.encode("utf-8"), 0)


class LanguageModel(nn.Module):
    def __init__(
        self, dictionary, is_forward_lm, hidden_size, nlayers, embedding_size, nout
    ):
        super().__init__()
        self.dictionary = dictionary
        self.is_forward_lm = is_forward_lm
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.encoder = nn.Embedding(len(dictionary.idx2item), embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, nlayers)
        self.proj = nn.Linear(hidden_size, nout) if nout is not None else None
        self.decoder = nn.Linear(nout or hidden_size, len(dictionary.idx2item))

    def forward(self, ids):
        weight = next(self.parameters())
        hidden = weight.new_zeros(self.nlayers, 1, self.hidden_size)
        output, _ = self.rnn(self.encoder(ids), (hidden, hidden.clone()))
        if self.proj is not None:
            output = self.proj(output)
        return self.decoder(output)

    @torch.inference_mode()
    def perplexity(self, text):
        if not self.is_forward_lm:
            text = text[::-1]
        device = next(self.parameters()).device
        ids = torch.tensor(
            [self.dictionary.get_idx_for_item(c) for c in text], device=device
        )
        logits = self.forward(ids[:-1].unsqueeze(1)).squeeze(1)
        return math.exp(nn.functional.cross_entropy(logits, ids[1:]).item())


# torch resolves storage types itself, these are the remaining globals in Flair's checkpoints
_ALLOWED_GLOBALS = frozenset(
    {("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2")}
)


class _Unpickler(pickle.Unpickler):
    """Only resolves the globals Flair's checkpoints contain, so loading cannot run foreign code."""

    def find_class(self, module, name):
        if (module, name) == ("flair.data", "Dictionary"):
            return Dictionary
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"{module}.{name} is not allowed in a Flair language model"
        )


class _SafePickle:
    Unpickler = _Unpickler

    @staticmethod
    def load(f, **kwargs):
        return _Unpickler(f, **kwargs).load()


def cache_root(cache=None):
    """Same location as Flair, so already downloaded models get reused."""
    return Path(cache or os.getenv("FLAIR_CACHE_ROOT", Path.home() / ".flair"))


def model_path(name, cache=None):
    name = name.lower()
    if name not in MODELS:
        raise ValueError(f"unknown Flair language model: {name}")
    url = MODELS[name]
    filename = url.rsplit("/", 1)[1]
    path = cache_root(cache) / "embeddings" / filename
    if filename == "best-lm.pt":
        # several models share this file name, Flair keeps them apart by subdirectory
        path = path.parent / name / filename
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        # a unique name per download, so concurrent workers never share a partial file
        part = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
        try:
            with urllib.request.urlopen(url) as response, open(part, "wb") as f:
                shutil.copyfileobj(response, f)
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)
    return path


def load_language_model(name, cache=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(
        model_path(name, cache),
        map_location=device,
        pickle_module=_SafePickle,
        weights_only=False,
    )
    model = LanguageModel(
        state["dictionary"],
        state["is_forward_lm"],
        state["hidden_size"],
        state["nlayers"],
        state["embedding_size"],
        state["nout"],
    )
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval()
