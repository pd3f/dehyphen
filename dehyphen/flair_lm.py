"""Flair's character language models, reduced to what scoring perplexity needs.

Model names and URLs follow Flair's `FlairEmbeddings` (MIT, Alan Akbik et al.),
without the CLEF-HIPE models: their files are gone and they expect lowercased text.
The checkpoints are read with pickle and the LSTM runs with numpy, without PyTorch.
"""

import io
import math
import os
import pickle
import shutil
import urllib.request
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

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


def sigmoid(x):
    # equal to 1 / (1 + exp(-x)), without overflowing for large negative x
    return 0.5 * (1 + np.tanh(0.5 * x))


class LanguageModel:
    """Flair's `LanguageModel`: embedding, LSTM, optional projection, decoder."""

    def __init__(self, dictionary, is_forward_lm, nlayers, weights):
        self.dictionary = dictionary
        self.is_forward_lm = is_forward_lm
        self.nlayers = nlayers
        self.weights = weights

    def lstm(self, x, layer):
        w = self.weights
        w_hh = w[f"rnn.weight_hh_l{layer}"]
        bias = w[f"rnn.bias_ih_l{layer}"] + w[f"rnn.bias_hh_l{layer}"]
        inputs = x @ w[f"rnn.weight_ih_l{layer}"].T + bias
        h = np.zeros(w_hh.shape[1], dtype=np.float32)
        c = np.zeros_like(h)
        out = np.empty((len(inputs), len(h)), dtype=np.float32)
        for t, gates in enumerate(inputs):
            # PyTorch's gate order: input, forget, cell, output
            i, f, g, o = np.split(gates + w_hh @ h, 4)
            c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
            h = sigmoid(o) * np.tanh(c)
            out[t] = h
        return out

    def perplexity(self, text):
        if not self.is_forward_lm:
            text = text[::-1]
        ids = np.array([self.dictionary.get_idx_for_item(c) for c in text])
        w = self.weights
        x = w["encoder.weight"][ids[:-1]]
        for layer in range(self.nlayers):
            x = self.lstm(x, layer)
        if "proj.weight" in w:
            x = x @ w["proj.weight"].T + w["proj.bias"]
        logits = x @ w["decoder.weight"].T + w["decoder.bias"]
        logits -= logits.max(axis=1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        # cross entropy of each next character, averaged
        nll = -log_probs[np.arange(len(ids) - 1), ids[1:]].astype(np.float64).mean()
        return math.exp(nll)


class _Tensor:
    """Stands in for `torch._utils._rebuild_tensor_v2` until the storages are read."""

    def __init__(
        self, storage, offset, size, stride, grad=False, hooks=None, meta=None
    ):
        if meta:
            raise pickle.UnpicklingError(f"unsupported tensor metadata {meta}")
        self.storage = storage
        self.offset = offset
        self.size = size
        self.stride = stride

    def array(self, storages):
        data = storages[self.storage]
        n = math.prod(self.size)
        expected = [math.prod(self.size[i + 1 :]) for i in range(len(self.size))]
        strided = len(self.stride) != len(self.size) or any(
            d > 1 and s != e for d, s, e in zip(self.size, self.stride, expected)
        )
        outside = self.offset < 0 or min(self.size, default=0) < 0
        if strided or outside or self.offset + n > len(data):
            raise pickle.UnpicklingError("only contiguous tensors within their storage")
        # a copy is aligned; views into the read bytes make the LSTM about 15 % slower
        return data[self.offset : self.offset + n].reshape(self.size).copy()


_GLOBALS = {
    ("flair.data", "Dictionary"): Dictionary,
    ("collections", "OrderedDict"): OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): _Tensor,
    ("torch", "FloatStorage"): np.float32,
}


class _Unpickler(pickle.Unpickler):
    """Only resolves the globals Flair's checkpoints contain, so loading cannot run foreign code."""

    def __init__(self, file):
        super().__init__(file)
        # storage key: number of float32 elements
        self.storages = {}

    def find_class(self, module, name):
        if (module, name) not in _GLOBALS:
            raise pickle.UnpicklingError(
                f"{module}.{name} is not allowed in a Flair language model"
            )
        cls = _GLOBALS[module, name]
        # a fresh subclass, so a pickle setting attributes on it leaves dehyphen's classes alone
        return type(cls.__name__, (cls,), {}) if cls in (Dictionary, _Tensor) else cls

    def persistent_load(self, pid):
        # ("storage", type, key, location, numel), the legacy format adds a view
        view = pid[5] if len(pid) > 5 else None
        if pid[0] != "storage" or pid[1] is not np.float32 or view is not None:
            raise pickle.UnpicklingError(f"unsupported storage {pid[:2]}")
        self.storages[pid[2]] = pid[4]
        return pid[2]


def _floats(raw, n):
    if len(raw) != 4 * n:
        raise EOFError(f"storage of {len(raw)} bytes instead of {4 * n}")
    return np.frombuffer(raw, dtype="<f4")


# first object of a checkpoint in torch's legacy (non-zip) format
_LEGACY_MAGIC = 0x1950A86A20F9469CFC6C


def _read_legacy(f):
    """Pickles of magic number, version, system info, the object, the storage keys; then the storages."""
    if _Unpickler(f).load() != _LEGACY_MAGIC:
        raise pickle.UnpicklingError("not a PyTorch checkpoint")
    _Unpickler(f).load()
    if not _Unpickler(f).load().get("little_endian"):
        raise pickle.UnpicklingError("only little-endian checkpoints")
    unpickler = _Unpickler(f)
    state = unpickler.load()
    storages = {}
    for key in _Unpickler(f).load():
        n = unpickler.storages[key]
        if int.from_bytes(f.read(8), "little") != n:
            raise pickle.UnpicklingError(f"storage {key} has an unexpected size")
        storages[key] = _floats(f.read(4 * n), n)
    return state, storages


def _read_zip(path):
    """`<name>/data.pkl` holds the object, `<name>/data/<key>` each storage."""
    with zipfile.ZipFile(path) as z:
        pkl = next((n for n in z.namelist() if n.endswith("/data.pkl")), None)
        if pkl is None:
            raise pickle.UnpicklingError("no data.pkl in the checkpoint")
        prefix = pkl[: -len("data.pkl")]
        byteorder = prefix + "byteorder"
        if byteorder in z.namelist() and z.read(byteorder) != b"little":
            raise pickle.UnpicklingError("only little-endian checkpoints")
        unpickler = _Unpickler(io.BytesIO(z.read(pkl)))
        state = unpickler.load()
        storages = {
            key: _floats(z.read(f"{prefix}data/{key}"), n)
            for key, n in unpickler.storages.items()
        }
    return state, storages


def read_checkpoint(path):
    """The object a PyTorch checkpoint holds, with its tensors as numpy arrays in `state_dict`."""
    with open(path, "rb") as f:
        # as torch tells its formats apart: a zip file starts with a local file header
        if f.read(4) == b"PK\x03\x04":
            state, storages = _read_zip(path)
        else:
            f.seek(0)
            state, storages = _read_legacy(f)
    state["state_dict"] = {k: t.array(storages) for k, t in state["state_dict"].items()}
    return state


# seconds without data before a download fails
DOWNLOAD_TIMEOUT = 60


def cache_root(cache=None):
    """Flair's cache directory, so most models Flair already downloaded get reused."""
    return Path(cache or os.getenv("FLAIR_CACHE_ROOT", Path.home() / ".flair"))


def model_path(name, cache=None):
    name = name.lower()
    if name not in MODELS:
        raise ValueError(f"unknown Flair language model: {name}")
    url = MODELS[name]
    filename = url.rsplit("/", 1)[1]
    path = cache_root(cache) / "embeddings" / filename
    if filename == "best-lm.pt":
        # several models share this file name, so each gets its own subdirectory
        path = path.parent / name / filename
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        # a unique name per download, so concurrent workers never share a partial file
        part = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
        try:
            with (
                urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response,
                open(part, "wb") as f,
            ):
                shutil.copyfileobj(response, f)
                # http.client does not raise when the connection closes before Content-Length
                if f.tell() != int(response.headers.get("Content-Length", f.tell())):
                    raise OSError(f"incomplete download of {url}")
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)
    return path


def load_language_model(name, cache=None):
    path = model_path(name, cache)
    try:
        state = read_checkpoint(path)
        lm = LanguageModel(
            state["dictionary"],
            state["is_forward_lm"],
            state["nlayers"],
            state["state_dict"],
        )
        # one step touches every weight, so a damaged checkpoint fails here and not on first use
        lm.perplexity("ab")
    except Exception as e:
        raise RuntimeError(
            f"cannot load {path}; if it is damaged, delete it to download it again"
        ) from e
    return lm
