"""Write the tiny checkpoints `test_flair_lm.py` reads, and their perplexities as PyTorch computes them.

Needs PyTorch, which dehyphen itself does not: `python tests/data/make_checkpoints.py`
"""

import itertools
import json
import math
import sys
import types
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

DATA = Path(__file__).parent
TEXTS = ["ab ba", "aab", "b a b a", "bbbb"]


class Dictionary:
    pass


# pickles as `flair.data.Dictionary`, like the published checkpoints
Dictionary.__module__ = "flair.data"
sys.modules["flair"] = types.ModuleType("flair")
sys.modules["flair.data"] = types.ModuleType("flair.data")
sys.modules["flair.data"].Dictionary = Dictionary


class LanguageModel(nn.Module):
    """Flair's `LanguageModel` without training code."""

    def __init__(self, is_forward_lm, hidden_size, nlayers, embedding_size, nout):
        super().__init__()
        self.is_forward_lm = is_forward_lm
        self.encoder = nn.Embedding(4, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, nlayers)
        self.proj = nn.Linear(hidden_size, nout) if nout is not None else None
        self.decoder = nn.Linear(nout or hidden_size, 4)

    @torch.inference_mode()
    def perplexity(self, ids, text):
        if not self.is_forward_lm:
            text = text[::-1]
        ids = torch.tensor([ids[c.encode()] for c in text])
        output, _ = self.rnn(self.encoder(ids[:-1].unsqueeze(1)))
        if self.proj is not None:
            output = self.proj(output)
        logits = self.decoder(output).squeeze(1)
        return math.exp(nn.functional.cross_entropy(logits, ids[1:]).item())


def shared_storage(state_dict):
    """The tensors as views into one storage, at offsets, as in Flair's `multi-fast` models."""
    flat = torch.cat([t.flatten() for t in state_dict.values()])
    sizes = [t.numel() for t in state_dict.values()]
    offsets = itertools.accumulate(sizes, initial=0)
    return OrderedDict(
        (k, flat[o : o + t.numel()].view(t.shape))
        for (k, t), o in zip(state_dict.items(), offsets)
    )


def save(name, zip_format, shared, **config):
    dictionary = Dictionary()
    dictionary.idx2item = [b"<unk>", b"a", b"b", b" "]
    dictionary.item2idx = {c: i for i, c in enumerate(dictionary.idx2item)}
    lm = LanguageModel(**config).eval()
    state_dict = lm.state_dict()
    if shared:
        state_dict = shared_storage(state_dict)
    state = {"dictionary": dictionary, "state_dict": state_dict, "dropout": 0.1}
    state |= config
    torch.save(
        state, DATA / name, _use_new_zipfile_serialization=zip_format, pickle_protocol=4
    )
    return {t: lm.perplexity(dictionary.item2idx, t) for t in TEXTS}


def main():
    torch.manual_seed(0)
    perplexities = {
        # the format of most published models
        "lm-legacy.pt": save(
            "lm-legacy.pt",
            False,
            shared=True,
            is_forward_lm=True,
            hidden_size=8,
            nlayers=1,
            embedding_size=4,
            nout=None,
        ),
        "lm-zip.pt": save(
            "lm-zip.pt",
            True,
            shared=False,
            is_forward_lm=False,
            hidden_size=8,
            nlayers=2,
            embedding_size=4,
            nout=3,
        ),
    }
    (DATA / "perplexities.json").write_text(json.dumps(perplexities, indent=1) + "\n")


if __name__ == "__main__":
    main()
