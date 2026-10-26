import datetime
import pickle
import sys
import types

import pytest
import torch

from dehyphen import flair_lm
from dehyphen.flair_lm import MODELS, LanguageModel, load_language_model, model_path


@pytest.fixture()
def flair_dictionary_class(monkeypatch):
    """A class that pickles as `flair.data.Dictionary`, like the published checkpoints."""
    module = types.ModuleType("flair.data")

    class Dictionary:
        pass

    Dictionary.__module__ = "flair.data"
    Dictionary.__qualname__ = "Dictionary"
    module.Dictionary = Dictionary
    monkeypatch.setitem(sys.modules, "flair", types.ModuleType("flair"))
    monkeypatch.setitem(sys.modules, "flair.data", module)
    return Dictionary


def save_checkpoint(path, dictionary_class, nout):
    dictionary = dictionary_class()
    dictionary.idx2item = [b"<unk>", b"a", b"b", b" "]
    dictionary.item2idx = {c: i for i, c in enumerate(dictionary.idx2item)}
    lm = LanguageModel(
        dictionary, True, hidden_size=8, nlayers=1, embedding_size=4, nout=nout
    )
    state = {
        "dictionary": dictionary,
        "state_dict": lm.state_dict(),
        "is_forward_lm": True,
        "hidden_size": 8,
        "nlayers": 1,
        "embedding_size": 4,
        "nout": nout,
        "dropout": 0.1,
    }
    # the published models use torch's legacy (non-zip) format and pickle protocol 4
    torch.save(state, path, _use_new_zipfile_serialization=False, pickle_protocol=4)
    return lm


@pytest.mark.parametrize("nout", [None, 3])
def test_load_flair_checkpoint(tmp_path, monkeypatch, flair_dictionary_class, nout):
    source = tmp_path / "lm-test.pt"
    original = save_checkpoint(source, flair_dictionary_class, nout)
    monkeypatch.setitem(MODELS, "test-forward", source.as_uri())

    lm = load_language_model("test-forward", cache=tmp_path / "cache", device="cpu")

    assert (tmp_path / "cache" / "embeddings" / "lm-test.pt").is_file()
    assert isinstance(lm.dictionary, flair_lm.Dictionary)
    assert lm.dictionary.get_idx_for_item("b") == 2
    assert lm.dictionary.get_idx_for_item("z") == 0
    original.dictionary = lm.dictionary
    assert lm.perplexity("ab ba") == pytest.approx(original.eval().perplexity("ab ba"))


def test_reject_foreign_globals(tmp_path):
    path = tmp_path / "evil.pt"
    torch.save({"dictionary": datetime.date(2020, 1, 1)}, path)
    with pytest.raises(pickle.UnpicklingError, match="datetime.date is not allowed"):
        torch.load(path, pickle_module=flair_lm._SafePickle, weights_only=False)


def test_model_path_uses_cache_without_download(tmp_path):
    cached = tmp_path / "embeddings" / "lm-mix-german-forward-v0.2rc.pt"
    cached.parent.mkdir()
    cached.touch()
    assert model_path("de-forward", cache=tmp_path) == cached
    assert model_path("DE-forward", cache=tmp_path) == cached


def test_model_path_separates_generic_file_names(tmp_path):
    for name in ("uk-forward", "am-forward"):
        cached = tmp_path / "embeddings" / name / "best-lm.pt"
        cached.parent.mkdir(parents=True)
        cached.touch()
        assert model_path(name, cache=tmp_path) == cached


def test_failed_download_leaves_no_file(tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint")
    monkeypatch.setitem(MODELS, "test-forward", source.as_uri())

    def broken_copy(src, dst):
        dst.write(b"check")
        raise OSError("connection reset")

    monkeypatch.setattr(flair_lm.shutil, "copyfileobj", broken_copy)
    with pytest.raises(OSError, match="connection reset"):
        model_path("test-forward", cache=tmp_path / "cache")
    assert list((tmp_path / "cache" / "embeddings").iterdir()) == []


def test_unknown_model():
    with pytest.raises(ValueError, match="unknown Flair language model"):
        model_path("1337-forward")
