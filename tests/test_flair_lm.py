import datetime
import json
import pickle
import socket
import sys
import threading
import types
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from dehyphen import flair_lm
from dehyphen.flair_lm import MODELS, load_language_model, model_path, read_checkpoint

# written by PyTorch with `data/make_checkpoints.py`
DATA = Path(__file__).parent / "data"
PERPLEXITIES = json.loads((DATA / "perplexities.json").read_text())


@pytest.mark.parametrize("name", ["lm-legacy.pt", "lm-zip.pt"])
def test_load_flair_checkpoint(tmp_path, monkeypatch, name):
    monkeypatch.setitem(MODELS, "test-forward", (DATA / name).as_uri())

    lm = load_language_model("test-forward", cache=tmp_path)

    assert (tmp_path / "embeddings" / name).is_file()
    assert isinstance(lm.dictionary, flair_lm.Dictionary)
    assert lm.dictionary.get_idx_for_item("b") == 2
    assert lm.dictionary.get_idx_for_item("z") == 0
    for text, perplexity in PERPLEXITIES[name].items():
        assert lm.perplexity(text) == pytest.approx(perplexity, rel=1e-5)


def write_legacy(path, *objects, raw=b""):
    """A checkpoint in PyTorch's legacy format: pickles followed by `raw`."""
    with open(path, "wb") as f:
        for obj in objects:
            pickle.dump(obj, f, protocol=4)
        f.write(raw)


HEADER = [flair_lm._LEGACY_MAGIC, 1001, {"little_endian": True}]


@pytest.mark.parametrize("position", [0, 3])
def test_reject_foreign_globals(tmp_path, position):
    path = tmp_path / "evil.pt"
    objects = [*HEADER, {}, []]
    objects[position] = datetime.date(2020, 1, 1)
    write_legacy(path, *objects)
    with pytest.raises(pickle.UnpicklingError, match="datetime.date is not allowed"):
        read_checkpoint(path)


def test_reject_other_storage_types(tmp_path):
    path = tmp_path / "lm.pt"
    # protocol 0: the global `torch.HalfStorage`, then stop
    write_legacy(path, *HEADER, raw=b"ctorch\nHalfStorage\n.")
    with pytest.raises(
        pickle.UnpicklingError, match="torch.HalfStorage is not allowed"
    ):
        read_checkpoint(path)


def test_reject_other_files(tmp_path):
    path = tmp_path / "lm.pt"
    write_legacy(path, 42)
    with pytest.raises(pickle.UnpicklingError, match="not a PyTorch checkpoint"):
        read_checkpoint(path)


@pytest.mark.parametrize(
    "offset, size, stride",
    [
        (0, (2, 2), (1, 2)),
        (1, (2, 2), (2, 1)),
        (0, (5,), (1,)),
        (-1, (2,), (1,)),
        (0, (2, 2), ()),
        (0, (-1, 4), (4, 1)),
    ],
)
def test_reject_tensors_outside_contiguous_storage(offset, size, stride):
    tensor = flair_lm._Tensor("0", offset, size, stride)
    with pytest.raises(pickle.UnpicklingError, match="only contiguous tensors"):
        tensor.array({"0": np.zeros(4, dtype="<f4")})


def test_reject_tensor_metadata():
    with pytest.raises(pickle.UnpicklingError, match="unsupported tensor metadata"):
        flair_lm._Tensor("0", 0, (1,), (1,), False, OrderedDict(), {"neg": True})


def test_size_one_dimensions_ignore_stride():
    tensor = flair_lm._Tensor("0", 1, (1, 3), (7, 1))
    data = np.arange(4, dtype="<f4")
    assert tensor.array({"0": data}).tolist() == [[1, 2, 3]]


def test_reject_foreign_globals_in_zip(tmp_path):
    path = tmp_path / "evil.pt"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("evil/data.pkl", pickle.dumps({"x": datetime.date(2020, 1, 1)}))
    with pytest.raises(pickle.UnpicklingError, match="datetime.date is not allowed"):
        read_checkpoint(path)


def test_truncated_legacy_checkpoint(tmp_path):
    path = tmp_path / "lm.pt"
    data = (DATA / "lm-legacy.pt").read_bytes()
    path.write_bytes(data[:-4])
    with pytest.raises(EOFError, match="bytes instead of"):
        read_checkpoint(path)


def test_truncated_zip_storage(tmp_path):
    path = tmp_path / "lm.pt"
    with zipfile.ZipFile(DATA / "lm-zip.pt") as src, zipfile.ZipFile(path, "w") as dst:
        for info in src.infolist():
            data = src.read(info)
            dst.writestr(info, data[:-4] if info.filename == "lm-zip/data/1" else data)
    with pytest.raises(EOFError, match="bytes instead of"):
        read_checkpoint(path)


def test_reject_big_endian_zip(tmp_path):
    path = tmp_path / "lm.pt"
    with zipfile.ZipFile(DATA / "lm-zip.pt") as src, zipfile.ZipFile(path, "w") as dst:
        for info in src.infolist():
            data = b"big" if info.filename.endswith("/byteorder") else src.read(info)
            dst.writestr(info, data)
    with pytest.raises(pickle.UnpicklingError, match="little-endian"):
        read_checkpoint(path)


def test_reject_big_endian_legacy(tmp_path):
    path = tmp_path / "lm.pt"
    write_legacy(path, flair_lm._LEGACY_MAGIC, 1001, {"little_endian": False})
    with pytest.raises(pickle.UnpicklingError, match="little-endian"):
        read_checkpoint(path)


def test_zip_without_byteorder_is_little_endian(tmp_path):
    path = tmp_path / "lm.pt"
    with zipfile.ZipFile(DATA / "lm-zip.pt") as src, zipfile.ZipFile(path, "w") as dst:
        for info in src.infolist():
            if not info.filename.endswith("/byteorder"):
                dst.writestr(info, src.read(info))
    expected = read_checkpoint(DATA / "lm-zip.pt")["state_dict"]
    for key, array in read_checkpoint(path)["state_dict"].items():
        assert (array == expected[key]).all()


def test_pickle_cannot_change_classes(tmp_path):
    path = tmp_path / "evil.pt"
    # protocol 0: set the attribute `array` of the global to 1
    evil = b"ctorch._utils\n_rebuild_tensor_v2\n(N}Varray\nI1\nstb."
    write_legacy(path, *HEADER, raw=evil)
    # the file ends before the storage keys
    with pytest.raises(EOFError):
        read_checkpoint(path)
    assert callable(flair_lm._Tensor.array)
    assert read_checkpoint(DATA / "lm-zip.pt")["state_dict"]


def test_reject_zip_without_object(tmp_path):
    path = tmp_path / "lm.pt"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("lm/data/0", b"")
    with pytest.raises(pickle.UnpicklingError, match="no data.pkl"):
        read_checkpoint(path)


STORAGE = object()


class StoragePickler(pickle.Pickler):
    """Pickles `STORAGE` as a reference to `self.pid`, as PyTorch does with storages."""

    def persistent_id(self, obj):
        return self.pid if obj is STORAGE else None


@pytest.fixture()
def float_storage(monkeypatch):
    """A class that pickles as `torch.FloatStorage`."""
    module = types.ModuleType("torch")

    class FloatStorage:
        pass

    FloatStorage.__module__ = "torch"
    FloatStorage.__qualname__ = "FloatStorage"
    module.FloatStorage = FloatStorage
    monkeypatch.setitem(sys.modules, "torch", module)
    return FloatStorage


@pytest.mark.parametrize(
    "kind, dtype, view",
    [("module", "float", ()), ("storage", "dict", ()), ("storage", "float", (None,))],
)
def test_reject_unsupported_storages(tmp_path, float_storage, kind, dtype, view):
    dtype = float_storage if dtype == "float" else OrderedDict
    if view == (None,):
        view = (("1", 0, 1),)
    path = tmp_path / "lm.pt"
    write_legacy(path, *HEADER)
    with open(path, "ab") as f:
        pickler = StoragePickler(f, protocol=4)
        pickler.pid = (kind, dtype, "0", "cpu", 1, *view)
        pickler.dump({"state_dict": {"w": STORAGE}})
    with pytest.raises(pickle.UnpicklingError, match="unsupported storage"):
        read_checkpoint(path)


def test_accept_float_storages(tmp_path, float_storage):
    path = tmp_path / "lm.pt"
    write_legacy(path, *HEADER)
    with open(path, "ab") as f:
        pickler = StoragePickler(f, protocol=4)
        pickler.pid = ("storage", float_storage, "0", "cpu", 2, None)
        pickler.dump({"state_dict": {}, "w": STORAGE})
        pickle.dump(["0"], f, protocol=4)
        f.write((2).to_bytes(8, "little") + bytes(8))
    assert read_checkpoint(path)["w"] == "0"


def test_reject_storage_of_unexpected_size(tmp_path):
    data = bytearray((DATA / "lm-legacy.pt").read_bytes())
    with open(DATA / "lm-legacy.pt", "rb") as f:
        for _ in range(5):  # header, object and storage keys
            flair_lm._Unpickler(f).load()
        start = f.tell()
    n = int.from_bytes(data[start : start + 8], "little")
    data[start : start + 8] = (n + 1).to_bytes(8, "little")
    path = tmp_path / "lm.pt"
    path.write_bytes(data)
    with pytest.raises(pickle.UnpicklingError, match="unexpected size"):
        read_checkpoint(path)


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


def serve_once(response):
    """URL of a server that answers one request with `response` and closes."""
    server = socket.create_server(("127.0.0.1", 0))

    def reply():
        conn, _ = server.accept()
        with conn:
            conn.recv(65536)
            conn.sendall(response)
        server.close()

    threading.Thread(target=reply, daemon=True).start()
    return f"http://127.0.0.1:{server.getsockname()[1]}/lm.pt"


def test_truncated_download_leaves_no_file(tmp_path, monkeypatch):
    url = serve_once(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\ncheckpoint")
    monkeypatch.setitem(MODELS, "test-forward", url)
    with pytest.raises(OSError, match="incomplete download"):
        model_path("test-forward", cache=tmp_path)
    assert list((tmp_path / "embeddings").iterdir()) == []


def test_stalled_download_times_out(tmp_path, monkeypatch):
    with socket.create_server(("127.0.0.1", 0)) as server:
        url = f"http://127.0.0.1:{server.getsockname()[1]}/lm.pt"
        monkeypatch.setitem(MODELS, "test-forward", url)
        monkeypatch.setattr(flair_lm, "DOWNLOAD_TIMEOUT", 0.1)
        with pytest.raises(OSError, match="timed out"):
            model_path("test-forward", cache=tmp_path)
    assert list((tmp_path / "embeddings").iterdir()) == []


def corrupt_zip_member():
    """`lm-zip.pt` with one byte of a storage flipped, so its CRC fails."""
    data = bytearray((DATA / "lm-zip.pt").read_bytes())
    with zipfile.ZipFile(DATA / "lm-zip.pt") as z:
        info = z.getinfo("lm-zip/data/1")
    header = info.header_offset
    name_len = int.from_bytes(data[header + 26 : header + 28], "little")
    extra_len = int.from_bytes(data[header + 28 : header + 30], "little")
    data[header + 30 + name_len + extra_len] ^= 0xFF
    return bytes(data)


def legacy_bytes(*objects):
    return b"".join(pickle.dumps(obj, protocol=4) for obj in objects)


@pytest.mark.parametrize(
    "content",
    [
        b"<html>blocked by proxy</html>",
        corrupt_zip_member(),
        legacy_bytes(flair_lm._LEGACY_MAGIC, 1001, ["little_endian"]),
        (DATA / "lm-legacy.pt").read_bytes().replace(b"encoder.", b"fncoder."),
    ],
)
def test_broken_cached_model_names_path(tmp_path, monkeypatch, content):
    monkeypatch.setitem(MODELS, "test-forward", "https://example.org/lm.pt")
    cached = tmp_path / "embeddings" / "lm.pt"
    cached.parent.mkdir()
    cached.write_bytes(content)
    with pytest.raises(RuntimeError, match="delete it to download it again"):
        load_language_model("test-forward", cache=tmp_path)


def test_unknown_model():
    with pytest.raises(ValueError, match="unknown Flair language model"):
        model_path("1337-forward")
