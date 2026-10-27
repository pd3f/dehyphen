"""Measure how well dehyphen rejoins words that were hyphenated at the end of a line.

Wraps clean text into short lines, hyphenated with pyphen like a typesetter would,
and compares the output of each scorer with the original words, per kind of line end.

    poetry install --with bench
    poetry run python bench/bench.py --help
"""

import argparse
import difflib
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import pyphen

from dehyphen import FlairScorer, format_to_paragraph, text_to_format
from dehyphen.dehyphen import Scorer, count_words

DATA = Path(__file__).parent / "data"
# hyphens and dashes after which a typesetter may break a word
BREAKS = "-\u2010\u2011\u2013\u2014"

# name: (language, url), downloaded once into `DATA`
CORPORA = {
    "gg": ("de", "https://www.gesetze-im-internet.de/gg/xml.zip"),
    "geg": ("de", "https://www.gesetze-im-internet.de/geg/xml.zip"),
    "bsig": ("de", "https://www.gesetze-im-internet.de/bsig_2025/xml.zip"),
    "pride": ("en", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"),
}


class JoinAll(Scorer):
    """Removes every hyphen at the end of a line, like a regular expression would."""

    def score(self, texts):
        return [len(t) for t in texts]


def download(name):
    url = CORPORA[name][1]
    path = DATA / (name + Path(url).suffix)
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load(name):
    path = download(name)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            xml = next(n for n in z.namelist() if n.endswith(".xml"))
            root = ET.fromstring(z.read(xml))
        # list items and table cells are separate elements without whitespace in between
        texts = [" ".join(p.itertext()) for p in root.iter("P")]
    else:
        text = path.read_text(encoding="utf-8")
        text = text.split("*** START OF")[1].split("*** END OF")[0]
        texts = [p for p in re.split(r"\n\s*\n", text) if "[" not in p]
    # skip headings and other short fragments
    return [p.split() for p in texts if len(p.split()) >= 10]


def split(word, room, hyphenator):
    """Split `word` so that its first part fits into `room`, only after a hyphen or dash if it has one."""
    if any(c in BREAKS for c in word):
        marks = re.escape(BREAKS)
        breaks = re.finditer(rf"(?<=[^\s{marks}])[{marks}]+(?=\w)", word)
        heads = [m for m in breaks if m.end() <= room]
        if not heads:
            return None
        m = heads[-1]
        kind = "inner dash" if m[0] == "\u2014" or len(m[0]) > 1 else "compound"
        return word[: m.end()], word[m.end() :], kind
    parts = hyphenator.wrap(word, room)
    return (*parts, "syllable") if parts else None


def wrap(words, width, hyphenator):
    """Lines of at most `width` characters and the kind of each line end with a hyphen, by word index."""
    lines, line, kinds = [], [], {}
    for i, word in enumerate(words):
        room = width - len(" ".join(line)) - 1 if line else width
        if len(word) <= room:
            line.append(word)
            continue
        parts = split(word, room, hyphenator)
        if parts:
            head, word, kinds[i] = parts
            line.append(head)
        elif line and line[-1][-1] in BREAKS:
            # a word like `Arbeits-` before `und`, or a dash
            dash = len(line[-1]) == 1 or line[-1].endswith(("--", "\u2014"))
            kinds[i - 1] = "dash" if dash else "suspended"
        if line:
            lines.append(line)
        line = [word]
    lines.append(line)
    return "\n".join(" ".join(x) for x in lines), kinds


def wrong_words(original, output):
    """Indices of the original words that the output does not reproduce."""
    wrong = set()
    matcher = difflib.SequenceMatcher(None, original, output, autojunk=False)
    for op, i1, i2, _, _ in matcher.get_opcodes():
        if op != "equal":
            wrong.update(range(i1, max(i2, i1 + 1)))
    return wrong


def run(scorer, cases, words, context):
    """Number of line ends and of correct ones per kind, and the seconds it took."""
    if isinstance(scorer, FlairScorer):
        scorer.score_text.cache_clear()
    total, correct = Counter(), Counter()
    seconds = 0.0
    for original, lines, kinds in cases:
        start = time.perf_counter()
        lines = scorer.dehyphen_paragraph([line[:] for line in lines], words, context)
        seconds += time.perf_counter() - start
        wrong = wrong_words(original, format_to_paragraph(lines).split())
        for i, kind in kinds.items():
            total[kind] += 1
            correct[kind] += i not in wrong
    return total, correct, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("corpus", nargs="*", default=list(CORPORA), choices=CORPORA)
    parser.add_argument("--width", type=int, default=60, help="characters per line")
    parser.add_argument("--model", default="multi-v0", help="language of FlairScorer")
    parser.add_argument("--fast", action="store_true", help="use Flair's fast model")
    parser.add_argument("--context", type=int, default=2, help="words before and after")
    args = parser.parse_args()

    flair = FlairScorer(lang=args.model, fast=args.fast)
    # name: (scorer, whether it knows the words of the whole document)
    scorers = {
        "join-all": (JoinAll(), False),
        "words, else join": (JoinAll(), True),
        "flair": (flair, False),
        "words, else flair": (flair, True),
    }
    model = args.model + (" fast" if args.fast else "")
    for name in args.corpus:
        lang, _ = CORPORA[name]
        hyphenator = pyphen.Pyphen(lang=lang)
        cases = []
        for words in load(name):
            text, kinds = wrap(words, args.width, hyphenator)
            cases.append((words, text_to_format(text)[0], kinds))
        document = count_words([lines for _, lines, _ in cases])
        results = {
            s: run(scorer, cases, document if use_words else None, args.context)
            for s, (scorer, use_words) in scorers.items()
        }

        print(
            f"\n{name} ({lang}), {args.width} characters per line, flair {model}, context {args.context}\n"
        )
        print("| line end | n | " + " | ".join(results) + " |")
        print("|---|---:|" + "---:|" * len(results))
        kinds = sorted(next(iter(results.values()))[0])
        for kind in kinds + ["all"]:
            cells = []
            for total, correct, _ in results.values():
                n = sum(total.values()) if kind == "all" else total[kind]
                ok = sum(correct.values()) if kind == "all" else correct[kind]
                cells.append(f"{ok / n:.1%}")
            print(f"| {kind} | {n} | " + " | ".join(cells) + " |")
        seconds = " | ".join(f"{r[2]:.1f}" for r in results.values())
        print(f"| seconds | | {seconds} |")


if __name__ == "__main__":
    main()
