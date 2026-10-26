# `dehyphen` [![PyPI](https://img.shields.io/pypi/v/dehyphen.svg)](https://pypi.org/project/dehyphen/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dehyphen.svg)](https://pypi.org/project/dehyphen/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/dehyphen)](https://pypistats.org/packages/dehyphen)

*Experimental, use with care.*

Python package for **dehyphenation of broken text**, i.e., extracted from a PDF. Mainly for German but works for other languages as well.

`dehyphen` tries to reconstruct the original continuous text by choosing the most probable way to join lines or paragraphs (and remove hyphens).
Several options get scored by calculating the [perplexity](https://en.wikipedia.org/wiki/Perplexity#Perplexity_per_word) of the text, using the character-based language models of [Flair](https://github.com/flairNLP/flair).
The option with the lowest perplexity wins.

Check out the PDF text extraction pipeline [pd3f](https://github.com/pd3f/pd3f) that uses `dehyphen` internally.

## An Example

For this input

> die Bedeutung der finan-
>
> ziellen Interessen der Union

`dehyphen` joins the lines and removes the '-'.

> die Bedeutung der **finanziellen** Interessen der Union

But in this example

> Auch andere EU-
>
> Staaten, wie bspw. Polen,

the lines are also joined but the hyphen is kept (because it's part of the word).

> Auch andere **EU-Staaten**, wie bspw. Polen,

## Installation

```bash
pip install dehyphen
```

`dehyphen` requires Python 3.10 or newer and [PyTorch](https://pytorch.org/get-started/locally/).
On x86-64 Linux, `pip` installs PyTorch with CUDA support by default, which is a large download.
If you don't have a GPU, install the CPU build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Usage

```python
from dehyphen import FlairScorer, format_to_text, text_to_format

scorer = FlairScorer(lang="de")
```

Set `lang` to `de` for German, `en` for English, `es` for Spanish etc.
Without `lang`, a multilingual model is used.
All available models are listed in [`dehyphen/flair_lm.py`](dehyphen/flair_lm.py); omit the `-forward` / `-backward` suffix.
Some languages only have a forward model, set `backward=False` for them.

To speed up computations, set `fast=True` to get a smaller model (the ones ending in `-fast`). Only a few languages have one.
`FlairScorer(lang="multi-v0", fast=True)` is a small model that covers English, German, French, Italian, Dutch and Polish.

The models get downloaded on first use to `$FLAIR_CACHE_ROOT` or `~/.flair`, the same location Flair uses.
Set `cache` to choose another directory.
The models are loaded with a restricted unpickler that only accepts the objects contained in Flair's language models.

A GPU dramatically improves performance.
CUDA is used if available, set `device` (e.g. `device="cpu"` or `device="mps"`) to choose explicitly.

### Input format

All functions work on paragraphs, lines and words: a paragraph is a list of lines, a line is a list of words.
Within a paragraph, the last word of every line but the last ends with a single whitespace character (a space or `\n`) that joins it to the next line; all other words have none.
`text_to_format` converts text into a list of paragraphs: paragraphs are separated by an empty line, lines by a newline, which becomes a space.
`format_to_text` joins the paragraphs back into text, separated by an empty line.

```python
text_to_format(
    "die Bedeutung der finan-\nziellen Interessen\n\nAuch andere EU-\nStaaten"
)
# [[['die', 'Bedeutung', 'der', 'finan- '], ['ziellen', 'Interessen']],
#  [['Auch', 'andere', 'EU- '], ['Staaten']]]
```

Passing a plain string instead raises a `TypeError`.

### 1. Remove hyphens from the end of a line (within paragraphs)

```python
# returns the cleaned paragraphs
paragraphs = scorer.dehyphen(text_to_format(text))

# or a single paragraph
paragraph = scorer.dehyphen_paragraph(lines)
```

### 2. Join paragraphs, e.g., to reverse a page break

```python
# returns the joined paragraph if the language model thinks it was split, otherwise `None`
joined = scorer.is_split_paragraph(paragraph_1, paragraph_2)
```

### 3. Score your own options

`score` returns the perplexity for each text, summed over the forward and backward model; lower means more likely.
Give the model some context, single word fragments score poorly.

```python
scorer.score(["finanziellen Interessen", "finan-ziellen Interessen"])
# [4.15, 11.05]: the first option is more likely
```

## Example

```python
from dehyphen import FlairScorer, format_to_text, text_to_format

scorer = FlairScorer(lang="de")

text = """Zwar wird durch die Einführung eines eigenen Strafgesetzes die Bedeutung der finan-
ziellen Interessen der Union gewiss unterstrichen, dennoch erscheint die Aufspaltung
des strafrechtlichen Vermögensschutzes zweifelhaft. Zum anderen ergeben sich potentielle Auslegungsdif-

ferenzen durch die Verwendung teilweise abweichender Terminologie (finanzielle In-
teressen vs. Vermögen).

Auch andere EU-
Staaten, wie bspw. Polen, liegen im Hinblick auf den Erfolg näher bei der Richtlinie."""

paragraphs = scorer.dehyphen(text_to_format(text))

# join the first two paragraphs if they were split, e.g. by a page break
joined = scorer.is_split_paragraph(paragraphs[0], paragraphs[1])
if joined is not None:
    paragraphs = [joined] + paragraphs[2:]

print(format_to_text(paragraphs))
```

**Output text**:

> Zwar wird durch die Einführung eines eigenen Strafgesetzes die Bedeutung der **finanziellen** Interessen der Union gewiss unterstrichen, dennoch erscheint die Aufspaltung des strafrechtlichen Vermögensschutzes zweifelhaft. Zum anderen ergeben sich potentielle **Auslegungsdifferenzen** durch die Verwendung teilweise abweichender Terminologie (finanzielle **Interessen** vs. Vermögen).
>
> Auch andere **EU-Staaten**, wie bspw. Polen, liegen im Hinblick auf den Erfolg näher bei der Richtlinie.

Hyphens are removed, the paragraphs split by the page break are joined along **Auslegungsdifferenzen**, and **EU-Staaten** keeps its hyphen.

## Development

```bash
poetry install
poetry run pytest
```

## License

GPLv3
