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

`dehyphen` requires Python 3.10 or newer.

## Usage

```python
from dehyphen import FlairScorer, count_words, format_to_text, text_to_format

scorer = FlairScorer(lang="de")
```

Set `lang` to `de` for German, `en` for English, `es` for Spanish etc.
Without `lang`, a multilingual model is used.
All available models are listed in [`dehyphen/flair_lm.py`](dehyphen/flair_lm.py); omit the `-forward` / `-backward` suffix.
Some languages only have a forward model, set `backward=False` for them.

To speed up computations, set `fast=True` to get a smaller model (the ones ending in `-fast`). Only a few languages have one.
`FlairScorer(lang="multi-v0", fast=True)` is a small model that covers English, German, French, Italian, Dutch and Polish.

The models get downloaded on first use to `$FLAIR_CACHE_ROOT` or `~/.flair`, the directory Flair uses, so most models Flair already downloaded get reused.
Set `cache` to choose another directory.
The models are loaded with a restricted unpickler that only accepts the objects contained in Flair's language models.
They run on the CPU, with numpy.

### Input format

All functions work on paragraphs, lines and words: a paragraph is a list of lines, a line is a list of words.
Within a paragraph, the last word of every line but the last ends with a single whitespace character (a space or `\n`) that joins it to the next line; the last line may end with `\n` as well, all other words have none.
`text_to_format` converts text into a list of paragraphs: paragraphs are separated by empty lines (or lines of only whitespace), lines by a newline, which becomes a space.
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
paragraphs = text_to_format(text)

# returns the cleaned paragraphs
cleaned = scorer.dehyphen(paragraphs)

# or a single paragraph, knowing the words of all paragraphs
lines = scorer.dehyphen_paragraph(paragraphs[0], words=count_words(paragraphs))
```

A line can end in `-`, a soft hyphen (U+00AD), `‐`, `‑`, `–` or `−`. A dash standing alone, as in `nach Hause –`, is kept.
A dash ending a word, as in `word--` or `word—`, is joined with the next word without a space.

The document decides first: if its other lines contain the word only without the hyphen (`finanziellen`) or only with it (`EU-Staaten`), that spelling wins.
The language model is asked only for the remaining line ends, which also makes it faster.
It sees two words before and after the line end, set `context` to change that.

### 2. Join paragraphs, e.g., to reverse a page break

```python
# returns the joined paragraph if the language model thinks it was split, otherwise `None`
joined = scorer.is_split_paragraph(paragraph_1, paragraph_2)
```

A first paragraph whose last word ends with `\n` is complete and never joined.

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

## Changes in 0.6.0

- PyTorch is no longer a dependency: dehyphen reads Flair's checkpoints itself and runs the language models with numpy, on the CPU. The perplexities match PyTorch's to a relative 10⁻⁵.
- `FlairScorer` has no `device` parameter. The models in `FlairScorer.lms` are plain objects with `perplexity`, not PyTorch modules.

## Changes in 0.5.1

- `is_split_paragraph` never joins a first paragraph that ends with `\n`; 0.5.0 could join it and lose the line break.

## Changes in 0.5.0

- `dehyphen` and, with `words=count_words(paragraphs)`, `dehyphen_paragraph` decide with the words of the whole document before asking the language model.
- The language model sees two words before and after the line end instead of only the broken word.
- A dash ending a word (`word--`, `word—`) is joined with the next word without a space.
- Soft hyphens (U+00AD) at the end of a line are removed; a hyphen that is kept becomes a visible `-`.
- A dash standing alone at the end of a line is kept instead of removed.
- A word continued by a single character on the next line is no longer skipped.
- `is_split_paragraph` keeps the hyphen of a compound word (`EU-` + `Staaten`), sees a hyphen followed by whitespace, joins a paragraph of a single word and returns no empty lines.
- clean-text is no longer a dependency.

## Changes in 0.4.0

- Flair is no longer a dependency: dehyphen loads Flair's language models itself and works with current PyTorch. `FlairScorer.lms` holds these models, their `calculate_perplexity` is now `perplexity`.
- `cache` sets the model directory of this scorer instead of Flair's global cache.
- Malformed input raises a `TypeError` or `ValueError` instead of an `AssertionError`.
- `text_to_format` always returns a list of paragraphs, also for text without an empty line, and drops empty paragraphs.

## Development

```bash
poetry install
poetry run pytest
```

[`bench/`](bench/) measures how well dehyphen rejoins words that were hyphenated at the end of a line.

## License

GPLv3
