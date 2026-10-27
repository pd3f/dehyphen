# Benchmark

Measures how well dehyphen rejoins words that were hyphenated at the end of a line.
`bench.py` takes clean text, wraps it into lines of 60 characters, hyphenated with [pyphen](https://pyphen.org/) like a typesetter would, runs each scorer over it and compares the output with the original words.

## Corpora

| name | text | language |
|---|---|---|
| `gg` | Grundgesetz, from gesetze-im-internet.de | German |
| `geg` | Gebäudeenergiegesetz, from gesetze-im-internet.de | German |
| `bsig` | BSI-Gesetz, from gesetze-im-internet.de | German |
| `pride` | Pride and Prejudice, from Project Gutenberg | English |

Each corpus is downloaded once into `bench/data/`. Paragraphs with fewer than ten words, like headings, are skipped.

## Line ends

A word that already contains a hyphen or dash is only broken there; every other word is broken where pyphen allows it.

| kind | example | correct |
|---|---|---|
| syllable | `Gesetz-` / `gebung` | `Gesetzgebung` |
| compound | `Bund-` / `Länder-Kommission` | `Bund-Länder-Kommission` |
| suspended | `Arbeits-` / `und Sozialrecht` | `Arbeits- und Sozialrecht` |
| dash | `–` / `soweit` | `– soweit` |
| inner dash | `therefore--` / `I` | `therefore--I` |

## Scorers

| scorer | decides by |
|---|---|
| join-all | removes every hyphen, like a regular expression would |
| words, else join | the words of the whole document, else removes the hyphen |
| flair | Flair's language models alone |
| words, else flair | the words of the whole document, else Flair; this is what `Scorer.dehyphen` does |

All scorers share dehyphen's fixed rules, so a dash glued to a word is always joined without a space.

## Running

```bash
poetry install --with bench
poetry run python bench/bench.py                         # all corpora, multi-v0 model
poetry run python bench/bench.py pride gg --fast         # Flair's fast model
poetry run python bench/bench.py gg geg bsig --model de  # the German model
```

`--width` sets the characters per line, `--context` the words before and after a line end.

## Results

Measured on 2026-10-27 with 60 characters per line and two words of context; share of correct line ends.

Flair `multi-v0`, fast model:

| corpus | line ends | join-all | words, else join | flair | words, else flair |
|---|---:|---:|---:|---:|---:|
| `pride` | 1931 | 98.0% | 99.0% | 99.4% | 99.7% |
| `gg` | 1132 | 99.6% | 99.6% | 100.0% | 100.0% |
| `geg` | 1443 | 95.9% | 98.1% | 99.6% | 99.9% |
| `bsig` | 1306 | 94.8% | 98.7% | 99.9% | 99.9% |

Flair `de`:

| corpus | line ends | join-all | words, else join | flair | words, else flair |
|---|---:|---:|---:|---:|---:|
| `gg` | 1132 | 99.6% | 99.6% | 99.7% | 99.9% |
| `geg` | 1443 | 95.9% | 98.1% | 99.5% | 99.9% |
| `bsig` | 1306 | 94.8% | 98.7% | 100.0% | 100.0% |

The words of the document settle most line ends without the language model, which makes "words, else flair" 4 to 9 times faster than Flair alone.
The German model is no more accurate than the fast multilingual one, but about seven times slower.
English compounds are the hardest kind: `pride` gets 35 of 39 right.

The hyphenation is synthetic: pyphen's patterns, clean text, no OCR errors, no page headers between the lines.
Real documents score lower.
