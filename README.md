# `dehyphen` [![PyPI](https://img.shields.io/pypi/v/dehyphen.svg)](https://pypi.org/project/dehyphen/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dehyphen.svg)](https://pypi.org/project/dehyphen/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/dehyphen)](https://pypistats.org/packages/dehyphen)

*Experimental, use with care.*

Python package for dehyphenation of broken text, e.g., extracted from a PDF. Mainly for the German but works for other languages as well.

`dehyphen` tries to reconstruct the original text by choosing the most probably way to join lines or paragraphs (and remove hyphens).
Several options are getting scored by calculating the [perplexity](https://en.wikipedia.org/wiki/Perplexity#Perplexity_per_word) of text, using [Flair](https://github.com/flairNLP/flair)'s character-based language models.
Based on these scores, the best fitting option is taken to guess the original text.

If you are into text extraction for German PDFs: Stay tuned. I'm gonna release something soon-ish. Follow [@ddd_jetzt](https://twitter.com/ddd_jetzt) on Twitter for updates.

## Installation

```bash
pip install dehyphen
```

## Usage

```python
from dehyphen import FlairScorer

scorer = FlairScorer(lang="de")
```

You need to set `lang` to `de` for German, `en` for English, `es` for Spanish, etc. Otherwise, a multi-language-model will be chosen as the default. [See this section in the source code for more models](https://github.com/flairNLP/flair/blob/8c09e62d9a5a3c227b9ca0fb9f214de9620d4ca0/flair/embeddings/token.py#L431) (but omit the "-backwards" and "-forwards" as specified by Flair).

To speed up computations, choose a `-fast` language model from Flair. However, there are currently only a few. There is for instance a multi-language one named `multi-v0` that contains English, German, French and others. There is non for German.

Using CUDA (with a GPU) dramatically improves performance.

### 1. remove hyphens from the end of a line (within paragraphs)
```python

# returns cleaned paragraph
scorer.dehyphen(special_format)
```

The input text has to be in a special format. Paragraphs should be seperated by two newlines characters (`\n\n`). Line should be end with a single newline `\n`. Several helper functions exists to transform the data into the required format.

### 2. join paragraphs, e.g., to reverse a page break

```python

# returns the joined paragraphs if the language model thinks there were split, otherwise `None`
scorer.is_split_paragraph(paragraph_1, paragraph_2)
```

## Example

```python
from dehyphen import FlairScorer

scorer = FlairScorer(lang="de")

some_german_text = """Zwar wird durch die Einführung eines eigenen Strafgesetzes die Bedeutung der finan-
ziellen Interessen der Union gewiss unterstrichen, dennoch erscheint die Aufspaltung
des strafrechtlichen Vermögensschutzes zweifelhaft, insbesondere soweit es densel-
ben Schutzgegenstand, nämlich die vermögensrelevanten Interessen der Union be-
trifft. Zum einen wird es den Normunterworfenen ohne Not erschwert, die zu befolgen-
den Strafgesetze zu erfassen. Zum anderen ergeben sich potentielle Auslegungsdif-

ferenzen durch die Verwendung teilweise abweichender Terminologie (finanzielle In-
teressen vs. Vermögen). Schließlich wird der Schutz besagter Interessen ohnedies
bislang innerhalb des StGB gewährleistet. Daher empfiehlt es sich u.E., sämtliche Re-
gelungen des RegE in das StGB zu integrieren, soweit entsprechende Neuregelungen
überhaupt erforderlich sind. Hierdurch wird sich auch eine klarere Trennung von Straf-
recht und Verwaltungsrecht erreichen lassen.

Das Erfolgsverständnis entspricht daher eher dem wesentlich weiteren Betrugsbegriff
bspw. des US-amerikanischen Rechts (Federal Law bspw. Fraud, Defraud, Wire-
Fraud, Bank-Fraud, 18.U.S.C. §1341 ff.(2016)) , die teilweise auch ganz auf einen
Schaden verzichten. Fraud erfasst auch viele untreue- und unterschlagungsähnliche
Verhaltensweisen sowie betrügerische Verfügungen als solche. Auch andere EU-
Staaten, wie bspw. Polen , liegen im Hinblick auf den Erfolg näher bei der Richtlinie
als bei der deutschen Schadensdogmatik.
"""

special_format = text_to_format(some_german_text)
fixed_hyphens = scorer.dehyphen(special_format)

# checks if two paragraphs can be joined, useful to, e.g., reverse page breaks.
joined_paragraph = scorer.is_split_paragraph(fixed_hyphens[:2])

print(joined_paragraph)
```
**Output text**:

Zwar wird durch die Einführung eines eigenen Strafgesetzes die Bedeutung der finanziellen Interessen der Union gewiss unterstrichen, dennoch erscheint die Aufspaltung des strafrechtlichen Vermögensschutzes zweifelhaft, insbesondere soweit es denselben Schutzgegenstand, nämlich die vermögensrelevanten Interessen der Union betrifft. Zum einen wird es den Norm unterworfenen ohne Not erschwert, die zubefolgenden Strafgesetze zu erfassen. Zum anderen ergeben sich potentielle **Auslegungsdifferenzen** durch die Verwendung teilweise abweichender Terminologie (finanzielle Interessen vs. Vermögen). Schließlich wird der Schutz besagter Interessen ohnediesbislang innerhalb des StGB gewährleistet. Daher empfiehlt es sich u.E., sämtliche Regelungen des RegE in das StGB zu integrieren, soweit entsprechende Neuregelungenüberhaupt erforderlich sind. Hierdurch wird sich auch eine klarere Trennung von Strafrecht und Verwaltungsrecht erreichen lassen.

*Hyphens are removed, paragraphs are joined along the word **Auslegungsdifferenzen**.*

```python
print(fixed_hyphens[-1])
```
**Output text**:

Das Erfolgsverständnis entspricht daher eher dem wesentlich weiteren Betrugsbegriff bspw. des US-amerikanischen Rechts (Federal Law bspw. Fraud, Defraud, **Wire-Fraud**, Bank-Fraud, 18.U.S.C. §1341 ff.(2016)) , die teilweise auch ganz auf einen Schaden verzichten. Fraud erfasst auch viele untreue- und unterschlagungsähnliche Verhaltensweisen sowie betrügerische Verfügungen als solche. Auch andere **EU-Staaten**, wie bspw. Polen , liegen im Hinblick auf den Erfolg näher bei der Richtlinie als bei der deutschen Schadensdogmatik.Strafrecht und Verwaltungsrecht erreichen lassen.

***EU-Staaten** & **Wire-Fraud** are not dehyphenized.*


## License

GPLv3