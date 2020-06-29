# dehyphen

*Experimental, use with care.*

Dehyphenation of broken text, e.g., extracted from a PDF.

Reconstruction the orginal text by choosing among different options. Done by calculating the [perplexity](https://en.wikipedia.org/wiki/Perplexity#Perplexity_per_word) of texts, using [flair](https://github.com/flairNLP/flair)'s language models.


If you are into text extraction out of German PDFs: Stay tuned. I'm gonna relase something very soon. Follow [@ddd_jetzt](https://twitter.com/ddd_jetzt) on Twitter.

## Installation

```bash
pip install dehyphen
```

## Usage

You need to set `lang` to `de` for German, `en` for English, `es` for Spanish, etc. Otherwise a standard mutlilanguage-model will be choosen. [See this section in the source code for more models](https://github.com/flairNLP/flair/blob/8c09e62d9a5a3c227b9ca0fb9f214de9620d4ca0/flair/embeddings/token.py#L431), but omit the -backwards and -forwards

The input text has to be in a special format. Paragraphs should be seperated by two newlines `\n` characters. Line should be end with a single `n`.

```python
from dehyphen import *

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
fixed = dehyphen(special_format, lang="de") # you may pass a `lang` argument
# removes all trailing `
print(fixed)
# Zwar wird durch die Einführung eines eigenen Strafgesetzes die Bedeutung der
# finanziellen Interessen der Union gewiss unterstrichen, dennoch erscheint die
# Aufspaltung des strafrechtlichen Vermögensschutzes zweifelhaft, insbesondere soweit es
# denselben Schutzgegenstand, nämlich die vermögensrelevanten Interessen der Union
# betrifft. Zum einen wird es den Normunterworfenen ohne Not erschwert, die zu
# befolgenden Strafgesetze zu erfassen. Zum anderen ergeben sich potentielle Auslegungsdif-
#
# ferenzen durch die Verwendung teilweise abweichender Terminologie (finanzielle
# Interessen vs. Vermögen). Schließlich wird der Schutz besagter Interessen ohnedies
# bislang innerhalb des StGB gewährleistet. Daher empfiehlt es sich u.E., sämtliche
# Regelungen des RegE in das StGB zu integrieren, soweit entsprechende Neuregelungen
# überhaupt erforderlich sind. Hierdurch wird sich auch eine klarere Trennung von
# Strafrecht und Verwaltungsrecht erreichen lassen.
#
# Zwar wird durch die Einführung eines eigenen Strafgesetzes die Bedeutung der
# finanziellen Interessen der Union gewiss unterstrichen, dennoch erscheint die
# Aufspaltung des strafrechtlichen Vermögensschutzes zweifelhaft, insbesondere
# soweit es denselben Schutzgegenstand, nämlich die vermögensrelevanten Interessen
# der Union betrifft. Zum einen wird es den Normunterworfenen ohne Not erschwert,
# die zu befolgenden Strafgesetze zu erfassen. Zum anderen ergeben sich potentielle
# Auslegungsdifferenzen durch die Verwendung teilweise abweichender Terminologie
# (finanzielle Interessen vs. Vermögen). Schließlich wird der Schutz besagter
# Interessen ohnedies bislang innerhalb des StGB gewährleistet. Daher empfiehlt es
# sich u.E., sämtliche Regelungen des RegE in das StGB zu integrieren, soweit
# entsprechende Neuregelungen überhaupt erforderlich sind. Hierdurch wird sich auch
# eine klarere Trennung von Strafrecht und Verwaltungsrecht erreichen lassen.

# NB: EU-Staaten & Wire-Fraud **are** not dehyphenized

# checks if two paragraphs can be joined, useful to, e.g., reverse page breaks.
join_paragraphs_if_cool(fixed[:2], lang="de")
```

## License

MIT.