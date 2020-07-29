import pytest

from dehyphen import FlairScorer, format_to_paragraph, text_to_format


@pytest.fixture()
def setup_text():
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
Fraud, Bank-Fraud, 18.U.S.C. §1341 ff.(2016)), die teilweise auch ganz auf einen
Schaden verzichten. Fraud erfasst auch viele untreue- und unterschlagungsähnliche
Verhaltensweisen sowie betrügerische Verfügungen als solche. Auch andere EU-
Staaten, wie bspw. Polen, liegen im Hinblick auf den Erfolg näher bei der Richtlinie
als bei der deutschen Schadensdogmatik."""
    return text_to_format(some_german_text)


def test_dehyphen(setup_text):
    scorer = FlairScorer(lang="de")
    res1 = scorer.dehyphen_paragraph(setup_text[0])
    assert (
        format_to_paragraph(res1)
        == """Zwar wird durch die Einführung eines eigenen Strafgesetzes die Bedeutung der finanziellen Interessen der Union gewiss unterstrichen, dennoch erscheint die Aufspaltung des strafrechtlichen Vermögensschutzes zweifelhaft, insbesondere soweit es denselben Schutzgegenstand, nämlich die vermögensrelevanten Interessen der Union betrifft. Zum einen wird es den Normunterworfenen ohne Not erschwert, die zu befolgenden Strafgesetze zu erfassen. Zum anderen ergeben sich potentielle Auslegungsdif-"""
    )

    res2 = scorer.dehyphen_paragraph(setup_text[1])
    assert (
        format_to_paragraph(res2)
        == """ferenzen durch die Verwendung teilweise abweichender Terminologie (finanzielle Interessen vs. Vermögen). Schließlich wird der Schutz besagter Interessen ohnedies bislang innerhalb des StGB gewährleistet. Daher empfiehlt es sich u.E., sämtliche Regelungen des RegE in das StGB zu integrieren, soweit entsprechende Neuregelungen überhaupt erforderlich sind. Hierdurch wird sich auch eine klarere Trennung von Strafrecht und Verwaltungsrecht erreichen lassen."""
    )

    res3 = scorer.dehyphen_paragraph(setup_text[2])
    assert (
        format_to_paragraph(res3)
        == """Das Erfolgsverständnis entspricht daher eher dem wesentlich weiteren Betrugsbegriff bspw. des US-amerikanischen Rechts (Federal Law bspw. Fraud, Defraud, Wire-Fraud, Bank-Fraud, 18.U.S.C. §1341 ff.(2016)), die teilweise auch ganz auf einen Schaden verzichten. Fraud erfasst auch viele untreue- und unterschlagungsähnliche Verhaltensweisen sowie betrügerische Verfügungen als solche. Auch andere EU-Staaten, wie bspw. Polen, liegen im Hinblick auf den Erfolg näher bei der Richtlinie als bei der deutschen Schadensdogmatik."""
    )

    assert (
        format_to_paragraph(scorer.is_split_paragraph(res1, res2))
        == """Zwar wird durch die Einführung eines eigenen Strafgesetzes die Bedeutung der finanziellen Interessen der Union gewiss unterstrichen, dennoch erscheint die Aufspaltung des strafrechtlichen Vermögensschutzes zweifelhaft, insbesondere soweit es denselben Schutzgegenstand, nämlich die vermögensrelevanten Interessen der Union betrifft. Zum einen wird es den Normunterworfenen ohne Not erschwert, die zu befolgenden Strafgesetze zu erfassen. Zum anderen ergeben sich potentielle Auslegungsdifferenzen durch die Verwendung teilweise abweichender Terminologie (finanzielle Interessen vs. Vermögen). Schließlich wird der Schutz besagter Interessen ohnedies bislang innerhalb des StGB gewährleistet. Daher empfiehlt es sich u.E., sämtliche Regelungen des RegE in das StGB zu integrieren, soweit entsprechende Neuregelungen überhaupt erforderlich sind. Hierdurch wird sich auch eine klarere Trennung von Strafrecht und Verwaltungsrecht erreichen lassen."""
    )
