import pytest

from dehyphen import FlairScorer, format_to_paragraph, text_to_format


@pytest.fixture()
def flair_scorer():
    # fast model that also includes german
    scorer = FlairScorer(lang="multi-v0", fast=True)
    return scorer


def test_dehyphen(flair_scorer):
    some_german_text = text_to_format(
        """Zwar wird durch die Einführung eines eigenen Strafgesetzes die Bedeutung der finan-
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
    )

    res1 = flair_scorer.dehyphen_paragraph(some_german_text[0])
    assert (
        format_to_paragraph(res1)
        == """Zwar wird durch die Einführung eines eigenen Strafgesetzes die Bedeutung der finanziellen Interessen der Union gewiss unterstrichen, dennoch erscheint die Aufspaltung des strafrechtlichen Vermögensschutzes zweifelhaft, insbesondere soweit es denselben Schutzgegenstand, nämlich die vermögensrelevanten Interessen der Union betrifft. Zum einen wird es den Normunterworfenen ohne Not erschwert, die zu befolgenden Strafgesetze zu erfassen. Zum anderen ergeben sich potentielle Auslegungsdif-"""
    )

    res2 = flair_scorer.dehyphen_paragraph(some_german_text[1])
    assert (
        format_to_paragraph(res2)
        == """ferenzen durch die Verwendung teilweise abweichender Terminologie (finanzielle Interessen vs. Vermögen). Schließlich wird der Schutz besagter Interessen ohnedies bislang innerhalb des StGB gewährleistet. Daher empfiehlt es sich u.E., sämtliche Regelungen des RegE in das StGB zu integrieren, soweit entsprechende Neuregelungen überhaupt erforderlich sind. Hierdurch wird sich auch eine klarere Trennung von Strafrecht und Verwaltungsrecht erreichen lassen."""
    )

    res3 = flair_scorer.dehyphen_paragraph(some_german_text[2])
    assert (
        format_to_paragraph(res3)
        == """Das Erfolgsverständnis entspricht daher eher dem wesentlich weiteren Betrugsbegriff bspw. des US-amerikanischen Rechts (Federal Law bspw. Fraud, Defraud, Wire-Fraud, Bank-Fraud, 18.U.S.C. §1341 ff.(2016)), die teilweise auch ganz auf einen Schaden verzichten. Fraud erfasst auch viele untreue- und unterschlagungsähnliche Verhaltensweisen sowie betrügerische Verfügungen als solche. Auch andere EU-Staaten, wie bspw. Polen, liegen im Hinblick auf den Erfolg näher bei der Richtlinie als bei der deutschen Schadensdogmatik."""
    )

    assert (
        format_to_paragraph(flair_scorer.is_split_paragraph(res1, res2))
        == """Zwar wird durch die Einführung eines eigenen Strafgesetzes die Bedeutung der finanziellen Interessen der Union gewiss unterstrichen, dennoch erscheint die Aufspaltung des strafrechtlichen Vermögensschutzes zweifelhaft, insbesondere soweit es denselben Schutzgegenstand, nämlich die vermögensrelevanten Interessen der Union betrifft. Zum einen wird es den Normunterworfenen ohne Not erschwert, die zu befolgenden Strafgesetze zu erfassen. Zum anderen ergeben sich potentielle Auslegungsdifferenzen durch die Verwendung teilweise abweichender Terminologie (finanzielle Interessen vs. Vermögen). Schließlich wird der Schutz besagter Interessen ohnedies bislang innerhalb des StGB gewährleistet. Daher empfiehlt es sich u.E., sämtliche Regelungen des RegE in das StGB zu integrieren, soweit entsprechende Neuregelungen überhaupt erforderlich sind. Hierdurch wird sich auch eine klarere Trennung von Strafrecht und Verwaltungsrecht erreichen lassen."""
    )


def test_dehyphen_split_relaxed(flair_scorer):
    t1 = """inen gemeinsamen Einkauf, fördern ihre Mitglieder durch Weiterbildung und die Unterhaltung gemeinsam genutzter Räumlichkeiten und medizinischer Geräte. Als Einkaufsgemeinschaft können die Gesundheitsgenossenschaften Rahmenvereinbarungen mit Unternehmen abschließen, die den Mitgliedern zu Sonderkonditionen Produkte und Dienstleistungen anbieten. Da-  """
    t2 = """  zu zählen beispiels"""
    print(text_to_format(t1))
    r1 = flair_scorer.is_split_paragraph(text_to_format(t1), text_to_format(t2))

    assert format_to_paragraph(r1) == t1.strip()[:-1] + t2.strip()


def test_dehyphen_not_available_model():
    with pytest.raises(ValueError):
        FlairScorer(lang="1337", fast=True)

