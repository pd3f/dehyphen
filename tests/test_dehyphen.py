import pytest

from dehyphen import FlairScorer, format_to_paragraph, text_to_format
from dehyphen.dehyphen import Scorer
from dehyphen.format import assert_format


class PreferScorer(Scorer):
    """Scores the options that contain `preferred` as most likely."""

    def __init__(self, preferred):
        self.preferred = preferred

    def score(self, texts):
        return [0 if self.preferred in t else 1 for t in texts]


class ShortestScorer(Scorer):
    """Scores the shortest option as most likely, which removes a hyphen whenever possible."""

    def score(self, texts):
        return [len(t) for t in texts]


@pytest.fixture()
def flair_scorer():
    # fast (small) model that also includes German
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
    r1 = flair_scorer.is_split_paragraph(text_to_format(t1)[0], text_to_format(t2)[0])

    assert format_to_paragraph(r1) == t1.strip()[:-1] + t2.strip()


def test_dehyphen_not_available_model():
    with pytest.raises(ValueError):
        FlairScorer(lang="1337", fast=True)


def test_strange_error(flair_scorer):
    lines = [
        ["Abstract\n"],
        ["classiﬁcation(Remusetal.,2019)andusethepre- "],
        ["deﬁnedsetoflabelstoevaluateourapproachto\n"],
        ["Inthispaper,wefocusontheclassiﬁcation", "thisclassiﬁcationtask", "1", ". "],
        ["ofbooksusingshortdescriptivetexts(cover\n"],
        ["blurbs)andadditionalmetadata.Building\n"],
    ]
    flair_scorer.dehyphen_paragraph(lines)


def test_multiple_lines_into_one(flair_scorer):

    lines = [
        [
            "logy.com/library/detail.aspx?g=56bce942-af1f-4f78-943e-c2eb54b75e10&utm_source=Le- "
        ],
        ["xology+Daily+Newsfeed&utm_medium=HTML+email+-+Body+- "],
        [
            "+General+section&utm_campaign=Lexology+subscriber+daily+feed&utm_content=Lexo- "
        ],
        ["logy+Daily+Newsfeed+2018-08-22&utm_term).\n"],
    ]

    lines_out = flair_scorer.dehyphen_paragraph(lines)
    for x in lines_out:
        assert len(x) > 0
    assert [
        [
            "logy.com/library/detail.aspx?g=56bce942-af1f-4f78-943e-c2eb54b75e10&utm_source=Lexology+Daily+Newsfeed&utm_medium=HTML+email+-+Body++General+section&utm_campaign=Lexology+subscriber+daily+feed&utm_content=Lexology+Daily+Newsfeed+2018-08-22&utm_term).\n"
        ]
    ] == lines_out


def test_soft_hyphen():
    lines = text_to_format("die finan\u00ad\nziellen Interessen")[0]
    res = PreferScorer("finanziellen").dehyphen_paragraph(lines)
    assert format_to_paragraph(res) == "die finanziellen Interessen"


def test_soft_hyphen_of_compound_word_becomes_visible():
    lines = text_to_format("andere EU\u00ad\nStaaten")[0]
    res = PreferScorer("EU-Staaten").dehyphen_paragraph(lines)
    assert format_to_paragraph(res) == "andere EU-Staaten"


@pytest.mark.parametrize("dash", ["-", "\u2013"])
def test_keep_lone_dash(dash):
    lines = text_to_format(f"nach Hause {dash}\nund ging")[0]
    res = ShortestScorer().dehyphen_paragraph(lines)
    assert format_to_paragraph(res) == f"nach Hause {dash} und ging"


def test_split_paragraph_with_trailing_whitespace():
    para1 = [["sämtliche", "Re- "]]
    para2 = [["gelungen", "des", "RegE"]]
    res = PreferScorer("Regelungen").is_split_paragraph(para1, para2)
    assert format_to_paragraph(res) == "sämtliche Regelungen des RegE"


def test_split_paragraph_of_one_word():
    para1 = text_to_format("sämtliche Re-")[0]
    res = PreferScorer("Regelungen").is_split_paragraph(para1, [["gelungen."]])
    assert format_to_paragraph(res) == "sämtliche Regelungen."


def test_split_paragraph_keeps_format():
    para1 = text_to_format("sämtliche Re-")[0]
    para2 = text_to_format("gelungen\ndes RegE")[0]
    res = PreferScorer("Regelungen").is_split_paragraph(para1, para2)
    assert_format(res)
    assert format_to_paragraph(res) == "sämtliche Regelungen des RegE"


def test_split_paragraph_compound_word():
    para1 = text_to_format("andere EU-")[0]
    para2 = text_to_format("Staaten, wie")[0]
    res = PreferScorer("EU-Staaten").is_split_paragraph(para1, para2)
    assert format_to_paragraph(res) == "andere EU-Staaten, wie"


def test_split_paragraph_without_hyphen():
    para1 = [["der", "Satz", "geht\n"]]
    para2 = [["weiter."]]
    assert PreferScorer("\n\n").is_split_paragraph(para1, para2) is None
    res = PreferScorer("geht weiter").is_split_paragraph(para1, para2)
    assert format_to_paragraph(res) == "der Satz geht weiter."
