import pytest

from dehyphen import FlairScorer
from dehyphen.dehyphen import Scorer
from dehyphen.format import assert_format, text_to_format


def test_reject_plain_text():
    with pytest.raises(TypeError, match="text_to_format"):
        Scorer().dehyphen("Bundes-\nminister")
    with pytest.raises(ValueError, match="non-empty list of words"):
        Scorer().dehyphen_paragraph(["Bundes-", "minister"])


def test_reject_single_character_option():
    with pytest.raises(ValueError, match="two characters"):
        FlairScorer(forward=False, backward=False).score(["a", "bc"])


def test_format_check():
    class test_clas:
        def __init__(self, data):
            self.data = data

        # make it iterable
        def __getitem__(self, key):
            return self.data[key]

    t = test_clas([["w1", "w2"], ["a1", "b2"]])
    for x in t:
        print(x)
    assert_format(t)


def test_text_to_format():
    t1 = """inen gemeinsamen Einkauf, fördern ihre Mitglieder durch Weiterbildung und die Unterhaltung gemeinsam genutzter Räumlichkeiten und medizinischer Geräte. Als Einkaufsgemeinschaft können die Gesundheitsgenossenschaften Rahmenvereinbarungen mit Unternehmen abschließen, die den Mitgliedern zu Sonderkonditionen Produkte und Dienstleistungen anbieten. Da-  """
    t1f = text_to_format(t1)
    assert len(t1f) == 1
    assert_format(t1f[0])
    assert text_to_format("eine Zeile") == [[["eine", "Zeile"]]]


def test_whitespace_lines_separate_paragraphs():
    assert text_to_format("ein Ab-\nsatz\n \t\nnoch einer\n\n\n") == [
        [["ein", "Ab- "], ["satz"]],
        [["noch", "einer"]],
    ]
