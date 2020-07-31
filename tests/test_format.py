from dehyphen.format import assert_format, text_to_format


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
    assert_format(t1f)

