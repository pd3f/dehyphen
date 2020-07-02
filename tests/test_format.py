from dehyphen.format import assert_format


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
