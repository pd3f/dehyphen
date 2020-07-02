def assert_format(lines):
    # special format. 2D List, word should not contain whitespace, except the last words per line
    for l in lines:
        for i, word in enumerate(l):
            assert isinstance(word, str)
            if i == len(l) - 1:
                # only one trailing whitespace character is allowed
                assert len(word) - len(word.rstrip()) <= 1
            else:
                assert word == word.strip()


def paragraph_to_format(paragraph):
    lines = [l.split() for l in paragraph]
    for l in lines[:-1]:
        l[-1] += " "
    return lines


def text_to_format(text):
    lines = text.splitlines()
    paragraphs = split_list(lines, "")
    return list(map(paragraph_to_format, paragraphs))


def format_to_paragraph(lines):
    return "".join([" ".join(line) for line in lines])


def format_to_text(paragraphs):
    res = ""
    for lines in paragraphs:
        res += format_to_paragraph(lines) + "\n\n"
    return res


# utils


def split_list(input_list, sep):
    res = []
    for x in input_list:
        if x == sep:
            yield res
            res = []
        else:
            res.append(x)
    yield res
