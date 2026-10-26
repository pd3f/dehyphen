def assert_format(lines):
    # special format. 2D List, word should not contain whitespace, except the last words per line
    if isinstance(lines, str):
        raise TypeError(
            "expected lines of words, convert text with `text_to_format` first"
        )
    for l in lines:
        if isinstance(l, str) or len(l) == 0:
            raise ValueError(f"a line has to be a non-empty list of words: {l!r}")
        for i, word in enumerate(l):
            if not isinstance(word, str):
                raise TypeError(f"a word has to be a string: {word!r}")
            if i == len(l) - 1:
                # only one trailing whitespace character is allowed
                if len(word) - len(word.rstrip()) > 1:
                    raise ValueError(
                        f"last word of a line has more than one trailing whitespace: {word!r}"
                    )
            elif word != word.strip():
                raise ValueError(f"word contains whitespace: {word!r}")


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
