from collections import Counter
from itertools import chain, islice

from .format import assert_format

# characters that hyphenate a word at the end of a line
HYPHENS = "-\u00ad\u2010\u2011\u2013\u2212"
# dashes that end a word without hyphenating it
DASHES = ("--", "\u2014")
SOFT_HYPHEN = "\u00ad"
PUNCTUATION = ".,;:!?()[]{}\"'„“”‚‘’«»…"


def ends_with_whitespace(s):
    return len(s) != len(s.rstrip())


def hyphen_of(word):
    """The hyphen or dash `word` ends with as printed, `None` if there is none or the dash stands alone like `–`."""
    word = word.rstrip()
    marks = HYPHENS + "\u2014"
    if not word.rstrip(marks) or word[-1] not in marks:
        return None
    # a soft hyphen only shows at the end of a line
    return "-" if word[-1] == SOFT_HYPHEN else word[-1]


def is_dash(word):
    """Whether `word` ends with a dash that follows text directly, as in `word--` or `word—`."""
    return hyphen_of(word) is not None and word.rstrip().endswith(DASHES)


def normalize(word):
    word = word.strip().strip(PUNCTUATION).casefold()
    return word.translate({ord(c): "-" for c in HYPHENS})


def count_words(paragraphs):
    """How often each word occurs in the paragraphs, ignoring case, punctuation and kinds of hyphens.

    The parts of a word hyphenated at the end of a line do not count.
    """
    words = Counter()
    for lines in paragraphs:
        assert_format(lines)
        hyphenated = False
        for line in lines:
            start = int(hyphenated)
            hyphenated = hyphen_of(line[-1]) is not None and line[-1][-1] != "\n"
            words.update(normalize(w) for w in line[start : len(line) - hyphenated])
    return words


def context_of(lines, idx, n):
    """The `n` words before the last word of line `idx` and after the first word of the next line."""
    back = chain.from_iterable(reversed(lines[i]) for i in range(idx, -1, -1))
    ahead = chain.from_iterable(lines[i] for i in range(idx + 1, len(lines)))
    before = list(islice(back, 1, n + 1))[::-1]
    after = islice(ahead, 1, n + 1)
    return " ".join(w.strip() for w in before), " ".join(w.strip() for w in after)


class Scorer:
    def dehyphen(self, paragraphs):
        paragraphs = list(paragraphs)
        words = count_words(paragraphs)
        return [self.dehyphen_paragraph(p, words) for p in paragraphs]

    def dehyphen_paragraph(self, lines, words=None, context=2):
        """dehyphenation of paragraph (lines)

        Lines have to be in special format. 2D List, word should not contain whitespace, except the last words per line
        `words`: `count_words` of the document; if it contains the word only with or only without the hyphen,
        that spelling wins without asking the language model
        `context`: words before and after the line end that the language model sees
        """
        assert_format(lines)

        # Iterate over all lines, change the next line if necessary. This ensure the updates version will get used in the next loop.
        # Make sure to access the lines with the `idx`.
        for idx in range(len(lines) - 1):
            # don't work on last line

            last_word = lines[idx][-1]

            # don't work if there has to be a newline
            if last_word[-1] == "\n":
                continue

            hyphen = hyphen_of(last_word)
            if hyphen is None:
                continue

            next_word = lines[idx + 1][0]
            dash = is_dash(last_word)
            stem = last_word.strip()[:-1]
            # a soft hyphen shows as `-` at the end of a line
            last_word = lines[idx][-1] = stem + hyphen + last_word[len(stem) + 1 :]

            # 1. two words (e.g. part of an abrev.), so do nothing
            option1 = last_word + next_word

            # 2. some compound-word (keep hyphen), remove whitespace
            option2 = stem + hyphen + next_word

            # 3. remove hyphen, most likely to happen
            option3 = stem + next_word

            kept, joined = (
                bool(words and words[normalize(x)]) for x in (option2, option3)
            )
            if dash:
                # a dash ending a word joins the next word without a space
                best_score_idx = 1
            elif kept != joined:
                best_score_idx = 1 if kept else 2
            else:
                options = [option1, option2, option3]
                if context:
                    before, after = context_of(lines, idx, context)
                    options = [
                        f"{before} {o.rstrip()} {after}".strip() for o in options
                    ]
                scores = self.score(options)
                best_score_idx = scores.index(min(scores))

            # option 2
            if best_score_idx == 1:
                lines[idx + 1][0] = option2
                lines[idx].pop()

                if len(lines[idx]) > 0:
                    lines[idx][-1] += " "

            # option 3
            if best_score_idx == 2:
                lines[idx + 1][0] = option3
                lines[idx].pop()

                if len(lines[idx]) > 0:
                    lines[idx][-1] += " "

        # Some lines can be empty now, remove them.
        return [x for x in lines if len(x) > 0]

    def is_split_paragraph(self, para1, para2):
        """currently only checks for the last / first line of the both paragraphs"""
        assert_format(para1)
        assert_format(para2)

        line1 = " ".join(para1[-1]).rstrip()
        line2 = " ".join(para2[0])
        last_word = para1[-1][-1].rstrip()
        hyphen = hyphen_of(last_word)
        if hyphen is not None:
            # a soft hyphen shows as `-` at the end of a line
            line1, last_word = line1[:-1] + hyphen, last_word[:-1] + hyphen

        dash = is_dash(last_word)
        # a dash ending a word joins the next word without a space
        options = [line1 + "\n\n" + line2, line1 + ("" if dash else " ") + line2]
        if hyphen is not None and not dash:
            # keep the hyphen of a compound word, or remove it
            options += [line1 + line2, line1[:-1] + line2]

        scores = self.score(options)

        best_score_idx = scores.index(min(scores))

        if best_score_idx == 0:
            return None

        if best_score_idx == 1 and not dash:
            para1[-1][-1] = last_word + " "
            return para1 + para2

        # joins the last word of para1 with the first word of para2, the last option without the hyphen
        head = last_word[:-1] if best_score_idx == 3 else last_word
        para1[-1][-1] = head + para2[0].pop(0)

        # the first line of para2 is empty if it was only that word
        rest = [l for l in para2 if l]
        if rest and not ends_with_whitespace(para1[-1][-1]):
            para1[-1][-1] += " "
        return para1 + rest
