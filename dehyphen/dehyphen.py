from .format import assert_format

# characters that hyphenate a word at the end of a line
HYPHENS = "-\u00ad\u2010\u2011\u2013\u2212"
SOFT_HYPHEN = "\u00ad"


def ends_with_whitespace(s):
    return len(s) != len(s.rstrip())


def hyphen_of(word):
    """The hyphen `word` ends with as printed, `None` if there is none or the word is a lone dash."""
    word = word.rstrip()
    if len(word) < 2 or word[-1] not in HYPHENS:
        return None
    # a soft hyphen only shows at the end of a line
    return "-" if word[-1] == SOFT_HYPHEN else word[-1]


class Scorer:
    def dehyphen(self, paragraphs):
        return [self.dehyphen_paragraph(p) for p in paragraphs]

    def dehyphen_paragraph(self, lines):
        """dehyphenation of paragraph (lines)

        Lines have to be in special format. 2D List, word should not contain whitespace, except the last words per line
        """
        assert_format(lines)

        # Iterate over all lines, change the next line if necessary. This ensure the updates version will get used in the next loop.
        # Make sure to access the lines with the `idx`.
        for idx in range(len(lines) - 1):
            # don't work on last line

            l = lines[idx]

            if len(l) == 0:
                continue

            last_word = l[-1]

            # don't work if there has to be a newline
            if last_word[-1] == "\n":
                continue

            hyphen = hyphen_of(last_word)
            if hyphen is None:
                continue

            next_word = lines[idx + 1][0]

            # for flair, the words has to be at least 2 characters long
            if len(next_word) < 2:
                continue

            stem = last_word.strip()[:-1]

            # 1. two words (e.g. part of an abrev.), so do nothing
            option1 = last_word + next_word

            # 2. some compound-word (keep hyphen), remove whitespace
            option2 = stem + hyphen + next_word

            # 3. remove hyphen, most likely to happen
            option3 = stem + next_word

            scores = self.score((option1, option2, option3))

            best_score_idx = scores.index(min(scores))
            assert best_score_idx in (0, 1, 2)

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

        options = [line1 + "\n\n" + line2, line1 + " " + line2]
        hyphen = hyphen_of(last_word)
        if hyphen is not None:
            # keep the hyphen of a compound word, or remove it
            options += [line1[:-1] + hyphen + line2, line1[:-1] + line2]

        scores = self.score(options)

        best_score_idx = scores.index(min(scores))

        if best_score_idx == 0:
            return None

        if best_score_idx == 1:
            para1[-1][-1] = last_word + " "
            return para1 + para2

        # joins the last word of para1 with the first word of para2
        joint = hyphen if best_score_idx == 2 else ""
        para1[-1][-1] = last_word[:-1] + joint + para2[0].pop(0)

        # the first line of para2 is empty if it was only that word
        rest = [l for l in para2 if l]
        if rest and not ends_with_whitespace(para1[-1][-1]):
            para1[-1][-1] += " "
        return para1 + rest
