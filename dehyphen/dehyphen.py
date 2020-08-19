from cleantext import clean

from .format import assert_format


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

            # transliterate to transform different kinds of hyphens to "-"
            last_char = clean(last_word.strip()[-1])
            if last_char != "-":
                continue

            next_word = lines[idx + 1][0]

            # for flair, the words has to be at least 2 characters long
            if min(map(len, (last_word, next_word))) < 2:
                continue

            # 1. two words (e.g. part of an abrev.), so do nothing
            option1 = last_word + next_word

            # 2. some compound-word (keep hyphen), remove whitespace
            option2 = last_word.strip() + next_word

            # 3. remove hyphen, most likely to happen
            option3 = last_word.strip()[:-1] + next_word

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
        """currently only checks for the last / first line of the both paragraphs
        """
        assert_format(para1)
        assert_format(para2)

        # flair does not work with only one char
        if len(para1) == 1 and len(para1[0]) == 1:
            return None
        if len(para2) == 1 and len(para2[0]) == 1:
            return None

        options = []
        # the lines may have newlines or whitespace in the end
        options.append(" ".join(para1[-1]) + "\n\n" + " ".join(para2[0]))
        options.append(" ".join(para1[-1] + para2[0]))

        last_word_para1 = para1[-1][-1]
        if clean(last_word_para1[-1]) == "-":
            options.append(" ".join(para1[-1])[:-1] + " ".join(para2[0]))

        scores = self.score(options)

        best_score_idx = scores.index(min(scores))

        if best_score_idx == 0:
            return None

        if best_score_idx == 1:
            para1[-1][-1] += " "
            return para1 + para2

        if best_score_idx == 2:
            # joins the last word of para2 with the first word of para1
            para1[-1][-1] = last_word_para1[:-1] + para2[0].pop(0) + " "
            return para1 + para2
