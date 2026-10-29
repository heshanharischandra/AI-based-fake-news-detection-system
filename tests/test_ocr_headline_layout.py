"""Geometry regressions for news headlines split into separate OCR blocks."""
import unittest

from app.services.ocr_service import _headline_candidates


def line(text, left, top, right, bottom, confidence=90):
    return dict(text=text, left=left, top=top, right=right, bottom=bottom,
                confidence=confidence, key=(1, 1, 1))


class HeadlineLayoutTests(unittest.TestCase):
    def test_horizontal_sinhala_fragments_form_one_complete_headline(self):
        lines = [line("'අග්බෝ' සොයන", 40, 61, 538, 132, 90),
                 line('මෙහෙයුම්\u200c', 572, 62, 865, 136, 66),
                 line('දෙවන දිනය ඇරඹේ', 898, 62, 1491, 136, 95)]
        best = max(_headline_candidates(lines, (1574, 1090)), key=lambda item: item['score'])
        self.assertEqual(best['text'], "'අග්බෝ' සොයන මෙහෙයුම්\u200c දෙවන දිනය ඇරඹේ")

    def test_large_horizontal_gaps_do_not_join_columns(self):
        lines = [line('First separate column', 10, 10, 120, 30),
                 line('Second separate column', 400, 10, 600, 30)]
        candidates = _headline_candidates(lines, (800, 600))
        self.assertTrue(candidates)
        self.assertFalse(any('First' in c['text'] and 'Second' in c['text'] for c in candidates))

    def test_small_badge_does_not_merge_into_large_headline(self):
        lines = [line('NEWS', 5, 10, 35, 20),
                 line('Main news headline', 45, 10, 400, 50)]
        candidates = _headline_candidates(lines, (800, 600))
        self.assertFalse(any(c['text'].startswith('NEWS Main') for c in candidates))

    def test_multiline_headline_still_joins_and_input_is_not_modified(self):
        lines = [line('Main headline first line', 20, 20, 400, 50),
                 line('continues on second line', 20, 60, 400, 90)]
        before = [dict(row) for row in lines]
        candidates = _headline_candidates(lines, (800, 600))
        self.assertTrue(any(c['text'] == 'Main headline first line continues on second line' for c in candidates))
        self.assertEqual(lines, before)


if __name__ == '__main__':
    unittest.main()
