import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pilot.corpus import Document, build_index_views
from pilot.evaluate import auc
from pilot.queries import lexical_leak
from pilot.retrievers import BM25, rrf
from pilot.pdf_extract import clean_body


class PilotTests(unittest.TestCase):
    def test_views_keep_author_english_abstract_separate(self):
        document = Document("x", "source", "fr", "Titre", "resume", "abstract", "corps")
        views = build_index_views([document])
        self.assertIn("abstract", views["en_abstract"]["x"])
        self.assertNotIn("corps", views["en_abstract"]["x"])
        self.assertIn("corps", views["native"]["x"])

    def test_bm25_and_rrf_are_deterministic(self):
        index = BM25({"a": "blue cat", "b": "red dog"})
        self.assertEqual(index.rank("blue")[0][0], "a")
        self.assertEqual(rrf([["b", "a"], ["a", "b"]])[0][0], "a")

    def test_metrics_helpers(self):
        self.assertGreater(lexical_leak("a method for speech", "Speech recognition method"), 0)
        self.assertEqual(auc([(1.0, 1), (0.0, 0)]), 1.0)

    def test_source_title_occurs_at_most_once_in_first_1500_body_words(self):
        body = "Paper title\nHeader\nIntroduction\nTexte\n\fHeader\nTexte\nReferences\nCitation"
        cleaned = clean_body(body, title="Paper title")
        self.assertNotIn("Header", cleaned)
        self.assertNotIn("Citation", cleaned)
        first_1500 = " ".join(cleaned.split()[:1500])
        self.assertLessEqual(first_1500.lower().count("paper title"), 1)


if __name__ == "__main__":
    unittest.main()
