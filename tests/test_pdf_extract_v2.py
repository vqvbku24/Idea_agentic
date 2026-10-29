import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pilot.pdf_extract_v2 import split_abstracts_v2


class ExtractorV2Fixtures(unittest.TestCase):
    def test_inline_and_standalone_headers_keep_first_line(self):
        result = split_abstracts_v2(
            "Résumé\nPremière ligne française.\nMOTS-CLÉS: test\n"
            "Abstract. First English line.\nSecond line.\nIntroduction\nCorps.",
            title="Titre",
        )
        self.assertIn("Première ligne", result["fr_abstract"])
        self.assertIn("First English line", result["en_abstract"])

    def test_title_and_challenge_prefix_are_removed(self):
        result = split_abstracts_v2(
            "Abstract\nA title to remove\nDEFT 2021\nActual abstract text.\n"
            "Introduction\nCorps.",
            title="A title to remove",
        )
        self.assertEqual(result["en_abstract"], "Actual abstract text.")
        self.assertIn("en_prefix_removed", result["flags"])

    def test_front_matter_and_keywords_are_not_body(self):
        result = split_abstracts_v2(
            "Résumé\nRésumé.\nMOTS-CLÉS: mot\nAbstract\nAbstract text.\n"
            "KEYWORDS: keyword\nAuteur\nemail@example.org\n1 Introduction\n"
            "Le corps commence ici.",
            title="Titre",
        )
        self.assertNotIn("email@", result["body"])
        self.assertNotIn("keyword", result["body"].lower())
        self.assertIn("corps commence", result["body"])


if __name__ == "__main__":
    unittest.main()
