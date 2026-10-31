import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from leak_checker import check_blind_entry, apply_redaction


class LeakCheckerTests(unittest.TestCase):
    def test_synthetic_french_body_starts_pass_all_checks(self):
        """Synthetic fixtures starting with 'Introduction La...', 'Introduction Les...', 'Introduction Dans...' must NOT fail english_title_like_start."""
        prefixes = [
            "Introduction La préservation de notre héritage culturel passe par la numérisation.",
            "Introduction Les systèmes de résumé automatique sont des constituants essentiels.",
            "Introduction Dans le domaine croissant des agents conversationnels animés.",
        ]
        for body in prefixes:
            row = {
                "paper_id": "test.paper.1",
                "body_fr": body + " Suite du corps de l'article scientifique sans métadonnées.",
            }
            checks = check_blind_entry(
                filename="t4_passB_input.jsonl",
                row=row,
                doc_title="Un Titre Français Valide",
                doc_abstract_en="A genuine author abstract in English.",
                authors=["Jean Dupont", "Marie Curie"],
            )
            # english_title_like_start must NOT be present in t4_passB checks
            check_names = [c["check"] for c in checks]
            self.assertNotIn("english_title_like_start", check_names)

            # All checks must PASS
            failures = [c for c in checks if c["status"] == "FAIL"]
            self.assertEqual(len(failures), 0, f"Expected 0 failures, got {failures} for body: {body}")

    def test_t4_passb_catches_author_email_affiliation(self):
        """t4_passB must fail if email, author, or affiliation is present."""
        row_with_email = {"paper_id": "test.2", "body_fr": "Contact: test@univ-paris.fr for questions."}
        checks = check_blind_entry("t4_passB_input.jsonl", row_with_email, "Titre", "Abs", ["A. Author"])
        statuses = {c["check"]: c["status"] for c in checks}
        self.assertEqual(statuses["at_sign"], "FAIL")
        self.assertEqual(statuses["first_page_front_matter"], "FAIL")

        row_with_author = {"paper_id": "test.3", "body_fr": "Introduction par Jean Dupont dans cet article."}
        checks = check_blind_entry("t4_passB_input.jsonl", row_with_author, "Titre", "Abs", ["Jean Dupont"])
        statuses = {c["check"]: c["status"] for c in checks}
        self.assertEqual(statuses["author_names"], "FAIL")

    def test_abstract_en_title_leak_detected_and_redacted(self):
        """abstract_en with Title Case start fails english_title_like_start, but passes after redaction."""
        raw_abstract = (
            "Automatic Summarization Using a Genetic Algorithm This paper proposes a novel method "
            "for automatic summarization based on a genetic algorithm."
        )
        row = {"paper_id": "2015.court.13", "abstract_en": raw_abstract}
        checks = check_blind_entry("t4_passA_input.jsonl", row, "Titre FR", raw_abstract, [])
        statuses = {c["check"]: c["status"] for c in checks}
        self.assertEqual(statuses["english_title_like_start"], "FAIL")

        # After applying confirmed redaction
        redacted = apply_redaction(raw_abstract, "Automatic Summarization Using a Genetic Algorithm ")
        row_redacted = {"paper_id": "2015.court.13", "abstract_en": redacted}
        checks_after = check_blind_entry("t4_passA_input.jsonl", row_redacted, "Titre FR", raw_abstract, [])
        statuses_after = {c["check"]: c["status"] for c in checks_after}
        self.assertEqual(statuses_after["english_title_like_start"], "PASS")


if __name__ == "__main__":
    unittest.main()

