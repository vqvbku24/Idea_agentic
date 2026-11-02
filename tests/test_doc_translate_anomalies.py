# EXPLORATORY - agent-proxy labels, human verification pending
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.retrievers import BM25, DocTranslateBM25
from pilot.queries import Query
from pilot.evaluate import rank_without_source, auc


class TestDocTranslateAnomalies(unittest.TestCase):
    def test_anomaly_a_absent_control_candidate_filtering(self):
        """Anomaly (a): rank_without_source must exclude query.source_id.
        In the old code, candidates was passed to 'view', so candidate filtering was ignored,
        causing present_top == absent_top and AUC == 0.50 everywhere.
        """
        # Create a mock DocTranslateBM25 instance using raw BM25 index
        doc_texts = {
            "source_1": "reinforcement learning policy gradient optimization",
            "distractor_1": "quantum computing superconductor qubit",
            "distractor_2": "graph neural network node classification",
        }
        mock_ranker = object.__new__(DocTranslateBM25)
        mock_ranker.index = BM25(doc_texts)
        mock_ranker.doc_ids = mock_ranker.index.doc_ids
        mock_ranker.translated_texts = doc_texts

        query = Query(
            source_id="source_1",
            variant="V0",
            text="reinforcement learning policy gradient",
            accepted=True,
        )

        # Full ranking (with source)
        full_rank = mock_ranker.rank(query.text)
        self.assertEqual(full_rank[0][0], "source_1")
        present_top_score = full_rank[0][1]

        # Absent ranking (without source)
        absent_rank = rank_without_source(mock_ranker, query)
        absent_ids = [doc_id for doc_id, _ in absent_rank]
        self.assertNotIn("source_1", absent_ids, "Source must be excluded in absent control!")

        absent_top_score = absent_rank[0][1]
        self.assertGreater(
            present_top_score,
            absent_top_score,
            "Present top score should exceed absent top score for relevant query",
        )

        # Pair score AUC should NOT be 0.50
        scored_pairs = [(present_top_score, 1), (absent_top_score, 0)]
        self.assertEqual(auc(scored_pairs), 1.0)

    def test_anomaly_b_english_view_passthrough_vs_mangled(self):
        """Anomaly (b): On en_abstract view, text is already English.
        Passing already-English text through French-to-English MT model corrupts syntax/terms.
        DocTranslateBM25 should index author English text directly on en_abstract.
        """
        original_en = "Bayesian optimization for hyperparameter tuning in neural networks"
        # Synthetic simulation of what MT does when fed non-French text: degrades/drops terms
        mangled_en = "Tuning in neural networks"

        corpus_clean = {"paper_1": original_en}
        corpus_mangled = {"paper_1": mangled_en}

        bm25_clean = BM25(corpus_clean)
        bm25_mangled = BM25(corpus_mangled)

        query = "Bayesian optimization"
        score_clean = bm25_clean.rank(query)[0][1]
        score_mangled = bm25_mangled.rank(query)[0][1]

        self.assertGreater(
            score_clean,
            score_mangled,
            "Clean author English text retains technical terms, while mangled text loses them",
        )


if __name__ == "__main__":
    unittest.main()
