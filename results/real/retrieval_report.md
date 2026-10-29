# Retrieval P0 sanity control

- corpus: 620 documents (20 sources, 100 French distractors, 500 English distractors)
- French distractors with author-written English abstract: 0
- exact model IDs: dense `BAAI/bge-m3`, translation `Helsinki-NLP/opus-mt-fr-en`
- dense document max length: 1024 tokens
- translation chunk maximum: 400 tokens
- optional accepted IDs loaded: 0

| Retriever | View | Recall@10 |
|---|---|---:|
| bm25 | native | 0.500 |
| dense_bge_m3 | native | 0.250 |
| hybrid_rrf | native | 0.400 |
| bm25 | en_abstract | 1.000 |
| dense_bge_m3 | en_abstract | 1.000 |
| hybrid_rrf | en_abstract | 1.000 |
| bm25 | both | 1.000 |
| dense_bge_m3 | both | 0.250 |
| hybrid_rrf | both | 0.450 |
| doc_translate_bm25 | native | 1.000 |

## Sanity interpretation

The en_abstract view is perfect for this complete-English-abstract control. The lower dense score on native/both is reported as observed and was not tuned or patched.

- runtime_seconds: 254.80
