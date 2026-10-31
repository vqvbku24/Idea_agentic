# Project Handoff: Idea Agentic Pilot

## 1. Mục tiêu

Pilot kiểm tra tính khả thi của việc truy hồi prior-art đa ngôn ngữ cho 20 bài báo tiếng Pháp thuộc JEP/TALN/RECITAL, dùng abstract tiếng Pháp làm query và đánh giá khả năng tìm đúng tài liệu trong corpus có distractors tiếng Anh/Pháp.

Phạm vi P0 đã hoàn thành:

- Thu thập metadata ACL Anthology, không hard-code paper IDs.
- Chọn paper theo năm với seed `13`.
- Tải PDF tối đa `1 request/second`.
- Trích xuất abstract Pháp, abstract Anh và body.
- Lưu provenance `pdf | metadata | missing`.
- Không dùng machine translation để lấp English abstract bị thiếu.
- Xây dựng retrieval BM25, dense BGE-M3, hybrid RRF và document-translation BM25.
- Chạy retrieval trên 20 source papers, 100 French distractors và 500 English distractors.

## 2. Ràng buộc bất biến

Không được:

- Sửa `src/pilot/pdf_extract.py` (extractor v1 đã freeze).
- Sửa các tag hiện có.
- Sửa `config/pilot_mini.yaml`.
- Sửa `config/decision_rules_mini.yaml`.
- Sửa `data/processed/sources_final.json`.
- Gọi LLM/provider.
- Đọc `data/human/human_queries.csv`.
- Đọc facet-audit files nếu không thật sự cần.
- Tạo quyết định human giả.
- Commit `data/`, PDF, paper text hoặc generated outputs lớn.
- Tạo tag mới nếu gate chưa đạt.

Khi ambiguity xuất hiện, dùng `unsure` theo adversarial default. Trong quyết định rule v2.1, `unsure` được tính là false positive.

## 3. Các mốc chính

### Extraction và source acquisition

- `b422f1f` — freeze extractor v1 (`pdf-extract-frozen`).
- Đã tạo development subset 10 paper.
- Đã chọn 20 source paper mới, loại development subset.
- Đã thêm body cleanup:
  - loại repeated page headers/footers;
  - loại từ references trở đi;
  - kiểm tra title không lặp quá một lần trong 1500 từ body đầu.

### Human/audit và retrieval

- `4580860` — human templates và LLM abstraction; không gọi provider.
- `c3cafb1` — freeze retrieval v1 (`retrieval-frozen`).
- Retrieval gồm:
  - BM25;
  - `dense_bge_m3` dùng `BAAI/bge-m3`;
  - hybrid RRF với `k=60`;
  - `doc_translate_bm25`;
  - các view `native`, `native_abstract`, `en_abstract`, `both`, `native_kw_en`.

### Extractor v2

- `c3bbeec` — extractor v2/amendment-2.
- Tag hiện có: `extractor-v2`, `amendment-2`.
- Extractor v2 là file mới `src/pilot/pdf_extract_v2.py`; không sửa v1.
- X2 artefacts gồm source, development và French distractor outputs.

### Reproducibility/Colab

- `23deaf4` — reproducible final pre-freeze run.
- `407bf8d` — Drive-backed data `.gitignore`.
- `e18c87d` — Stage A/B tooling và rejected amendment gate.
- `f7d20be` — loại mutable run metadata và self-including ZIP khỏi data bundle.
- `cc108f5` — notebook workflow ban đầu.
- `8c5af33` — thêm `french_distractors_native.jsonl` bắt buộc cho retrieval bundle.
- `3c81169` — pin notebook vào distractor bundle fix.
- `3fa867b` — không overwrite English metadata đã được hash-verify.
- `dd1171c` — bundle-first verification và xoá stale extracted files trước restore.
- `aa0e3b0` — pin notebook vào verifier hardened mới nhất.

Commit mới nhất cần dùng cho Colab:

```text
aa0e3b0
```

Notebook:

- [agentic_v2.ipynb](./agentic_v2.ipynb)

## 4. Stage A đã làm gì

Đã tạo:

- `scripts/build_data_bundle.py`
- `scripts/build_view_manifest.py`
- `scripts/run_stage_a.py`
- `scripts/sign_gate.py`

Artefacts:

- `results/real/data_bundle_x2.zip`
- `results/real/data_manifest.json`
- `results/real/view_manifest.json`
- `results/real/resume_check_x2.csv`
- `results/real/HUMAN_REVIEW_SHEET.md`
- `data/human/leak_review_draft.csv`
- `data/human/v21_review_draft.csv`
- `data/human/human_review_decisions.csv`

Kết quả Stage A:

- Bundle ban đầu gồm 17 file; bundle hiện tại sau sửa gồm 17 file và không chứa `run_meta_x2.json` hoặc chính `data_bundle_x2.zip`.
- Paper text files trong manifest: 10 sau khi thêm legacy French distractor input.
- View sizes:
  - `native`: 620
  - `native_abstract`: 620
  - `native_kw_en`: 620
  - `en_abstract`: 611
  - `both`: 611
- Có 9 document thiếu author-written English abstract; không auto-fill.
- Leak draft ban đầu: 10 FAIL cells.
- v2.1 draft: 26 items.
- Resume check: `fr_ok=20/20`; English abstract comparison là heuristic và không phải tất cả abstract Anh bắt đầu bằng cùng đoạn raw French page.

## 5. Signing gate và quyết định v2.1

Human đã chạy:

```powershell
python scripts/sign_gate.py --reviewer "<initials>"
```

Gate hợp lệ:

- 35/35 items đã quyết định.
- `leak_review.csv` hash:
  `d66bee6d65578d8eeee415476c72e9bb1430148431a36b2f2c5ab58a0345eb2a`
- `v21_review.csv` hash:
  `6d8a15fe69218bc77f0a678adf18f060772c86e5f8dc0e9893f88d7c0c78217f`
- Reviewer hiện được ghi literal là `<initials>`; đây là placeholder, không nên coi là initials thật nếu cần audit formal.

Quyết định Stage B:

- 26 v2.1 items đã review.
- 20 false positives, gồm unsure nếu có.
- Ngưỡng áp dụng: tối đa 2 false positives và không false positive nào trong 20 source.
- Outcome: **v2.1 rejected**.
- Không tạo `src/pilot/pdf_extract_v2_1.py`.
- Không sửa extractor v1/v2.
- Chỉ tạo sensitivity diagnostic:
  `results/real/sensitivity_native_abstract_x2.jsonl`
- Report:
  `results/real/stage_b_rejected_report.md`

Stage B rebuild:

- 200 blind checks.
- 9 FAIL còn lại.
- Vì leak checks chưa đóng, không được tạo `retrieval-frozen-v2`.

## 6. Kết quả Colab cũ đã review

Artefact cũ từng nằm ở:

```text
results/real-colab/real
```

Run cũ có:

- commit `407bf8d`, không phải commit mới;
- không có `data_manifest.json` và `view_manifest.json` đầy đủ;
- không có signed `data/human/leak_review.csv` và `v21_review.csv` trong Colab output;
- retrieval sanity French-query trên `native_abstract` trước đó đạt 1.0 cho BM25/dense/hybrid;
- abstract English dense reference đạt cosine gần 1.0;
- native ở max length 8192 thấp hơn 0.99, khoảng min `0.851`;
- translation output khoảng 2767 chunks, 17 detected French (`0.614%`).

Run này **không đủ điều kiện Stage C**.

## 7. Lỗi Colab đã gặp và nguyên nhân

### Clone không thấy commit

Nguyên nhân: commit local chưa push GitHub.

Khắc phục:

```powershell
git push origin main
```

### Thiếu `data_manifest.json`

Nguyên nhân: chạy `colab_run_manifest.py` trước khi copy bundle/manifest vào repo.

### Thiếu `french_distractors_native.jsonl`

Nguyên nhân: bundle chỉ có `_x2`, trong khi `run_retrieval_v2.py` còn đọc legacy file này.

Đã sửa `build_data_bundle.py` để đóng gói cả:

- `results/real/french_distractors_native.jsonl`
- `results/real/french_distractors_native_x2.jsonl`

### Hash mismatch `run_meta_x2.json`

Nguyên nhân:

- file mutable `run_meta_x2.json` bị đưa vào immutable manifest;
- một lần bundle còn tự chứa chính `data_bundle_x2.zip`, làm ZIP phình lên nhiều GB.

Đã loại cả hai khỏi bundle.

### Hash mismatch English metadata

Nguyên nhân:

- notebook giải nén English metadata đúng từ bundle;
- sau đó copy bản khác từ Drive đè lên;
- hoặc Colab còn stale file từ run trước.

Đã sửa:

- bundle là nguồn chuẩn;
- notebook xoá files theo manifest trước khi extract;
- notebook chỉ copy external `french_distractors_metadata.jsonl`;
- không copy/overwrite `english_distractors_metadata.jsonl`.

## 8. Notebook Colab hiện tại

Notebook [agentic_v2.ipynb](./agentic_v2.ipynb) phải:

1. Mount Drive.
2. Xoá `/content/Idea_agentic` cũ.
3. Clone repo.
4. Checkout `aa0e3b0`.
5. Link chỉ:
   - `data/raw`
   - `data/cache`
6. Copy canonical:
   - `data_bundle_x2.zip`
   - `data_manifest.json`
   - `view_manifest.json`
7. Xoá stale extracted files theo manifest.
8. Extract bundle.
9. Copy chỉ French metadata ngoài bundle.
10. Cài package.
11. Verify manifest.
12. Chạy `run_retrieval_v2.py` đúng một lần.
13. Chạy `finalize_x2_audit.py`.
14. Chạy unit tests và compileall.
15. Copy output và timestamped `run_meta_*.json` về:

```text
MyDrive/Idea_agentic/results/real-colab/real
```

Ba file Drive bắt buộc phải được tạo cùng một lần:

```text
MyDrive/Idea_agentic/results/real/data_bundle_x2.zip
MyDrive/Idea_agentic/results/real/data_manifest.json
MyDrive/Idea_agentic/results/real/view_manifest.json
```

Không ghép manifest cũ với bundle mới.

## 9. Cách kiểm tra Colab trước khi chạy retrieval

Sau restore bundle, phải tồn tại:

```text
/content/Idea_agentic/results/real/data_manifest.json
/content/Idea_agentic/results/real/view_manifest.json
/content/Idea_agentic/results/real/papers_x2.jsonl
/content/Idea_agentic/results/real/development_papers_x2.jsonl
/content/Idea_agentic/results/real/french_distractors_native.jsonl
/content/Idea_agentic/results/real/french_distractors_native_x2.jsonl
/content/Idea_agentic/data/processed/english_distractors_metadata.jsonl
/content/Idea_agentic/data/processed/french_distractors_metadata.jsonl
```

Manifest command:

```bash
python scripts/colab_run_manifest.py --ref aa0e3b0
```

Output hợp lệ phải tạo:

```text
results/real/run_meta_YYYYMMDDTHHMMSSZ.json
```

Run metadata phải có:

- `git.head` bắt đầu bằng `aa0e3b0`;
- `git.ref = aa0e3b0`;
- `status_porcelain` rỗng;
- `data_manifest`;
- `view_manifest`;
- device/package metadata.

## 10. Cần lấy gì sau Colab run mới

Copy toàn bộ folder output mới về `results/real-colab/real`, tối thiểu gồm:

- `run_meta_*.json` — bắt buộc, không chỉ `run_meta_x2.json`;
- `data_manifest.json`;
- `view_manifest.json`;
- `retrieval_metrics.csv`;
- `retrieval_v2_diagnostics.md`;
- `dense_reference_x2.csv`;
- `translations_x2.jsonl`;
- `blind_leak_checks_x2.csv`;
- `leak_fail_cases_x2.md`;
- `candidate_rule_v2_1_review.md`;
- `finalize_x2_audit` outputs;
- `colab_logs/*.log`.

Không cần copy:

- `data/raw`;
- `data/cache`;
- Hugging Face model cache.

## 11. Stage C còn phải làm

Chỉ bắt đầu khi user cung cấp folder output Colab mới.

### C1. Review

Viết:

```text
results/real/colab_run_final_review.md
```

Kiểm tra:

- Colab tree sạch.
- Commit Colab đúng commit repo.
- Hash source/config/source list khớp.
- Data bundle hashes khớp `data_manifest.json`.
- View manifest giống local.
- French-query Recall@10:
  - BM25;
  - dense BGE-M3;
  - hybrid RRF;
  - `native_abstract`;
  - cả 500 và 100 English distractors;
  - threshold `>= 0.95`.
- English-query controls cho mọi retriever, không threshold cho `doc_translate_bm25`.
- Dense reference:
  - `abstract_en` cosine >= 0.99 ở 1024 và 8192;
  - native 1024 report riêng;
  - native 8192 nếu fail phải report, không tự sửa.
- Translation output tồn tại và share chunks còn French được report.
- Leak checks:
  - tất cả PASS, hoặc FAIL được signed false-positive exception.
- Unit tests và compileall pass.

### C2. Tag gate

Chỉ nếu mọi criterion pass và signed files tồn tại:

```text
extractor-v2
amendment-2
retrieval-frozen-v2
```

Không tạo tag nếu có bất kỳ FAIL nào.

### C3. Handoff cho T1/T3/T4

Chỉ tạo `handoff/` nếu Stage C pass, gồm tối thiểu:

- extraction review;
- `t3_input.jsonl`;
- `t4_passA_input.jsonl`;
- `t4_passB_input.jsonl`;
- leak PASS/FAIL table.

## 12. Trạng thái hiện tại

- Latest local commit: `aa0e3b0`.
- Latest Colab output user đã đưa trước đó: run cũ hoặc chưa đủ provenance.
- Stage A: complete.
- Stage B: complete with v2.1 rejected.
- Stage C: not started / waiting for a valid new Colab output.
- No new freeze tag may be created yet.

