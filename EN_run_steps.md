> [!IMPORTANT]
> **EXPLORATORY - agent-proxy labels, human verification pending**

# English Terminology Dose-Response Pilot: Colab GPU Execution Guide (Step E3)

This guide provides instructions for executing the neural dense retrieval (`dense_bge_m3`) and hybrid reciprocal rank fusion (`hybrid_rrf`) on Google Colab with GPU acceleration.

---

## 1. Local Files & Integrity Check
Before running on Colab, ensure you have:
- `results/real/en/en_data_bundle.zip` (2.55 MB, containing 300 source papers, 5,000 distractors, 576 test queries, and manifest).
- Git repository synced with GitHub `origin/main`.

---

## 2. Step-by-Step Instructions on Google Colab

### Bước 1: Chuẩn bị file dữ liệu trên Google Drive
Copy file sau từ máy tính lên Google Drive của bạn (vào thư mục bất kỳ trên Drive, ví dụ `MyDrive/` hoặc `MyDrive/BK-Paper-Cache/`):
- `d:\BK-Paper\Idea_agentic\results\real\en\en_data_bundle.zip`

### Bước 2: Mở Colab Notebook & Chọn GPU
1. Mở file [`agentic_v2.ipynb`](file:///d:/BK-Paper/Idea_agentic/agentic_v2.ipynb) trên Google Colab.
2. Kiểm tra Runtime: Vào **Runtime -> Change runtime type**, chọn **T4 GPU** (hoặc A100 GPU nếu có).

### Bước 3: Chạy các Cell theo thứ tự
1. **Cell 0**: Mount Google Drive (`drive.mount('/content/drive')`).
2. **Cell 1 & 2**: Clone hoặc `git pull origin main` repo `Idea_agentic`.
3. **Cell 5 & 6**: Cài đặt môi trường (`sentence-transformers`, `FlagEmbedding`, `rank-bm25`, `pyarrow`).
4. **Cuộn xuống phần "Colab Run 3: English Terminology Dose-Response Pilot"**:
   - **Cell Giải nén Bundle**: Tự động tìm và giải nén `en_data_bundle.zip` từ Google Drive vào workspace.
   - **Cell Chạy Retrieval**: Chạy script:
     ```bash
     !python -u scripts/run_retrieval_colab_en.py --drive-cache /content/drive/MyDrive/BK-Paper-Cache/
     ```
     Script sẽ tự động:
     - Nạp 5.270 tài liệu (300 bài nguồn + 5.000 bài gây nhiễu).
     - Nạp 576 truy vấn kiểm thử (R100, R50, R25, R0).
     - Nạp/Tính toán embeddings BGE-M3 (lưu cache trên Drive để không phải nhúng lại lần sau).
     - Chạy song song Okapi BM25, Dense BGE-M3, và Hybrid RRF ($k=60$).
     - Xuất các file xếp hạng đầy đủ: `rank_export_en.csv`, `rank_export_en.parquet`, `retrieval_en_metrics.csv`, `rank_export_en_manifest.json`.
   - **Cell Xuất về Google Drive**: Tự động copy toàn bộ kết quả xuất sang thư mục `BK-Paper-Results/en/` trên Google Drive của bạn.

---

## 3. Lấy kết quả về máy & Bước kế tiếp (Step E4)
Sau khi Colab chạy xong:
1. Tải 4 file kết quả từ thư mục `BK-Paper-Results/en/` trên Drive về máy tính của bạn:
   - `rank_export_en.csv`
   - `rank_export_en.parquet`
   - `rank_export_en_manifest.json`
   - `retrieval_en_metrics.csv`
2. Đặt chúng vào thư mục:
   `d:\BK-Paper\Idea_agentic\results\real\en\`
3. Báo cho assistant để tiếp tục thực hiện **Step E4** (kiểm tra tái lập, vẽ biểu đồ đường cong liều lượng dose-response, kiểm soát closed-book, kiểm tra các quy tắc quyết định tự động, và lập danh mục thất bại cho con người).
