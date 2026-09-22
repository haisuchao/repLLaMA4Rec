# Negative / Superseded Results — không copy model, chỉ tham khảo số liệu

> Mục đích: liệt kê các hướng/thực nghiệm **đã thử và không dùng cho paper FADE** (kết quả kém hoặc bị thay
> thế bởi hướng khác), để khi làm việc trên máy khác **không cần copy checkpoint/model** — chỉ cần đọc file
> này để biết "đã thử rồi, kết quả thế nào, tại sao bỏ" và tránh lặp lại vô ích.
>
> Số liệu dưới đây lấy trực tiếp từ `experiments.md`, `improvement_plan.md`, `MULTI_MACHINE_GUIDE.md` và các
> file `output/*/*/embeddings/results/*.txt` trên máy hiện tại (đọc ngày 2026-09-22). Baseline so sánh: FADE
> chuẩn hiện tại (Beauty aug-5, `− filter`: NDCG@10 raw = 0.0390; `+ filter full`: NDCG@10 = 0.0616 — xem Table
> 3 trong `paper/paper_draft.md`).

---

## 1. BM25 hard negatives — thất bại nặng, do false-negative poisoning

**Config**: `export_tevatron.py beauty --neg_strategy bm25` (50 BM25-mined hard negatives/query, thay random).

**Kết quả (Beauty)**: NDCG@10 raw = **0.0209** (< random-negative baseline 0.0372, tệ hơn cả zero-shot 0.0135
theo NDCG@5). Sau history-filter (mode=full) chỉ cải thiện +14.4% (0.0239) — so với +47-61% ở mọi model khác
đã fine-tune. Đây là bằng chứng lỗi nằm ở **training-time** (embedding space bị phá), không phải lỗi ranking
bề mặt mà filter có thể sửa.

**Nguyên nhân**: BM25 tìm item title-similar về mặt lexical → đây thường là true near-items (cùng category/
brand) → bị label nhầm thành negative → model học tránh retrieve các item có title tương tự item đúng, phản
tác dụng hoàn toàn.

**Kết luận**: KHÔNG dùng BM25/lexical-similarity để mine hard negative cho bài toán này. Nếu muốn thử hard
negative mining, nên dùng embedding-space mining (mine từ chính model đã fine-tune, không phải lexical) — xem
`improvement_plan.md` §6.3, ý tưởng chưa implement.

---

## 2. Untied dual-encoder (2 LoRA adapter riêng cho query/passage) — thất bại nặng, CHƯA rõ do bug hay bản chất

**Config**: `train.sh beauty --data-variant v2-cs5-aug --tag v2-cs5-aug-gs30-untied --group-size 30
--untie-encoder`. Code hỗ trợ đầy đủ: `tevatron/src/tevatron/retriever/{arguments,modeling/encoder,modeling/dense}.py`
— 2 LoRA adapter riêng (`query`/`passage`) chia sẻ 1 backbone frozen, switch qua `set_adapter()`.

**Kết quả (Beauty, checkpoint-8000)**: NDCG@10 = **0.0054** (raw, chưa filter) — gần như random, thấp hơn cả
zero-shot (0.0135). File: `output/beauty/qwen3-embedding-0.6b-v2-cs5-aug-gs30-untied/embeddings/results/eval_test_checkpoint-8000.txt`.

**CHƯA kết luận được nguyên nhân** — chỉ chạy 1 lần, chưa debug. Khả năng: (a) bug trong adapter-switching lúc
encode passage vs query (đáng nghi nhất — nên thêm assertion/log kiểm tra đúng adapter nào active tại mỗi lúc
forward), (b) 2 adapter riêng cần nhiều data/epoch hơn để cả 2 hội tụ (mỗi adapter chỉ nhận được nửa gradient
signal so với tied), (c) learning rate 1e-4 dùng chung có thể không phù hợp khi tách 2 adapter.

**Trước khi thử lại trên máy khác**: nên debug bằng cách in ra `model.encoder.active_adapter` (hoặc tương
đương) ngay trước mỗi lần gọi encode_query/encode_passage để xác nhận đúng adapter được dùng, trước khi kết
luận "untied kém hơn tied" — hiện KHÔNG đủ cơ sở để khẳng định điều này trong paper (Related Work đang để ngỏ
câu hỏi này, không claim gì).

---

## 3. v2 instruction-based format KHÔNG đi kèm augmentation — kém hơn baseline

**Config**: `export_tevatron_v2.py` (Title/Category/Brand structured + instruction text) mà KHÔNG bật
`--augment`. Ví dụ `v2` (gs8) và `v2-gs32`.

**Kết quả (Beauty, v2-gs32)**: NDCG@10 raw = **0.0178** — kém hơn baseline v1 title-only (0.0372). Valid
selection metric cũng rất thấp (ndcg_10=0.0184 so với 0.035-0.040 của các config v1 cùng nhóm).

**Kết luận đã chốt (xem `MULTI_MACHINE_GUIDE.md` §3, quyết định 2026-07-14)**: dừng đầu tư riêng vào v2 format.
So sánh trực tiếp cho thấy v2 (khi CÓ kèm augmentation, ví dụ `v2-cs5-aug-gs20`) chỉ nhỉnh hơn v1 ~1-2%
NDCG@10 trong khi `query_max_len` phải dài hơn hẳn (256-320 vs 128) → chậm hơn rõ rệt. **Paper FADE hiện dùng
v1** (`export_tevatron.py`, title-only, không phải `export_tevatron_v2.py`) cho toàn bộ Table 2/3/4. Nếu muốn
thử lại structured format, PHẢI kèm augmentation ngay từ đầu, đừng test v2 alone.

---

## 4. Llama-3.2-1B fine-tune trên Sports — model không học được gì

**Config**: `train.sh sports meta-llama/Llama-3.2-1B` (giống cấu hình đã dùng thành công cho Beauty).

**Kết quả**: valid NDCG@10 = **0.0000** trong suốt quá trình train — model hoàn toàn không học. Chưa điều tra
nguyên nhân (LR quá cao cho Llama? data Sports khó hơn?). Beauty cùng cấu hình train bình thường (NDCG@10=0.0313).

**Kết luận**: Llama-3.2-1B nói chung đã bị loại khỏi hướng chính của project (Qwen3-Embedding tốt hơn nhiều vì
là embedding-specific model — xem README §4.0 bảng so sánh backbone). Sports-specific failure này chưa cần
điều tra thêm trừ khi có ai muốn hồi sinh nhánh Llama.

---

## 5. MovieLens-1M — KHÔNG phải kết quả xấu, chỉ là bị loại khỏi scope paper

**Lưu ý riêng — đừng nhầm với "thất bại"**: các thực nghiệm ML-1M (nhiều config: cs5/cs10/cs20, gs32/gs50,
+aug) đều cho kết quả TỐT về mặt số liệu (ví dụ `cs5-gs50-aug`: NDCG@10 raw=0.0618, full-filtered=0.1041, HR@10
full-filtered=0.1887 — cao nhất trong 3 dataset cũ). Nhưng **paper đã đổi dataset thứ 3 từ ML-1M sang Amazon
Toys and Games** (quyết định 2026-08-11, lý do: tác giả thấy ML-1M "confuse" — có thể vì domain khác hẳn
Beauty/Sports, hoặc vì thiếu metadata director/actor). Data + code ML-1M vẫn còn nguyên trong repo
(`dataset/raw/ml-1m/`, `recbole/props/ml-1m/`), nhưng **không cần chạy thêm gì cho ML-1M** trừ khi tác giả đổi
ý quay lại dùng nó.

---

## Tóm tắt — KHÔNG cần làm lại trên máy mới

| Hướng | Kết quả | Hành động khuyến nghị |
|---|---|---|
| BM25 hard negatives | Thất bại (false-negative poisoning) | Không thử lại; nếu cần hard negative, dùng embedding-space mining |
| Untied dual-encoder | Thất bại nặng, nguyên nhân chưa rõ | Debug adapter-switching trước khi thử lại, đừng chạy lại y hệt |
| v2 format (không aug) | Kém hơn baseline | Chỉ thử v2 khi có augmentation đi kèm |
| Llama-3.2-1B (Sports) | Không học được | Bỏ qua trừ khi hồi sinh nhánh Llama |
| ML-1M (toàn bộ) | Số liệu tốt nhưng ngoài scope paper | Không cần chạy thêm, không phải "thất bại" |
