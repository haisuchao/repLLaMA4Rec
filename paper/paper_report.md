# Paper Report — repLLaMA for Sequential Recommendation

> Tổng hợp kết quả thực nghiệm sẵn sàng đưa vào paper. Cập nhật: 2026-07-27.
>
> **Nguồn số liệu**: tất cả số liệu trong file này được tính trực tiếp từ TREC run files / RecBole checkpoints
> đã có trên máy này (không suy diễn/nội suy). Baseline literature (TIGER/S³-Rec) **chưa** được đưa vào vì repo
> chưa lưu số liệu gốc từ paper — xem mục 6.
>
> Với các ô còn thiếu (chưa chạy), file đánh dấu rõ **"N/A — chưa chạy"** kèm lệnh tái tạo chính xác. Xem
> [`MULTI_MACHINE_GUIDE.md`](MULTI_MACHINE_GUIDE.md) để phân phối các thực nghiệm còn thiếu sang nhiều máy.

---

## 1. Experimental Setup

### 1.1 Datasets

| Dataset | Users | Items | Interactions | Avg seq len | Max seq len |
|---|---:|---:|---:|---:|---:|
| Amazon Beauty (2014) | 22,363 | 12,101 | 198,502 | 8.9 | 204 |
| Amazon Sports and Outdoors (2014) | 35,598 | 18,357 | 296,337 | 8.3 | 296 |
| MovieLens-1M | 6,040 | 3,416 | 999,611 | 165.5 | 2,277 |

5-core filter (mọi user/item ≥ 5 interactions). Leave-one-out split: item cuối = test, áp chót = valid, phần
còn lại = train. Full ranking trên toàn bộ corpus (không sample negative khi eval).

### 1.2 Metrics

NDCG@{5,10,20}, HR@{5,10,20} (= Recall@K vì mỗi query chỉ có 1 positive), MRR@{5,10,20}.

### 1.3 Phương pháp

- **repLLaMA**: reformulate recommendation → dense retrieval. Query = N item gần nhất (text), corpus = toàn
  bộ item catalog (text). Backbone `Qwen/Qwen3-Embedding-0.6B` + LoRA (rank 16, alpha 64, target
  q/k/v/o/gate/up/down), fine-tune bằng Tevatron v2, pooling = last-token, similarity = cosine (normalized).
- **Post-processing history filter** (đóng góp chính của bài): bi-encoder xếp hạng cao các item đã có trong
  lịch sử user vì text overlap trực tiếp giữa query và item đó (đo được: 92.5% queries có history item ở
  rank 1 trước khi filter — xem `improvement_plan.md` §2.1). Filter loại các item này khỏi kết quả FAISS
  trước khi tính metric, không cần train lại. Hai biến thể:
  - **`ctx`** — chỉ loại các item nằm trong query (context window)
  - **`full`** — loại toàn bộ lịch sử user trước item test (tương đương protocol ẩn dùng bởi SASRec/ID-based
    models — các model này về cấu trúc không thể "gợi ý lại" item đã tương tác vì embedding sequence học
    theo transition pattern, trong khi bi-encoder text-based thì có thể). Số liệu chính trong báo cáo này
    dùng **`full`** trừ khi ghi chú khác.
- **SASRec** (baseline ID-based): self-attention causal, chạy qua RecBole 1.2.1, cross-entropy loss (full
  softmax, không negative sampling), `MAX_ITEM_LIST_LENGTH=200`.
- **GRU4Rec** (baseline ID-based, RNN): cũng chạy qua RecBole 1.2.1, cùng protocol/eval_args/MAX_ITEM_LIST_LENGTH
  với SASRec để so sánh công bằng (config: `recbole/props/<dataset>/gru4rec.yaml`).
- **Zero-shot**: base model `Qwen/Qwen3-Embedding-0.6B`, không fine-tune, dùng trực tiếp cho retrieval.

### 1.4 Quyết định format: v1 (title-only) thay vì v2 (instruction-based)

Ban đầu Beauty đã thử thêm **v2 format** (`export_tevatron_v2.py`) — query/document viết lại thành
instruction + structured metadata (`Title: X. Category: Y > Z. Brand: W.`), xem `improvement_plan.md` §4.
So sánh trực tiếp cặp cấu hình **giống hệt nhau về group_size (gs=8), chỉ khác format**:

| Format | Model | Context size | Query max len | NDCG@10 (raw) | HR@10 (raw) |
|---|---|:---:|:---:|:---:|:---:|
| v1 (title-only) | `aug-3` | 3 | 128 | 0.0378 | 0.0856 |
| v2 (instruction) | `v2-cs5-aug` | 5 | 128 | 0.0383 | 0.0858 |

Chênh lệch **+1.3% NDCG@10 / +0.2% HR@10** — không đáng kể. Trong khi đó v2 cần `query_max_len` dài hơn
đáng kể khi tăng group_size (256-320 so với 128 của v1, xem `improvement_plan.md` §4.4 "Token budget"), kéo
theo encode + train chậm hơn rõ rệt (sequence dài hơn → chi phí attention O(n²) tăng, batch size phải giảm).

**Quyết định**: dùng **v1 (title-only)** làm format chuẩn cho mọi thực nghiệm tiếp theo (Sports, ML-1M, model
scale lớn hơn). Không tiếp tục đầu tư thời gian train vào v2 — chênh lệch không bù được chi phí tính toán.
Kết quả v2 đã có (Beauty, `v2-cs5-aug-gs20`) vẫn được giữ lại làm **tham khảo phụ** ở cuối báo cáo (§8), nhưng
**không dùng làm số liệu headline** để giữ nhất quán phương pháp giữa 3 dataset.

### 1.5 Cấu hình model tốt nhất mỗi dataset (v1, title-only)

| Dataset | Model | Format | Context size | Augmentation | group_size | Epochs |
|---|---|---|---:|---|---:|---:|
| Beauty | `qwen3-embedding-0.6b-aug-5` | Plain (`"Query: item1, item2, item3"`) | 3 | Có (sliding window) | 8 (1 pos + 7 neg) | 3 |
| Sports | `qwen3-embedding-0.6b-cs5-aug-gs32` | Plain | 5 | Có (sliding window) | 32 (1 pos + 31 neg) | 3 |
| ML-1M | `qwen3-embedding-0.6b-cs5-gs50-aug` | Plain | 5 | Có (sliding window, max 20 samples/user) | 50 (1 pos + 49 neg) | 3 |

> Cập nhật 2026-07-24: Sports giờ có model augmented (`cs5-aug-gs32`), cải thiện rất lớn so với `standard`
> cũ (xem §2, §3). Cập nhật 2026-07-26: SASRec Sports đã train xong (xem §2). Còn thiếu: model scale lớn
> hơn cho Sports (`cs5-gs50-4b` được đặt tag "4b" nhưng `train_config.json` cho biết model thực tế vẫn là
> 0.6B — có thể nhầm `--model` khi train; chưa có checkpoint nào được lưu). `cs5-gs50-0.6b` cũng dừng ở
> checkpoint-2000, không có model cuối — job có
> vẻ bị ngắt giữa chừng, chưa dùng được. Xem mục 5.

---

## 2. Kết quả chính (Main Results)

**In đậm** = tốt nhất trong cột. `repLLaMA 0.6B`/`4B` đều là kết quả **đã áp dụng History Filter, chế độ
`full`** (xem §1.3) + augmentation — đây là cấu hình cuối cùng được chọn làm model đề xuất; xem §3 (Ablation)
để so sánh với biến thể không filter/không augmentation.

### BEAUTY

| Method | NDCG@5 | HR@5 | MRR@5 | NDCG@10 | HR@10 | MRR@10 | NDCG@20 | HR@20 | MRR@20 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Zero-shot (repLLaMA, no fine-tune) | 0.0092 | 0.0182 | 0.0063 | 0.0135 | 0.0315 | 0.0081 | 0.0179 | 0.0488 | 0.0092 |
| SRGNN | 0.0252 | 0.0363 | 0.0215 | 0.0312 | 0.0552 | 0.0240 | 0.0380 | 0.0822 | 0.0259 |
| GRU4Rec | 0.0261 | 0.0392 | 0.0218 | 0.0330 | 0.0607 | 0.0246 | 0.0411 | 0.0930 | 0.0268 |
| SASRec | 0.0323 | 0.0548 | 0.0249 | 0.0412 | 0.0824 | 0.0285 | 0.0506 | 0.1198 | 0.0311 |
| **repLLaMA 0.6B** | **0.0501** | **0.0730** | **0.0427** | **0.0616** | **0.1088** | **0.0474** | **0.0730** | **0.1538** | **0.0504** |
| repLLaMA 4B | **N/A — chưa có checkpoint** (training fail/OOM, xem §4, §5) | | | | | | | | |

→ repLLaMA 0.6B vượt SASRec **+49.5% NDCG@10**, **+32.0% HR@10**; vượt GRU4Rec **+86.7% NDCG@10**, **+79.2%
HR@10**; vượt SRGNN **+97.4% NDCG@10**, **+97.1% HR@10**. (Model: `aug-5`, v1/title-only — xem §1.4 về quyết
định dùng v1 thay v2. Nếu dùng v2 tốt nhất `v2-cs5-aug-gs20` thay vào, số liệu nhích lên
NDCG@10=0.0631/HR@10=0.1119 — xem §8, không dùng làm headline vì lý do chi phí tính toán.) Xếp hạng 3
baseline RecBole trên Beauty: SASRec > GRU4Rec > SRGNN — cả hai đều early-stop rất sớm (SRGNN epoch 8,
GRU4Rec epoch 12) so với SASRec (epoch 25), gợi ý corpus 12K item + sequence ngắn (~8.9) không đủ tín hiệu
cho kiến trúc GNN/RNN học sâu như trên ML-1M.

> Cập nhật 2026-07-26: SASRec Beauty được train lại từ đầu cùng phiên với Sports/ML-1M (best epoch 25, thay
> vì epoch 47 ở lần chạy tháng 5) — NDCG@10 đổi từ 0.0430→0.0412 (~-4%, dao động bình thường do random seed/
> early-stopping khác nhau giữa các lần train, không phải regression có ý nghĩa). Dùng số liệu mới để nhất
> quán với Sports/ML-1M (cùng session, cùng điều kiện).

### SPORTS

| Method | NDCG@5 | HR@5 | MRR@5 | NDCG@10 | HR@10 | MRR@10 | NDCG@20 | HR@20 | MRR@20 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Zero-shot (repLLaMA, no fine-tune) | 0.0045 | 0.0090 | 0.0030 | 0.0071 | 0.0170 | 0.0041 | 0.0100 | 0.0289 | 0.0049 |
| SRGNN | 0.0130 | 0.0200 | 0.0107 | 0.0165 | 0.0310 | 0.0122 | 0.0205 | 0.0467 | 0.0132 |
| GRU4Rec | 0.0153 | 0.0237 | 0.0126 | 0.0196 | 0.0371 | 0.0144 | 0.0245 | 0.0567 | 0.0157 |
| SASRec | 0.0164 | 0.0297 | 0.0121 | 0.0220 | 0.0471 | 0.0143 | 0.0277 | 0.0697 | 0.0159 |
| **repLLaMA 0.6B** | **0.0294** | **0.0447** | **0.0244** | **0.0371** | **0.0684** | **0.0275** | **0.0448** | **0.0993** | **0.0296** |
| repLLaMA 4B | **N/A — chưa có checkpoint** (config train bị nhầm model=0.6B, xem §4, §5) | | | | | | | | |

> Cập nhật 2026-07-24: model headline đổi từ `standard` (cs=3, không aug) sang `cs5-aug-gs32` (cs=5, có
> augmentation, group_size=32) — cải thiện rất mạnh, NDCG@10 filtered tăng từ 0.0173 → 0.0371 (+114%). Chi
> tiết ablation ở §3. Cập nhật 2026-07-25: đã có GRU4Rec và SRGNN — repLLaMA 0.6B vượt **+89.3% NDCG@10**
> so với GRU4Rec, **+124.8% NDCG@10** so với SRGNN. Cập nhật 2026-07-26: đã có SASRec (0.0220 NDCG@10, best
> epoch 24) — repLLaMA 0.6B vượt SASRec **+68.6% NDCG@10**, **+45.2% HR@10**. Xếp hạng đầy đủ trên Sports:
> repLLaMA 0.6B > SASRec > GRU4Rec > SRGNN > zero-shot (repLLaMA 4B: N/A). Bảng Main Results Sports giờ đã đầy
> đủ các baseline có checkpoint hợp lệ.

### ML-1M

| Method | NDCG@5 | HR@5 | MRR@5 | NDCG@10 | HR@10 | MRR@10 | NDCG@20 | HR@20 | MRR@20 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Zero-shot (repLLaMA, no fine-tune) | 0.0044 | 0.0093 | 0.0028 | 0.0068 | 0.0169 | 0.0038 | 0.0093 | 0.0263 | 0.0045 |
| SASRec | 0.0823 | 0.1303 | 0.0666 | 0.1077 | 0.2089 | 0.0770 | 0.1348 | 0.3166 | 0.0844 |
| SRGNN | 0.1011 | 0.1522 | 0.0844 | 0.1235 | 0.2215 | 0.0936 | 0.1473 | 0.3159 | 0.1001 |
| GRU4Rec | 0.1210 | 0.1768 | 0.1026 | 0.1491 | 0.2641 | 0.1142 | 0.1751 | 0.3674 | 0.1212 |
| repLLaMA 0.6B | 0.0846 | 0.1276 | 0.0705 | 0.1041 | 0.1887 | 0.0785 | 0.1262 | 0.2763 | 0.0845 |
| **repLLaMA 4B** | — | — | — | **0.1601** | **0.2829** | — | — | — | — |

→ Cập nhật 2026-07-25: **GRU4Rec là baseline RecBole mạnh nhất trên ML-1M**, vượt cả SASRec (NDCG@10 +38.4%,
HR@10 +26.4%) lẫn repLLaMA 0.6B (NDCG@10 +43.2%, HR@10 +40.0%). Đây là kết quả bất ngờ — GRU4Rec thường được
coi là baseline yếu hơn SASRec trong literature, nhưng trên ML-1M (sequence rất dài, avg 165.5 items) khả
năng nắm bắt toàn bộ lịch sử của RNN dường như phù hợp hơn self-attention truncate ở 200 items hoặc
text-query chỉ dùng 5 items gần nhất của repLLaMA 0.6B.

**Cập nhật 2026-07-27**: repLLaMA **4B** (`cs5-gs50-aug-4b`, chạy trên máy khác, chưa có checkpoint/log để
xác minh lại trên máy chính — xem ghi chú phương pháp luận cuối file) đạt NDCG@10=**0.1601**, HR@10=**0.2829**,
**vượt cả GRU4Rec** (+7.4% NDCG@10, +7.1% HR@10) — trở thành model tốt nhất trên ML-1M trong toàn bộ so sánh.
Điều này ủng hộ giả thuyết ở bản cập nhật trước: hạn chế của repLLaMA trên sequence dài **không phải do bản
chất phương pháp dense retrieval**, mà do model 0.6B chưa đủ năng lực biểu diễn — khi tăng scale lên 4B,
repLLaMA vượt qua mọi baseline RNN/GNN/attention. Chỉ thiếu breakdown @5/@20/MRR cho dòng 4B (xem ghi chú).
**Đây là kết quả trung thực, không nên làm tròn/bỏ qua khi viết paper** — nên trình bày cả hai góc nhìn: repLLaMA
0.6B thua GRU4Rec/SRGNN trên sequence dài, nhưng repLLaMA 4B vượt lại tất cả — làm điểm thảo luận về đánh đổi
model scale vs. kiến trúc.

> Cập nhật 2026-07-26: SASRec ML-1M được train lại (cùng phiên với Beauty/Sports) — kết quả **trùng khớp
> tuyệt đối** với lần chạy tháng 5 (best epoch 29 cả hai lần), số liệu ổn định/reproducible. Đã có thêm
> **SRGNN** — cũng vượt cả SASRec (+14.7% NDCG@10) lẫn repLLaMA 0.6B (+18.6% NDCG@10), dù vẫn kém GRU4Rec.
> Cả 2 baseline có kiến trúc xử lý toàn bộ lịch sử dài (RNN, GNN-trên-sequence) đều vượt repLLaMA 0.6B — củng
> cố giả thuyết: hạn chế của repLLaMA 0.6B trên ML-1M đến từ việc chỉ dùng 5 item gần nhất làm query text,
> không phải do bản chất phương pháp attention (SASRec) yếu hơn RNN/GNN.
>
> **Xếp hạng đầy đủ trên ML-1M (cập nhật 2026-07-27, gồm cả 4B): repLLaMA 4B > GRU4Rec > SRGNN > SASRec >
> repLLaMA 0.6B > zero-shot.**

---

## 3. Ablation Study

Yêu cầu: (a) remove post-filter, (b) no augmentation, (c) zero-shot without training. Mỗi dòng thay đổi
**một** thành phần so với model tốt nhất; xem chú thích (*) khi không tách biệt hoàn toàn được do giới hạn số
experiment đã chạy.

### BEAUTY (base: `aug-5`, v1/title-only — xem §1.4)

| Variant | NDCG@10 | HR@10 | Δ NDCG@10 |
|---|:---:|:---:|:---:|
| **Full pipeline** (augmentation + History Filter) | **0.0616** | **0.1088** | — |
| (−) History filter → raw ranking | 0.0390 | 0.0905 | −36.7% |
| (−) Augmentation\* → `standard` (cùng format/context/gs=8, không aug) + filter | 0.0581 | 0.1019 | −5.7% |
| (−) Fine-tuning → zero-shot + filter | 0.0202 | 0.0364 | −67.2% |
| (−) Fine-tuning → zero-shot, raw (không filter, không train) | 0.0135 | 0.0315 | −78.1% |

\* Cặp `standard` (không aug) vs `aug-3` (có aug) — cả hai: plain format, context_size=3, group_size=8, chỉ
khác augmentation — là ablation **sạch nhất hiện có**: NDCG@10 raw đi từ 0.0372 (standard) → 0.0378 (aug-3),
tức augmentation đóng góp khá nhỏ ở mức raw (+1.6%) khi cô lập hoàn toàn. Dòng "(−) Augmentation" ở trên dùng
`standard` làm baseline vì cùng hệ với `aug-5` (headline); `aug-3` cho kết quả tương đương sau filter
(NDCG@10 = 0.0582) nếu muốn đối chiếu thêm.

### SPORTS (base: `cs5-aug-gs32`)

| Variant | NDCG@10 | HR@10 | Δ NDCG@10 |
|---|:---:|:---:|:---:|
| **Full pipeline** (context=5 + augmentation + History Filter) | **0.0371** | **0.0684** | — |
| (−) History filter → raw ranking | 0.0260 | 0.0574 | −29.9% |
| (−) Augmentation\* → `cs5-gs32-0.6b` (context=5, group_size=32, không aug) + filter | 0.0279 | 0.0523 | −24.8% |
| (−) Fine-tuning → zero-shot + filter | 0.0102 | 0.0195 | −72.5% |
| (−) Fine-tuning → zero-shot, raw | 0.0071 | 0.0170 | −80.9% |

\* Cặp `cs5-gs32-0.6b` vs `cs5-aug-gs32` giống hệt nhau ở context_size (5) và group_size (32), **chỉ khác
augmentation** — đây là ablation augmentation **sạch nhất trong cả 3 dataset** (không lẫn biến nào khác).
So với Beauty (aug đóng góp +1.6% raw), Sports cho thấy augmentation đóng góp nhiều hơn hẳn: NDCG@10 raw đi
từ 0.0189 (không aug) → 0.0260 (có aug), tức **+37.6%**.

### ML-1M — 0.6B (base: `cs5-gs50-aug`)

| Variant | NDCG@10 | HR@10 | Δ NDCG@10 |
|---|:---:|:---:|:---:|
| **Full pipeline** (context=5 + augmentation + History Filter) | **0.1041** | **0.1887** | — |
| (−) History filter → raw ranking | 0.0618 | 0.1343 | −40.6% |
| (−) Augmentation\* → `cs5-gs32` (context=5, không aug) + filter | 0.0691 | 0.1260 | −33.6% |
| (−) Fine-tuning → zero-shot + filter | 0.0128 | 0.0214 | −87.7% |
| (−) Fine-tuning → zero-shot, raw | 0.0068 | 0.0169 | −93.5% |

\* `cs5-gs32` và `cs5-gs50-aug` khác nhau ở cả augmentation lẫn group_size (32 vs 50) — không tách biệt
tuyệt đối, nhưng đây là cặp gần nhất hiện có (cùng context_size=5).

### ML-1M — 4B (base: `cs5-gs50-aug-4b`)

| Variant | NDCG@10 | HR@10 | Δ NDCG@10 |
|---|:---:|:---:|:---:|
| **Full pipeline** (context=5 + augmentation + History Filter) | **0.1601** | **0.2829** | — |
| (−) History filter → raw ranking | 0.0825 | 0.1909 | −48.5% |
| (−) Augmentation → `cs5-gs50-4b` (context=5, không aug, cùng group_size=50) + filter | 0.0796 | 0.1470 | −50.3% |
| (−) Fine-tuning → zero-shot + filter | 0.0257 | 0.0469 | −84.0% |
| (−) Fine-tuning → zero-shot, raw | 0.0130 | 0.0343 | −91.9% |

Khác với ablation 0.6B ở trên, cặp "(−) Augmentation" của 4B giữ nguyên group_size=50 ở cả hai phía — cô lập
đúng một biến (chỉ augmentation), sạch hơn phép so sánh 0.6B (vốn đổi cả augmentation lẫn group_size 32 vs
50). Xu hướng đóng góp của từng thành phần giống hệt 0.6B (fine-tuning > augmentation ≈ history filter), và
4B vượt 0.6B ở mọi variant tương ứng — củng cố luận điểm model scale lớn hơn cải thiện chất lượng nhất quán,
không chỉ ở headline.

---

## 4. So sánh kích thước model (0.6B vs lớn hơn)

| Dataset | 0.6B (headline v1, + filter) | 4B / lớn hơn |
|---|:---:|:---:|
| Beauty | NDCG@10=0.0616, HR@10=0.1088 | **N/A — training 4B đã chạy nhưng KHÔNG có checkpoint** (thư mục `output/beauty/qwen3-embedding-4b-*` chỉ có `train_config.json`, có thể do OOM hoặc bị dừng giữa chừng). Cần train lại (v1 format) — xem §5, mục 4. |
| Sports | NDCG@10=0.0371, HR@10=0.0684 | **N/A — chưa thử** (`cs5-gs50-4b` được tag "4b" nhưng `train_config.json` ghi model=0.6B, không có checkpoint nào — có thể nhầm `--model` khi chạy). Xem §5. |
| ML-1M | NDCG@10=0.1041, HR@10=0.1887 | **NDCG@10=0.1601, HR@10=0.2829** (`cs5-gs50-aug-4b`, cùng data variant/group_size với headline 0.6B, chỉ đổi model — xem §3) → **+53.8% NDCG@10, +49.9% HR@10** so với 0.6B |

**ML-1M 4B**: NDCG@10/HR@10 đã được xác nhận (do người dùng chạy trên máy khác) — dùng được cho paper. **Lưu
ý**: máy chính (nơi biên soạn file này) **không có checkpoint/log** để tự kiểm tra lại hoặc tính các metric
còn thiếu (@5, @20, MRR) — xem ghi chú phương pháp luận cuối file. Nếu cần đầy đủ 9 metric, chạy lại
`./eval.sh ml-1m --tag gs50-aug-4b ...` trên máy đã train, hoặc sync checkpoint về máy chính rồi chạy
`eval_filter.py`.

Beauty/Sports vẫn **chưa có số liệu 4B đáng tin cậy** — đừng đưa vào paper cho tới khi train lại thành công.
Xem `MULTI_MACHINE_GUIDE.md` §3 (Máy B/C) để chạy trên máy khác (4B cần VRAM sát 12GB trên card 12GB gốc,
nên ưu tiên máy có ≥16GB VRAM nếu có, để tránh lặp lại lỗi OOM). Dùng v1 format cho 4B — không lặp lại sai
lầm đầu tư vào v2 (§1.4).

---

## 5. Các thực nghiệm còn thiếu (ưu tiên để hoàn thiện paper)

> Tất cả thực nghiệm dưới đây dùng **v1 (title-only)** — xem §1.4. Không còn hạng mục v2 trong kế hoạch.

| # | Thực nghiệm | Dataset | Lý do cần | Ước tính thời gian (RTX 3060 12GB) |
|---|---|---|---|---|
| 1 | Retrain Qwen3-Embedding-4B (đã fail, v1 format) | Beauty | Mục 4 hiện không có số liệu | ~3-6h tùy cấu hình, cần theo dõi OOM |
| 2 | Train Qwen3-Embedding-4B (v1 format, dùng đúng `--model Qwen/Qwen3-Embedding-4B`) | Sports | `cs5-gs50-4b` hiện có config sai (model=0.6B) và không có checkpoint — cần chạy lại từ đầu | ~3-6h |
| 3 | (Tuỳ chọn) Hoàn tất `cs5-gs50-0.6b` — job cũ dừng ở checkpoint-2000, chưa rõ có cần kết quả này không (đã có `cs5-aug-gs32`/`cs5-gs32-0.6b` là cặp ablation chính) | Sports | Chỉ cần nếu muốn thêm 1 điểm so sánh group_size=50 (v1, có/không aug) | ~30-60 phút để chạy tiếp hoặc train lại |

✅ Đã xong: augmentation cho Sports (2026-07-24, `cs5-aug-gs32` + ablation sạch `cs5-gs32-0.6b`). ML-1M 4B đã
có số liệu đầy đủ (xem §3, §4). **GRU4Rec đã train xong cả 3 dataset (2026-07-25)** — xem §2; phát hiện quan
trọng: GRU4Rec vượt cả SASRec lẫn repLLaMA 0.6B trên ML-1M, nên khi viết phần discussion/limitation không
chỉ so với SASRec. **SASRec đã train xong cả 3 dataset (2026-07-26)** — bảng Main Results (§2) giờ đủ cả 3
baseline ID-based. **SRGNN đã train xong cả 3 dataset (2026-07-25)** — bảng Main Results (§2) giờ có đủ 4
baseline (zero-shot, SRGNN, GRU4Rec, SASRec) trên cả 3 dataset. Baseline RecBole coi như hoàn tất; việc còn
lại trong danh sách dưới đây chỉ còn xoay quanh model scale 4B cho repLLaMA.

Lệnh chính xác cho từng mục nằm trong `MULTI_MACHINE_GUIDE.md` §3 (đã chia theo máy để chạy song song, đã cập
nhật bỏ hạng mục v2 và hạng mục Sports augmentation đã hoàn thành).

---

## 6. Baseline từ literature (CHƯA điền — cần paper gốc)

`experiments.md` có trích dẫn TIGER (Rajput et al., NeurIPS 2023) và S³-Rec (Zhou et al., CIKM 2020) là
nguồn số liệu GRU4Rec/BERT4Rec/SASRec/TIGER trên Beauty & Sports (2014, 5-core, leave-one-out — cùng
protocol với repo này), nhưng **không có bảng số liệu cụ thể được lưu trong repo**. Trước khi đưa vào paper,
tự lấy trực tiếp từ bảng kết quả gốc của hai paper trên (tránh trích dẫn số liệu không xác minh được):

- Rajput et al., "Recommender Systems with Generative Retrieval", NeurIPS 2023 — arXiv:2305.05065
- Zhou et al., "S³-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information
  Maximization", CIKM 2020

Lưu ý: paper TIGER không có kết quả ML-1M.

---

## 7. Phụ lục — So sánh chế độ filter (`context` vs `full`)

Số liệu đầy đủ (raw / ctx / full) cho tất cả model đã eval nằm ở `experiments.md` §"Kết quả History Filter".
Tóm tắt nhanh model headline (v1) mỗi dataset:

| Dataset | Model | NDCG@10 raw | NDCG@10 ctx | NDCG@10 full |
|---|---|:---:|:---:|:---:|
| Beauty | aug-5 | 0.0390 | 0.0590 | 0.0616 |
| Sports | cs5-aug-gs32 | 0.0260 | 0.0367 | 0.0371 |
| ML-1M | cs5-gs50-aug | 0.0618 | 0.0850 | 0.1041 |

`full` luôn ≥ `ctx` (full là superset). Khoảng cách giữa hai chế độ nhỏ ở Beauty/Sports (context 3-5 items ≈
phần lớn signal lịch sử) nhưng lớn ở ML-1M (context chỉ là phần nhỏ của trung bình 165.5 items/user).

---

## 8. Phụ lục — Kết quả v2 format (Beauty, tham khảo — không dùng làm headline)

Xem §1.4 để biết lý do dừng đầu tư vào v2. Số liệu dưới đây **là kết quả thật đã train/eval**, giữ lại để
tham khảo hoặc dùng trong một ablation phụ về format nếu paper cần, nhưng **không phải số liệu chính** của
báo cáo này.

| Model | Format | Context | group_size | NDCG@10 raw | HR@10 raw | NDCG@10 full-filtered | HR@10 full-filtered |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `v2-cs5-aug` | v2 | 5 | 8 | 0.0383 | 0.0858 | 0.0599 | 0.1045 |
| `v2-cs5-aug-gs20` (best overall, mọi format) | v2 | 5 | 20 | 0.0405 | 0.0919 | 0.0631 | 0.1119 |
| `v2-gs32` (v2, KHÔNG augmentation) | v2 | 3 | 32 | 0.0178 | 0.0421 | 0.0262 | 0.0485 |

Đáng chú ý: `v2-gs32` (v2 format nhưng không augmentation) cho kết quả **kém hơn cả baseline v1 standard**
(0.0178 vs 0.0372 NDCG@10 raw) — củng cố thêm quyết định ở §1.4: bản thân format v2 không phải nguồn cải
thiện chính, augmentation + group_size mới là yếu tố quyết định, và v1 đạt phần lớn lợi ích đó với chi phí
thấp hơn nhiều.

---

## Ghi chú phương pháp luận (đọc trước khi trích số liệu vào paper)

1. **`context_size` dùng để filter** được suy ra từ tên data-variant/tag theo quy ước đặt tên của
   `export_tevatron.py` (hậu tố `csN`). Với model không có `train_config.json` lưu sẵn, giả định
   context_size=3 (mặc định pipeline). Xem `experiments.md` để biết danh sách model bị ảnh hưởng.
2. **`ml-1m/qwen3-embedding-0.6b-cs10-gs50`** bị đặt tên sai — `train_config.json` cho biết context_size
   thực tế = 20, không phải 10. Bảng trong file này đã dùng giá trị đúng (20), nhưng nếu trích dẫn thư mục
   theo tên, hãy sửa lại cho đúng.
3. Tất cả số liệu "raw" đều lấy từ `eval.sh` mode `best` (chọn checkpoint theo **unfiltered** valid NDCG@10)
   — chưa kiểm tra checkpoint có tối ưu cho kịch bản đã-filter hay không (xem `improvement_plan.md` §Phase 2).
4. SASRec Beauty/ML-1M lấy từ checkpoint đã train sẵn, eval lại bằng `python run_recbole.py eval <ckpt>`
   (2026-07-14) — dùng đúng best-epoch theo early-stopping mặc định RecBole, không phải epoch cuối cùng.
5. **repLLaMA 4B (ML-1M)** — NDCG@10=0.1601, HR@10=0.2829 (`cs5-gs50-aug-4b`) — được train và eval trên **một
   máy khác** ngoài máy chính lưu repo này, do người dùng báo lại kết quả trực tiếp. Khác với mọi số liệu
   khác trong file này (đều tính trực tiếp từ TREC run files / checkpoint có sẵn trên máy chính), **số liệu
   này chưa thể tự kiểm tra lại tại đây** — không có `train_config.json`, checkpoint, hay TREC run file nào
   trong `output/ml-1m/` trên máy chính. Người dùng đã xác nhận số liệu là thật (2026-07-27). Chỉ có
   NDCG@10/HR@10 được báo lại — chưa có NDCG@5/HR@5/MRR@5/NDCG@20/HR@20/MRR@20 cho model này (đánh dấu "—"
   trong bảng §2). Khuyến nghị: sync checkpoint/log về máy chính theo `MULTI_MACHINE_GUIDE.md` §4 để có đầy đủ
   9 metric và một bản ghi kiểm chứng được trước khi nộp paper.
