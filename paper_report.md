# Paper Report — repLLaMA for Sequential Recommendation

> Tổng hợp kết quả thực nghiệm sẵn sàng đưa vào paper. Cập nhật: 2026-07-14.
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

NDCG@{5,10,20}, HR@{5,10,20} (= Recall@K vì mỗi query chỉ có 1 positive), MRR@10.

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
| Sports | `qwen3-embedding-0.6b` (standard) — **chưa có augmented variant** | Plain | 3 | Không | 50 (1 pos + 49 neg) | 3 |
| ML-1M | `qwen3-embedding-0.6b-cs5-gs50-aug` | Plain | 5 | Có (sliding window, max 20 samples/user) | 50 (1 pos + 49 neg) | 3 |

> Sports hiện chỉ có 1 model đã train (standard) — chưa có augmentation để so sánh. Đây là gap ưu tiên cao
> nhất cần lấp — xem mục 5 và `MULTI_MACHINE_GUIDE.md`.

---

## 2. Kết quả chính (Main Results)

**In đậm** = tốt nhất trong cột. `repLLaMA + Filter` dùng chế độ `full` (xem §1.3).

### BEAUTY

| Method | NDCG@5 | HR@5 | NDCG@10 | HR@10 | NDCG@20 | HR@20 | MRR@10 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Zero-shot (repLLaMA, no fine-tune) | 0.0092 | 0.0182 | 0.0135 | 0.0315 | 0.0179 | 0.0488 | 0.0081 |
| SASRec | 0.0338 | 0.0558 | 0.0430 | 0.0842 | 0.0523 | 0.1213 | 0.0303 |
| repLLaMA (raw, no filter) | 0.0264 | 0.0509 | 0.0390 | 0.0905 | 0.0517 | 0.1409 | 0.0235 |
| **repLLaMA + History Filter** | **0.0501** | **0.0730** | **0.0616** | **0.1088** | **0.0730** | **0.1538** | **0.0474** |

→ repLLaMA+Filter vượt SASRec **+43.3% NDCG@10**, **+29.2% HR@10**. (Model: `aug-5`, v1/title-only — xem §1.4
về quyết định dùng v1 thay v2. Nếu dùng v2 tốt nhất `v2-cs5-aug-gs20` thay vào, số liệu nhích lên
NDCG@10=0.0631/HR@10=0.1119 — xem §8, không dùng làm headline vì lý do chi phí tính toán.)

### SPORTS

| Method | NDCG@5 | HR@5 | NDCG@10 | HR@10 | NDCG@20 | HR@20 | MRR@10 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Zero-shot (repLLaMA, no fine-tune) | 0.0045 | 0.0090 | 0.0071 | 0.0170 | 0.0100 | 0.0289 | 0.0041 |
| SASRec | **N/A — chưa train** (xem §5, `MULTI_MACHINE_GUIDE.md`) | | | | | | |
| repLLaMA (raw, no filter) | 0.0080 | 0.0155 | 0.0116 | 0.0266 | 0.0153 | 0.0409 | 0.0071 |
| **repLLaMA + History Filter** | **0.0142** | **0.0212** | **0.0173** | **0.0309** | **0.0207** | **0.0447** | **0.0131** |

> repLLaMA Sports mới chỉ có config "standard" (không augmentation) — con số này thấp hơn tiềm năng thực,
> tương tự Beauty trước khi thêm augmentation (`aug-5`) đã cải thiện đáng kể. Xem gap trong §5.

### ML-1M

| Method | NDCG@5 | HR@5 | NDCG@10 | HR@10 | NDCG@20 | HR@20 | MRR@10 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Zero-shot (repLLaMA, no fine-tune) | 0.0044 | 0.0093 | 0.0068 | 0.0169 | 0.0093 | 0.0263 | 0.0038 |
| SASRec | **0.0823** | **0.1303** | **0.1077** | **0.2089** | **0.1348** | **0.3166** | **0.0770** |
| repLLaMA (raw, no filter) | 0.0430 | 0.0762 | 0.0618 | 0.1343 | 0.0821 | 0.2147 | 0.0399 |
| repLLaMA + History Filter | 0.0846 | 0.1276 | 0.1041 | 0.1887 | 0.1262 | 0.2763 | 0.0785 |

→ Trên ML-1M, SASRec vẫn nhỉnh hơn repLLaMA+Filter (NDCG@10: 0.1077 vs 0.1041, HR@10: 0.2089 vs 0.1887).
Sequence dài (avg 165.5 items) là lợi thế tự nhiên cho ID-based sequential model; repLLaMA chỉ dùng
context_size=5 items text làm query nên mất phần lớn thông tin lịch sử. **Đây là kết quả trung thực, không
nên làm tròn/bỏ qua khi viết paper** — có thể dùng làm điểm thảo luận về hạn chế (limitation) của phương
pháp text-based với sequence dài.

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

### SPORTS (base: `standard`, dataset chưa có augmented/v2 model)

| Variant | NDCG@10 | HR@10 | Δ NDCG@10 |
|---|:---:|:---:|:---:|
| **Full pipeline hiện có** (standard + History Filter) | **0.0173** | **0.0309** | — |
| (−) History filter → raw ranking | 0.0116 | 0.0266 | −32.9% |
| (−) Augmentation | **N/A — chưa có model augmented cho Sports** | | |
| (−) Fine-tuning → zero-shot + filter | 0.0102 | 0.0195 | −41.0% |
| (−) Fine-tuning → zero-shot, raw | 0.0071 | 0.0170 | −59.0% |

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
| Sports | NDCG@10=0.0173, HR@10=0.0309 | N/A — chưa thử |
| ML-1M | NDCG@10=0.1041, HR@10=0.1887 | **NDCG@10=0.1601, HR@10=0.2829** (`cs5-gs50-aug-4b`, cùng data variant/group_size với headline 0.6B, chỉ đổi model — xem §3) → **+53.8% NDCG@10, +49.9% HR@10** so với 0.6B |

**ML-1M đã có số liệu 4B đáng tin cậy** (train thành công, đủ checkpoint, đã qua History Filter — xem §3).
Beauty/Sports vẫn **chưa có số liệu 4B đáng tin cậy** — đừng đưa vào paper cho tới khi train lại thành công.
Xem `MULTI_MACHINE_GUIDE.md` §3 (Máy B/C) để chạy trên máy khác (4B cần VRAM sát 12GB trên card 12GB gốc,
nên ưu tiên máy có ≥16GB VRAM nếu có, để tránh lặp lại lỗi OOM). Dùng v1 format cho 4B — không lặp lại sai
lầm đầu tư vào v2 (§1.4).

---

## 5. Các thực nghiệm còn thiếu (ưu tiên để hoàn thiện paper)

> Tất cả thực nghiệm dưới đây dùng **v1 (title-only)** — xem §1.4. Không còn hạng mục v2 trong kế hoạch.

| # | Thực nghiệm | Dataset | Lý do cần | Ước tính thời gian (RTX 3060 12GB) |
|---|---|---|---|---|
| 1 | Train SASRec | Sports | Chưa có baseline ID-based cho Sports — bảng Main Results đang thiếu 1 dòng | ~2-4h (35,598 users, tương tự scale Beauty×1.6) |
| 2 | Export augmented data (v1) + train + eval | Sports | Không có ablation "augmentation" cho Sports; repLLaMA Sports đang dùng config yếu nhất (standard) | Export: vài phút (CPU). Train: ~1-2h |
| 3 | Train v1 augmented + group_size cao hơn (gs20/32, giống pattern `aug-5`/Beauty) | Sports | Nhân bản pattern đã thành công ở Beauty (aug + gs tăng là nguồn cải thiện chính, rẻ hơn v2 nhiều) | ~1-2h (dùng lại data ở mục 2) |
| 4 | Retrain Qwen3-Embedding-4B (đã fail, v1 format) | Beauty | Mục 4 hiện không có số liệu | ~3-6h tùy cấu hình, cần theo dõi OOM |
| 5 | Train Qwen3-Embedding-4B (v1 format) | Sports, ML-1M | Hoàn thiện bảng so sánh scale ở mục 4 | ~3-6h/dataset |

Lệnh chính xác cho từng mục nằm trong `MULTI_MACHINE_GUIDE.md` §3 (đã chia theo máy để chạy song song, đã cập
nhật bỏ hạng mục v2).

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
| Sports | standard | 0.0116 | 0.0171 | 0.0173 |
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
