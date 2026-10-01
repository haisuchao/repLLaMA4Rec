# Chẩn đoán negative sampling cho reranker

Mục tiêu: hiểu vì sao reranker train với hard negatives = top-100 của retriever không cải thiện được
(rankLLaMA trên `aug-5`: NDCG@10 0.0387 < retriever raw 0.0390 < retriever + history filter 0.0616).

```bash
source tevatron-env/bin/activate
bash diag/mine_ranks.sh beauty aug-5 200          # encode corpus + train/valid/test, FAISS top-200 (~vài phút GPU)
python diag/analyze_negatives.py beauty aug-5     # thống kê → output/<ds>/<model>-<tag>/diag/negative_diagnosis.json
```

Retriever dùng adapter cuối ở thư mục gốc `output/beauty/qwen3-embedding-0.6b-aug-5/`. Kiểm tra chéo:
HR@10 test tính lại = 0.0906 raw / 0.1089 filtered, khớp `paper_report.md` (0.0905 / 0.1088).

## Kết quả — Beauty, retriever `aug-5` (2026-09-25)

### 1. Vị trí positive trong pool top-100

| Split | R@10 | R@100 | R@100 (history-filtered) | Positive nằm ngoài top-100 |
|---|---|---|---|---|
| train (N-2) | 0.128 | 0.417 | 0.423 | **58.3%** |
| valid (N-1) | 0.105 | 0.333 | 0.337 | 66.7% |
| test (N) | 0.091 | 0.292 | 0.297 | **70.9%** |

- `prepare_rerank_data.py --mode train` vẫn giữ các query này (positive lấy từ qrels, negatives từ top-100)
  → **58% nhóm train có positive xa về ngữ nghĩa hơn cả 7 negatives**. Reranker được dạy "chọn item ít giống
  lịch sử nhất", ngược với tình huống lúc test: chỉ 29% query test có positive trong top-100, và ở đó
  positive nằm lẫn giữa các ứng viên giống nhau.
- Lệch train/test có tồn tại (R@100 0.42 vs 0.29, do retriever đã thấy query train) nhưng vừa phải.
- Trần cho reranker trên top-100 đã lọc: HR@10 tối đa 0.297, hiện tại 0.109 (2.7×).

### 2. Thành phần negatives theo dải rank (train split, % negatives thuộc mỗi loại; một item có thể thuộc nhiều loại)

| Dải rank | history | future | dup | co-buy | next-of-last | clean |
|---|---|---|---|---|---|---|
| 1-10 | 23.5% | 1.2% | 1.0% | 15.4% | 13.9% | 55.9% |
| 11-20 | 3.0% | 0.8% | 0.7% | 14.1% | 11.4% | 76.0% |
| 21-50 | 1.2% | 0.5% | 0.4% | 12.3% | 7.8% | 81.4% |
| 51-100 | 0.5% | 0.3% | 0.2% | 9.3% | 4.5% | 86.7% |
| 101-200 | 0.3% | 0.2% | 0.1% | 6.4% | 2.4% | 91.2% |
| random | 0.05% | 0.02% | 0.01% | 0.5% | 0.2% | 99.3% |

Định nghĩa: `history` = mua trước positive; `future` = mua sau positive; `dup` = Jaccard từ của title với
positive ≥ 0.6; `co-buy` = ≥ 2 user khác mua cả positive lẫn item; `next-of-last` = ≥ 1 user khác mua item
ngay sau item cuối của query. Co-occurrence chỉ tính trên phần train của chuỗi, loại chính user đó.

- ~1/4 negatives ở top-10 là item lịch sử — history filter lúc inference đã loại chúng, nên đây là tín hiệu
  train lãng phí (và lệch với input lúc inference nếu inference có filter).
- **Tín hiệu collaborative không phân biệt được negatives top-10 với positive thật**: tỉ lệ next-of-last của
  positive là 12.2% (train) / 8.6% (test), của negatives top-10 là 13.9% / 11.6%. Theo proxy này, negatives
  top-10 "hợp lý" ít nhất ngang positive → bằng chứng (gián tiếp) cho false negatives dày ở dải trên cùng.
- Negatives top-10 phổ biến hơn positive một chút (popularity trung vị 19 vs 12–17).

## ⚠️ Bug rankLLaMA (Tevatron reranker) — mọi kết quả rankLLaMA trước 2026-09-26 không hợp lệ

Qwen3 config có `pad_token_id = None` → `tevatron/reranker/modeling.py` đặt `config.pad_token_id = 0`, trong khi
tokenizer pad bằng `<|endoftext|>` (151643). `AutoModelForSequenceClassification` chọn "token cuối khác pad" theo
`config.pad_token_id` → không nhận ra pad thật → lấy **vị trí cuối của batch**, rơi vào token pad với mọi chuỗi
ngắn hơn chuỗi dài nhất batch. Với flash-attention varlen, hidden state ở vị trí pad là hằng số → score hằng số.

Bằng chứng (reranker cũ `qwen3-embedding-0.6b-aug-5-reranker`, 651,800 cặp test): 617,889 cặp (94.8%) có đúng
score 3.671875; trung vị số giá trị score khác nhau trong 1 query = 1; thứ tự rerank ≈ thứ tự retriever
(Spearman trung vị 1.000) → NDCG@10 0.0395 ≈ retriever 0.0391. Bug có ở **cả train lẫn inference** → loss chỉ
đúng cho chuỗi dài nhất mỗi batch.

Fix: `tevatron/src/tevatron/reranker/driver/{train,rerank}.py` gán `config.pad_token_id = tokenizer.pad_token_id`
sau khi dựng model. Kiểm chứng: score khác nhau giữa ứng viên (trung vị 62.5 giá trị/query) và không phụ thuộc
batch size (|Δ| trung bình 0.004, nhiễu bf16).

Không ảnh hưởng: retriever (left padding + last pooling) và Qwen3-Reranker (`rerank_qwen3.py`,
`train_reranker_qwen3.py`: left padding + `logits[:, -1]`).

## Thí nghiệm kiểm chứng (rankLLaMA đã fix pad bug) — Beauty, 2026-09-26

`bash diag/run_rerank_exp.sh` (dữ liệu: `python diag/build_rerank_data.py beauty aug-5`). Cùng siêu tham số với
`train_reranker.sh` (Qwen3-Embedding-0.6B + LoRA 16/64, group 8, lr 1e-4, 3 epoch, effective batch 32), chỉ
khác dữ liệu train. Inference trên 6,518 query test có positive trong top-100; query còn lại tính là miss cho
mọi hệ thống → metric trên đủ 22,363 query, trùng với chạy toàn bộ. Một seed.

| Model | Dữ liệu train | #query | NDCG@10 raw | NDCG@10 filtered | HR@10 filtered |
|---|---|---|---|---|---|
| retriever `aug-5` | — | — | 0.0391 | **0.0617** | **0.1089** |
| R0 | như cũ: mọi query, neg = top-100 | 22,363 | 0.0336 | 0.0357 | 0.0683 |
| R0s | 9,332 query ngẫu nhiên từ R0 | 9,332 | 0.0252 | 0.0270 | 0.0550 |
| R1 | chỉ query có positive trong top-100 | 9,332 | **0.0561** | 0.0587 | 0.1010 |
| R2 | R1 + bỏ item lịch sử khỏi pool neg | 9,332 | 0.0452 | 0.0584 | 0.1008 |
| OLD | reranker cũ (train khi còn bug) | — | 0.0160 | 0.0180 | 0.0381 |

Bảng đầy đủ 9 metric: `output/beauty/qwen3-embedding-0.6b-aug-5/diag/rerank/results.md`.

- **Phát hiện 1 được xác nhận mạnh**: cùng số query, R1 vs R0s = +123% NDCG@10 raw (0.0561 vs 0.0252);
  R1 dùng ít hơn 2.4× dữ liệu vẫn hơn R0 +67%. Giữ các nhóm có positive ngoài pool làm reranker tệ hơn cả
  retriever (R0 < retriever ở cả raw lẫn filtered).
- Bỏ item lịch sử khỏi pool (R2) không giúp sau filter (0.0584 vs 0.0587, trong nhiễu) và làm raw kém đi
  (0.0452 vs 0.0561) — history negatives là tín hiệu có ích/vô hại, không phải nguyên nhân.
- Chưa reranker nào vượt retriever + history filter (R1 filtered 0.0587 vs 0.0617, −4.9%). Khoảng cách còn lại
  nằm ở việc phân biệt positive với các ứng viên mới, hợp lý trong top-100 — đúng chỗ phát hiện 3
  (false negatives dày ở dải trên) chỉ ra.

## Fusion retriever + reranker (R1) — Beauty, 2026-09-27

`python diag/fuse_scores.py beauty aug-5 R1`. s = α·norm(retriever) + (1−α)·norm(reranker), chuẩn hoá trong
từng query sau history filter. α chọn trên mẫu valid (2,000 query có positive trong top-100,
`valid_pairs_sample.jsonl`), áp dụng một lần lên test. Bảng: `diag/rerank/fusion_results.md`.

| Hệ thống (test, history-filtered) | α | NDCG@10 | HR@10 | MRR@10 | NDCG@20 | HR@20 |
|---|---|---|---|---|---|---|
| retriever | — | 0.0617 | 0.1089 | 0.0474 | 0.0729 | 0.1534 |
| R1 | 0 | 0.0587 | 0.1010 | 0.0458 | 0.0689 | 0.1415 |
| R1 + retriever, min-max | 0.5 | 0.0679 | **0.1169** | 0.0530 | 0.0788 | 0.1601 |
| **R1 + retriever, z-score** | 0.55 | **0.0681** | 0.1168 | **0.0534** | **0.0791** | **0.1604** |
| R1 + retriever, RRF | 0.6 | 0.0666 | 0.1157 | 0.0518 | 0.0778 | 0.1601 |

- Fusion (z-score) vượt retriever + filter: **+10.4% NDCG@10, +7.3% HR@10** — lần đầu reranker đóng góp thật.
- Đường cong valid trơn, đỉnh rộng quanh α = 0.5–0.6 (min-max và z-score gần như trùng) → lựa chọn α ổn định.
- Khớp với phân tích bổ sung: R1 và retriever đúng ở các query khác nhau (hợp HR@10 = 0.1439).

## Khử false negatives (R3) và SimANS (R4) — Beauty, 2026-09-27

`python diag/build_denoised_data.py beauty aug-5`. Cùng 9,332 query và cùng siêu tham số với R1; R1 cũng chỉ
thấy 21 negatives/query qua 3 epoch (7/epoch, lấy đều từ top-100) → chỉ khác **cách chọn** negatives.

- R3: bỏ negatives bị gắn cờ future / dup / co-buy ≥ 2 / next-of-last ≥ 1 (giữ 71.8% pool; co-buy chiếm 22.3%).
- R4: SimANS, p ∝ exp(−a·(s_neg − s_pos)²), σ = 0.01 (a = 5000); rank trung bình negative được chọn = 41.5.

| Test, history-filtered | Riêng lẻ NDCG@10 | + retriever (z-score) NDCG@10 | HR@10 | NDCG@10 valid mẫu (fused) |
|---|---|---|---|---|
| retriever | 0.0617 | — | 0.1089 | 0.1979 |
| **R1** (lấy đều top-100) | **0.0587** | **0.0681** | **0.1168** | 0.2181 |
| R3 (khử nhiễu collaborative) | 0.0553 | 0.0674 | 0.1150 | 0.2221 |
| R4 (SimANS) | 0.0556 | 0.0667 | 0.1135 | 0.2230 |

- Cả hai đều **không tốt hơn** lấy đều: riêng lẻ kém R1 ~5–6%, sau fusion kém 1–2%.
- Mẫu valid (2,000 query) xếp ngược lại (R4 > R3 > R1) trong khi test xếp R1 > R3 > R4 → chênh lệch sau fusion
  nằm trong nhiễu của mẫu valid; không đủ để chọn giữa các biến thể ở mức 1–2%. Kết luận an toàn: làm hẹp pool
  (bỏ item "hợp lý" hoặc dồn về gần positive) không giúp; cần nhiều seed/valid lớn hơn nếu muốn so ở mức này.
- Giả thuyết "false negatives ở dải trên gây hại" (phát hiện 3) **không được ủng hộ** với các proxy này: item
  co-buy/next-of-last gần positive có vẻ là hard negatives hữu ích hơn là nhãn sai.

## Augmented train data (R1aug) — Beauty, 2026-09-27

Như R1 (chỉ query có positive trong top-100, negatives lấy đều) nhưng trên query augmented của
`beauty-aug/train.jsonl` (sliding window, không leak valid/test): 61,547/131,413 query có positive trong pool
(6.6× R1). Cùng siêu tham số, 3 epoch (5,772 bước, ~6.4 giờ trên 3090 Ti).

| Test, history-filtered | α | NDCG@10 | HR@10 | MRR@10 | NDCG@20 | HR@20 |
|---|---|---|---|---|---|---|
| retriever | — | 0.0617 | 0.1089 | 0.0474 | 0.0729 | 0.1534 |
| R1 | 0 | 0.0587 | 0.1010 | 0.0458 | 0.0689 | 0.1415 |
| **R1aug** | 0 | **0.0654** | 0.1065 | **0.0528** | 0.0751 | 0.1452 |
| R1 + retriever (z-score) | 0.55 | 0.0681 | **0.1168** | 0.0534 | 0.0791 | **0.1604** |
| **R1aug + retriever (z-score)** | 0.3 | **0.0708** | 0.1153 | **0.0572** | **0.0810** | 0.1556 |
| R1aug + retriever (min-max) | 0.3 | **0.0709** | 0.1159 | 0.0571 | **0.0812** | 0.1566 |

- R1aug là reranker **đầu tiên vượt retriever + filter khi đứng riêng**: +6.0% NDCG@10, +11.4% MRR@10
  (raw: 0.0624 vs 0.0391, +60%).
- Fusion tốt nhất: **0.0708 NDCG@10 (+14.7% so với retriever + filter)**, MRR@10 +20.7%. α chọn trên valid giảm
  từ 0.55 (R1) xuống 0.3 → reranker mạnh hơn nên được tin nhiều hơn. Đường cong valid vẫn trơn.
- Lợi ích tập trung ở đầu danh sách (NDCG/MRR tăng, HR@10–20 sau fusion ≈ R1 + retriever): reranker đưa positive
  lên cao hơn nhưng không kéo thêm được nhiều positive vào top-10/20.
- So với R3/R4 (chênh lệch trong nhiễu), tăng dữ liệu là đòn bẩy rõ nhất: R1 → R1aug +11.4% NDCG@10 riêng lẻ.

## Group size 32 (R1-g32) — Beauty, 2026-09-28

Dữ liệu R1 (9,332 query), group 32 (1 pos + 31 neg, lấy đều từ top-100 → 93/99 negatives mỗi query qua 3 epoch,
so với 21/99 ở group 8). 1 query × 32 cặp/forward, grad accum 32 → effective batch 32 query/step như R1;
3 epoch, 876 bước, 3h08 (gấp ~3.3× R1). Loss 3.52 → 2.88 (ngẫu nhiên = ln 32 = 3.47).

| Test | NDCG@10 raw | NDCG@10 filtered | HR@10 filtered | + retriever (z-score) NDCG@10 | HR@10 | valid mẫu (fused) |
|---|---|---|---|---|---|---|
| retriever | 0.0391 | 0.0617 | 0.1089 | — | — | 0.1979 |
| R1 (group 8) | **0.0561** | **0.0587** | **0.1010** | **0.0681** (α 0.55) | **0.1168** | 0.2181 |
| R1-g32 | 0.0554 | 0.0579 | 0.0973 | 0.0674 (α 0.5) | 0.1154 | 0.2270 |

- **Không cải thiện**: test kém R1 1–1.4% ở mọi chế độ (trong nhiễu) → không đạt tiêu chí (≥ +3–5%), không chạy
  R1aug-g32 (~25 giờ).
- Lần thứ tư mẫu valid xếp ngược test (R3, R4, R1-g32 đều ≥ R1 trên valid, ≤ R1 trên test) → mẫu valid 2,000 query
  không đủ để phân biệt biến thể ở mức vài %. Cần valid lớn hơn / nhiều seed nếu muốn so tinh hơn.
- Khác retriever (gs8 → gs32: +9% nhưng trong nhiễu run-to-run), group size không phải đòn bẩy cho reranker ở
  quy mô này; lượng query train (R1 → R1aug, +11%) mới là đòn bẩy.

## Group size 32 trên dữ liệu augmented (R1aug-g32) — Beauty, 2026-09-29

Dữ liệu R1aug (61,547 query), group 32, 3 epoch, 1 query × 32 cặp/forward, accum 32; train 20h24 (3.2× R1aug).

| Test, history-filtered | α (valid) | NDCG@10 | HR@10 | MRR@10 | HR@20 |
|---|---|---|---|---|---|
| R1aug riêng | 0 | 0.0654 | 0.1065 | 0.0528 | 0.1452 |
| R1aug-g32 riêng | 0 | 0.0652 | 0.1061 | 0.0527 | 0.1426 |
| R1aug + retriever (z-score) | 0.3 | 0.0708 | 0.1153 | 0.0572 | 0.1556 |
| R1aug-g32 + retriever (z-score) | 0.45 | 0.0723 | 0.1181 | 0.0584 | 0.1607 |

So sánh ở **cùng α** trên test (chỉ để phân tích, không dùng để chọn α) — NDCG@10 / HR@10:

| α | 0 | 0.2 | 0.3 | 0.4 | 0.45 | 0.5 | 0.6 |
|---|---|---|---|---|---|---|---|
| R1aug | 0.0654 / 0.1065 | 0.0691 / 0.1123 | 0.0708 / 0.1153 | 0.0719 / 0.1179 | 0.0721 / 0.1189 | 0.0721 / 0.1198 | 0.0713 / 0.1194 |
| R1aug-g32 | 0.0652 / 0.1061 | 0.0693 / 0.1126 | 0.0710 / 0.1152 | 0.0724 / 0.1179 | 0.0723 / 0.1181 | 0.0723 / 0.1187 | 0.0721 / 0.1195 |

- **Group 32 không cải thiện** (chênh ≤ 0.0008 ở mọi α). Phần "+2.1%" của bảng trên đến hoàn toàn từ α khác
  nhau (0.3 vs 0.45) do mẫu valid chọn, không phải từ mô hình. Khớp với R1-g32 trên dữ liệu nhỏ.
- α tối ưu thật trên test nằm quanh 0.4–0.5 cho cả hai; mẫu valid 2,000 query chọn α dao động 0.3–0.45 →
  kết quả fused nên báo cáo là ~0.071–0.072 NDCG@10 (+15–17% so với retriever + filter).

## Depth train 200 (R1aug-d200) — Beauty, 2026-09-29

Như R1aug nhưng giữ query có positive trong **top-200** (74,145 query, +20.5%, chứa toàn bộ query R1aug); pool
negatives = top-200 trừ positive (199), lấy đều 7/epoch. Cùng embedding query (top-100 trùng 131,412/131,413).
Group 8, 3 epoch, 6,954 bước (~7.7h). Inference vẫn depth 100.

Test, history-filtered, z-score fusion — NDCG@10 / HR@10 ở cùng α (chỉ để phân tích):

| α | 0 (riêng lẻ) | 0.3 | 0.4 | 0.45 | 0.5 | 0.6 |
|---|---|---|---|---|---|---|
| R1aug | 0.0654 / 0.1065 | 0.0708 / 0.1153 | 0.0719 / 0.1179 | 0.0721 / 0.1189 | 0.0721 / 0.1198 | 0.0713 / 0.1194 |
| R1aug-g32 | 0.0652 / 0.1061 | 0.0710 / 0.1152 | 0.0724 / 0.1179 | 0.0723 / 0.1181 | 0.0723 / 0.1187 | 0.0721 / 0.1195 |
| R1aug-d200 | **0.0662 / 0.1100** | 0.0709 / 0.1154 | 0.0714 / 0.1175 | 0.0716 / 0.1179 | 0.0715 / 0.1183 | 0.0706 / 0.1187 |

α chọn trên valid: R1aug-d200 → 0.5, fused test 0.0715 / 0.1183.

- Riêng lẻ nhỉnh hơn R1aug (+1.2% NDCG@10, +3.3% HR@10) nhưng sau fusion bằng hoặc kém nhẹ (≤ 0.0007) → **không
  cải thiện có ý nghĩa**. 20% query thêm (positive ở hạng 101–200) + pool dễ hơn bù trừ nhau.
- Cả ba biến thể hội tụ về cùng mức fused ~0.071–0.072 NDCG@10 → với reranker 0.6B + depth inference 100,
  hiệu năng đã bão hoà theo các trục dữ liệu/negative sampling đã thử (group size, depth, SimANS, khử nhiễu).
  Trần hiện tại do Recall@100 của retriever (29.7% query có positive trong top-100 sau filter).

## Pipeline chuẩn trên Sports và Toys — 2026-09-30

`bash train_reranker.sh <ds> --tag <retriever_tag>` + `bash rerank.sh <ds> --tag <retriever_tag>` (cấu hình R1aug:
query augmented cùng format retriever `cs5-aug`, chỉ query có positive trong top-100, group 8, 3 epoch; fusion
z-score, α chọn trên 2,000 query valid). Retriever = adapter cuối (best checkpoint không còn) — Sports retriever
filtered 0.0369 vs 0.0371 trong paper. Log: `log/queue_<ds>_{train,rerank}.log`.

| Dataset | Query train (in-pool / aug) | Train | α | Retriever filtered NDCG@10 / HR@10 / MRR@10 | Reranker riêng | **Fused** | Δ fused vs retriever |
|---|---|---|---|---|---|---|---|
| Beauty (`aug-5`) | 61,556 / 131,413 | 6.4h | 0.45 | 0.0617 / 0.1092 / 0.0473 | 0.0654 / 0.1066 / 0.0528 | **0.0721 / 0.1189 / 0.0577** | +16.8% / +9.0% / +22.0% |
| Sports (`cs5-aug-gs32`) | 64,360 / 189,543 | 6.1h | 0.55 | 0.0369 / 0.0679 / 0.0275 | 0.0368 / 0.0641 / 0.0285 | **0.0423 / 0.0737 / 0.0328** | +14.9% / +8.5% / +19.5% |
| Toys (`cs5-gs32-aug-0.6b`) | 51,271 / 109,361 | 3.8h | 0.55 | 0.0722 / 0.1272 / 0.0554 | 0.0755 / 0.1265 / 0.0599 | **0.0814 / 0.1382 / 0.0640** | +12.8% / +8.6% / +15.5% |

(Beauty: lần chạy `rerank.sh` kiểm chứng với mô hình R1aug, xem mục pipeline chính.)

- Kết quả nhất quán cả 3 dataset: fusion +13–17% NDCG@10, +8.5–9% HR@10, +15–22% MRR@10 so với retriever +
  history filter. Reranker riêng lẻ ≈ retriever (Sports) hoặc nhỉnh hơn (Beauty, Toys) ở NDCG/MRR nhưng HR@10
  thấp hơn một chút — lợi ích chủ yếu đến từ fusion (hai mô hình sai ở các query khác nhau).
- Đường cong α trên valid trơn, đỉnh quanh 0.45–0.6 ở cả 3 dataset.

## Backbone reranker 4B (Qwen3-Embedding-4B) — Beauty, 2026-10-01

`bash train_reranker.sh beauty --tag aug-5 --reranker-model Qwen/Qwen3-Embedding-4B` + `rerank.sh` tương ứng.
Cùng dữ liệu train với 0.6B (`train_data/` dùng chung, 61,556 query), cùng LoRA 16/64, group 8, lr 1e-4, 3 epoch;
batch 1 × accum 32 + gradient checkpointing (cấu hình nhanh nhất đo được; tắt checkpointing OOM ở 24 GB).
Train 32h10 (5.0× 0.6B), inference test 58 phút cho 623k cặp. Retriever top-100 trùng 22,363/22,363 query với
lần chạy 0.6B.

Test, history-filtered, z-score fusion — NDCG@10 / HR@10 / MRR@10 ở cùng α:

| α | 0 (riêng lẻ) | 0.3 | 0.4 | 0.45 | 0.5 |
|---|---|---|---|---|---|
| 0.6B | 0.0654 / 0.1066 / 0.0528 | 0.0709 / 0.1155 / 0.0572 | 0.0719 / 0.1180 / 0.0578 | 0.0721 / 0.1189 / 0.0577 | 0.0721 / 0.1198 / 0.0576 |
| 4B | 0.0650 / 0.1073 / 0.0521 | 0.0704 / 0.1154 / 0.0566 | 0.0714 / 0.1185 / 0.0570 | 0.0718 / 0.1196 / 0.0572 | 0.0715 / 0.1187 / 0.0570 |

α chọn trên valid: 4B → 0.4, fused test 0.0714 / 0.1185 / 0.0570 (+15.8% NDCG@10 so với retriever + filter).

- **4B không tốt hơn 0.6B** (chênh ≤ 0.0006 ở mọi α, trong nhiễu) dù loss train thấp hơn ~0.03–0.04 suốt quá
  trình train → mô hình lớn hơn khớp dữ liệu train tốt hơn chút nhưng không tổng quát hoá tốt hơn.
- Cùng với group 32, depth 200, SimANS, khử nhiễu: mọi trục đã thử (dữ liệu/negatives/sức chứa mô hình) đều hội
  tụ về fused ~0.071–0.072 NDCG@10 trên Beauty → nút thắt không nằm ở reranker mà ở **thông tin đầu vào** (query chỉ
  3 tên item gần nhất, item chỉ có title) và **Recall@100 của retriever** (29.7%).
