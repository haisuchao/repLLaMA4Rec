# Kết quả thực nghiệm

> Cập nhật lần cuối: 2026-05-18
>
> **Ký hiệu trạng thái:** ✅ Hoàn thành · ⏳ Đang chạy / chưa eval · ❌ Thất bại

---

## Bảng tổng hợp kết quả

> Cập nhật tự động bằng: `python show_results.py --update-experiments`

<!-- RESULTS_START -->
### BEAUTY

| Model | NDCG@5 | HR@5 | NDCG@10 | HR@10 | NDCG@20 | HR@20 | MRR@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [Qwen3-Embedding-0.6B · bm25](#exp-beauty-qwen3-embedding-0.6b-bm25) | 0.0166 | 0.0255 | 0.0209 | 0.0388 | 0.0269 | 0.0626 | 0.0154 |
| [Qwen3-Embedding-0.6B · cs10-gs32](#exp-beauty-qwen3-embedding-0.6b-cs10-gs32) | 0.0233 | 0.0453 | 0.0356 | 0.0835 | 0.0469 | 0.1287 | 0.0211 |
| [Qwen3-Embedding-0.6B · cs5-gs50-ep5](#exp-beauty-qwen3-embedding-0.6b-cs5-gs50-ep5) | 0.0225 | 0.0439 | 0.0346 | 0.0813 | 0.0463 | 0.1279 | 0.0205 |
| [Qwen3-Embedding-0.6B · ep10-gs50](#exp-beauty-qwen3-embedding-0.6b-ep10-gs50) | 0.0234 | 0.0445 | 0.0350 | 0.0807 | 0.0466 | 0.1268 | 0.0212 |
| [Qwen3-Embedding-0.6B · gs16](#exp-beauty-qwen3-embedding-0.6b-gs16) | 0.0226 | 0.0434 | 0.0344 | 0.0803 | 0.0456 | 0.1244 | 0.0206 |
| [Qwen3-Embedding-0.6B · gs32](#exp-beauty-qwen3-embedding-0.6b-gs32) | 0.0234 | 0.0450 | 0.0360 | 0.0840 | 0.0473 | 0.1288 | 0.0215 |
| [Qwen3-Embedding-0.6B · gs8](#exp-beauty-qwen3-embedding-0.6b-gs8) | 0.0217 | 0.0426 | 0.0329 | 0.0775 | 0.0434 | 0.1189 | 0.0195 |
| [llama-3.2-1b](#exp-beauty-llama-3.2-1b) | 0.0209 | 0.0414 | 0.0313 | 0.0736 | 0.0416 | 0.1148 | 0.0185 |
| [llama-3.2-1b · zero-shot](#exp-beauty-llama-3.2-1b-zeroshot) | 0.0012 | 0.0021 | 0.0015 | 0.0030 | 0.0020 | 0.0048 | 0.0011 |
| [qwen3-1.7b · zero-shot](#exp-beauty-qwen3-1.7b-zeroshot) | 0.0005 | 0.0007 | 0.0005 | 0.0009 | 0.0006 | 0.0013 | 0.0004 |
| [qwen3-embedding-0.6b](#exp-beauty-qwen3-embedding-0.6b) | 0.0251 | 0.0485 | 0.0372 | 0.0861 | 0.0487 | 0.1318 | 0.0224 |
| [qwen3-embedding-0.6b-aug-3](#exp-beauty-qwen3-embedding-0.6b-aug-3) | 0.0263 | 0.0498 | 0.0378 | 0.0856 | 0.0498 | 0.1333 | 0.0233 |
| [qwen3-embedding-0.6b-aug-5](#exp-beauty-qwen3-embedding-0.6b-aug-5) | 0.0264 | 0.0509 | 0.0390 | 0.0905 | 0.0518 | 0.1409 | 0.0235 |
| [qwen3-embedding-0.6b-aug-5-ep-1-lor-32](#exp-beauty-qwen3-embedding-0.6b-aug-5-ep-1-lor-32) | 0.0251 | 0.0475 | 0.0365 | 0.0830 | 0.0478 | 0.1278 | 0.0224 |
| [qwen3-embedding-0.6b · zero-shot](#exp-beauty-qwen3-embedding-0.6b-zeroshot) | 0.0092 | 0.0182 | 0.0135 | 0.0315 | 0.0179 | 0.0488 | 0.0081 |

### SPORTS

| Model | NDCG@5 | HR@5 | NDCG@10 | HR@10 | NDCG@20 | HR@20 | MRR@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [llama-3.2-1b · zero-shot](#exp-sports-llama-3.2-1b-zeroshot) | 0.0005 | 0.0008 | 0.0006 | 0.0013 | 0.0009 | 0.0023 | 0.0004 |
| [qwen3-embedding-0.6b · zero-shot](#exp-sports-qwen3-embedding-0.6b-zeroshot) | 0.0045 | 0.0090 | 0.0071 | 0.0170 | 0.0100 | 0.0289 | 0.0041 |

### ML-1M

| Model | NDCG@5 | HR@5 | NDCG@10 | HR@10 | NDCG@20 | HR@20 | MRR@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [qwen3-embedding-0.6b](#exp-ml-1m-qwen3-embedding-0.6b) | 0.0259 | 0.0472 | 0.0396 | 0.0899 | 0.0543 | 0.1485 | 0.0245 |

<!-- RESULTS_END -->

> **† Kết quả từ paper**: GRU4Rec, BERT4Rec, SASRec lấy từ bảng kết quả S³-Rec (Zhou et al., CIKM 2020) được trích dẫn lại trong TIGER (Rajput et al., NeurIPS 2023). TIGER là kết quả gốc của paper đó. Protocol tương thích: Amazon 2014, 5-core filter, leave-one-out split, full ranking. Metric `HR@K = Recall@K` khi chỉ có 1 item relevant. Không có số liệu @20 và MRR từ paper gốc. Không có kết quả ML-1M trong paper này.

---

## Mô tả thực nghiệm (tự động)

> Sinh tự động từ `train_config.json` bằng: `python show_results.py --update-experiments`

<!-- AUTO_EXP_START -->
### BEAUTY

<a id="exp-beauty-qwen3-embedding-0.6b-bm25"></a>
#### Qwen3-Embedding-0.6B · bm25

| Thuộc tính | Giá trị |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-0.6B` |
| Dataset | beauty |
| Data variant | bm25 |
| train_group_size | 8 (1 positive + 7 negatives) |
| per_device_batch | 4 |
| gradient_accumulation | 8 (effective batch = 32) |
| Learning rate | 1e-4 |
| Epochs | 3 |
| Save steps | 1000 |
| Query max len | 128 |
| Passage max len | 196 |
| Best checkpoint | checkpoint-1000 |
| Selection metric | ndcg_10 (valid) = 0.0244 |
| Trained at | 2026-05-15T23:25:23+07:00 |

```bash
./train.sh beauty --data-variant bm25
./eval.sh beauty --tag bm25
```

---

<a id="exp-beauty-qwen3-embedding-0.6b-cs10-gs32"></a>
#### Qwen3-Embedding-0.6B · cs10-gs32

| Thuộc tính | Giá trị |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-0.6B` |
| Dataset | beauty |
| Data variant | cs10 |
| train_group_size | 32 (1 positive + 31 negatives) |
| per_device_batch | 4 |
| gradient_accumulation | 8 (effective batch = 32) |
| Learning rate | 1e-4 |
| Epochs | 3 |
| Save steps | 1000 |
| Query max len | 256 |
| Passage max len | 196 |
| Best checkpoint | checkpoint-2000 |
| Selection metric | ndcg_10 (valid) = 0.0392 |
| Trained at | 2026-05-21T13:26:15+07:00 |

```bash
./train.sh beauty --data-variant cs10 --tag cs10-gs32 --group-size 32
./eval.sh beauty --tag cs10-gs32
```

---

<a id="exp-beauty-qwen3-embedding-0.6b-cs5-gs32"></a>
#### Qwen3-Embedding-0.6B · cs5-gs32

| Thuộc tính | Giá trị |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-0.6B` |
| Dataset | beauty |
| Data variant | cs5 |
| train_group_size | 32 (1 positive + 31 negatives) |
| per_device_batch | 4 |
| gradient_accumulation | 8 (effective batch = 32) |
| Learning rate | 1e-4 |
| Epochs | 3 |
| Save steps | 1000 |
| Query max len | 128 |
| Passage max len | 196 |
| Trained at | 2026-05-20T16:50:39+07:00 |

```bash
./train.sh beauty --data-variant cs5 --tag cs5-gs32 --group-size 32
./eval.sh beauty --tag cs5-gs32
```

---

<a id="exp-beauty-qwen3-embedding-0.6b-cs5-gs50-ep5"></a>
#### Qwen3-Embedding-0.6B · cs5-gs50-ep5

| Thuộc tính | Giá trị |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-0.6B` |
| Dataset | beauty |
| Data variant | cs5 |
| train_group_size | 32 (1 positive + 31 negatives) |
| per_device_batch | 4 |
| gradient_accumulation | 8 (effective batch = 32) |
| Learning rate | 1e-4 |
| Epochs | 5 |
| Save steps | 1000 |
| Query max len | 128 |
| Passage max len | 196 |
| Best checkpoint | checkpoint-2000 |
| Selection metric | ndcg_10 (valid) = 0.0381 |
| Trained at | 2026-05-21T07:33:16+07:00 |

```bash
./train.sh beauty --data-variant cs5 --tag cs5-gs50-ep5 --group-size 32
./eval.sh beauty --tag cs5-gs50-ep5
```

---

<a id="exp-beauty-qwen3-embedding-0.6b-ep10-gs50"></a>
#### Qwen3-Embedding-0.6B · ep10-gs50

| Thuộc tính | Giá trị |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-0.6B` |
| Dataset | beauty |
| Data variant | — |
| train_group_size | 50 (1 positive + 49 negatives) |
| per_device_batch | 4 |
| gradient_accumulation | 8 (effective batch = 32) |
| Learning rate | 1e-4 |
| Epochs | 10 |
| Save steps | 1000 |
| Query max len | 128 |
| Passage max len | 196 |
| Best checkpoint | checkpoint-2000 |
| Selection metric | ndcg_10 (valid) = 0.0383 |
| Trained at | 2026-05-19T11:06:32+07:00 |

```bash
./train.sh beauty --tag ep10-gs50 --group-size 50
./eval.sh beauty --tag ep10-gs50
```

---

<a id="exp-beauty-qwen3-embedding-0.6b-gs16"></a>
#### Qwen3-Embedding-0.6B · gs16

| Thuộc tính | Giá trị |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-0.6B` |
| Dataset | beauty |
| Data variant | — |
| train_group_size | 8 (1 positive + 7 negatives) |
| per_device_batch | 4 |
| gradient_accumulation | 8 (effective batch = 32) |
| Learning rate | 1e-4 |
| Epochs | 3 |
| Save steps | 1000 |
| Query max len | 128 |
| Passage max len | 196 |
| Best checkpoint | checkpoint-2097 |
| Selection metric | ndcg_10 (valid) = 0.0379 |
| Trained at | 2026-05-19T08:27:26+07:00 |

```bash
./train.sh beauty --tag gs16
./eval.sh beauty --tag gs16
```

---

<a id="exp-beauty-qwen3-embedding-0.6b-gs32"></a>
#### Qwen3-Embedding-0.6B · gs32

| Thuộc tính | Giá trị |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-0.6B` |
| Dataset | beauty |
| Data variant | — |
| train_group_size | 32 (1 positive + 31 negatives) |
| per_device_batch | 4 |
| gradient_accumulation | 8 (effective batch = 32) |
| Learning rate | 1e-4 |
| Epochs | 3 |
| Save steps | 1000 |
| Query max len | 128 |
| Passage max len | 196 |
| Best checkpoint | checkpoint-2097 |
| Selection metric | ndcg_10 (valid) = 0.0396 |
| Trained at | 2026-05-18T15:13:04+00:00 |

```bash
./train.sh beauty --tag gs32 --group-size 32
./eval.sh beauty --tag gs32
```

---

<a id="exp-beauty-qwen3-embedding-0.6b-gs50-ep5"></a>
#### Qwen3-Embedding-0.6B · gs50-ep5

| Thuộc tính | Giá trị |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-0.6B` |
| Dataset | beauty |
| Data variant | — |
| train_group_size | 50 (1 positive + 49 negatives) |
| per_device_batch | 4 |
| gradient_accumulation | 8 (effective batch = 32) |
| Learning rate | 1e-4 |
| Epochs | 5 |
| Save steps | 1000 |
| Query max len | 128 |
| Passage max len | 196 |
| Trained at | 2026-05-19T09:43:19+07:00 |

```bash
./train.sh beauty --tag gs50-ep5 --group-size 50
./eval.sh beauty --tag gs50-ep5
```

---

<a id="exp-beauty-qwen3-embedding-0.6b-gs8"></a>
#### Qwen3-Embedding-0.6B · gs8

| Thuộc tính | Giá trị |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-0.6B` |
| Dataset | beauty |
| Data variant | — |
| train_group_size | 8 (1 positive + 7 negatives) |
| per_device_batch | 4 |
| gradient_accumulation | 8 (effective batch = 32) |
| Learning rate | 1e-4 |
| Epochs | 3 |
| Save steps | 1000 |
| Query max len | 128 |
| Passage max len | 196 |
| Best checkpoint | checkpoint-2097 |
| Selection metric | ndcg_10 (valid) = 0.0354 |
| Trained at | 2026-05-19T01:48:51+00:00 |

```bash
./train.sh beauty --tag gs8
./eval.sh beauty --tag gs8
```

---

<!-- AUTO_EXP_END -->

---

## Mô tả chi tiết các thực nghiệm

---

<a id="exp-paper-baselines"></a>

### EXP-P · Paper Baselines (TIGER, NeurIPS 2023)

| Thuộc tính | Giá trị |
|---|---|
| Nguồn | Rajput et al., "Recommender Systems with Generative Retrieval", NeurIPS 2023 (arXiv:2305.05065) |
| Dataset | Amazon Beauty · Amazon Sports and Outdoors (phiên bản 2014, McAuley) |
| Filter | 5-core (loại user < 5 reviews) |
| Split | Leave-one-out: item cuối = test, áp chót = valid |
| Evaluation | Full ranking trên toàn bộ corpus |
| Metrics | Recall@5, NDCG@5, Recall@10, NDCG@10 (không có @20, MRR) |
| Lưu ý | GRU4Rec / BERT4Rec / SASRec: kết quả lấy từ S³-Rec (Zhou et al., CIKM 2020), không reproduce độc lập. TIGER là mô hình đề xuất của paper. |

**Mô tả các model:**

- **GRU4Rec** (Hidasi et al., 2015): RNN-based, dùng GRU để model sequential interactions theo session.
- **BERT4Rec** (Sun et al., CIKM 2019): Bidirectional Transformer với masked item prediction, dùng toàn bộ interaction history.
- **SASRec** (Kang & McAuley, ICDM 2018): Causal self-attention Transformer, chỉ nhìn các item trước đó.
- **TIGER** (Rajput et al., NeurIPS 2023): Generative retrieval — encode item thành Semantic ID (RQ-VAE trên SentenceT5 embeddings), dùng Transformer seq2seq để predict Semantic ID của item tiếp theo. Giới hạn lịch sử training ở 20 items.

---

<a id="exp-beauty-sasrec"></a>
<a id="exp-sports-sasrec"></a>
<a id="exp-ml-1m-sasrec"></a>

### EXP-B · SASRec (Baseline)

| Thuộc tính | Giá trị |
|---|---|
| Loại model | Sequential Recommendation (ID-based) |
| Framework | RecBole 1.2.1 |
| Dataset | Beauty · Sports · ML-1M |
| Loss | Cross-Entropy (full ranking, softmax trên toàn bộ items) |
| MAX_ITEM_LIST_LENGTH | 200 (tối đa 200 items gần nhất) |
| Cải tiến | — (config mặc định RecBole) |
| Ghi chú | Baseline ID-based để so sánh với repLLaMA (text-based) |

```bash
python run_recbole.py SASRec beauty
python run_recbole.py SASRec sports
python run_recbole.py SASRec ml-1m
```

---

<a id="exp-beauty-qwen3-embedding-0.6b-zeroshot"></a>
<a id="exp-sports-qwen3-embedding-0.6b-zeroshot"></a>

### EXP-Q1 · Qwen3-0.6B zero-shot

| Thuộc tính | Giá trị |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-0.6B` |
| Fine-tuning | Không (zero-shot) |
| Query format | `"Query: <item1>, <item2>, <item3> </s>"` — 3 items gần nhất |
| Pooling | Last token (`<\|im_end\|>`) |
| Dataset | Beauty · Sports |
| Augmentation | — |
| Ghi chú | Baseline để đo gain của fine-tuning |

```bash
./eval.sh beauty base
./eval.sh sports base
```

---

<a id="exp-beauty-qwen3-embedding-0.6b"></a>

### EXP-Q2 · Qwen3-0.6B fine-tuned 50 neg · Beauty

| Thuộc tính | Giá trị |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-0.6B` |
| LoRA rank / alpha | 16 / 64 |
| LoRA target modules | q, k, v, o, gate, up, down |
| Dataset training | Beauty standard (22,363 samples, 1 sample/user) |
| Epochs | 3 |
| train_group_size | 50 (1 positive + 49 negatives) |
| Negative sampling | Random từ pre-mined pool (50 negatives/query) |
| per_device_batch | 1 |
| gradient_accumulation | 32 (effective batch = 32 queries) |
| Learning rate | 1e-4 |
| Context size | 3 items |
| Augmentation | Không |
| Cải tiến | — (config baseline) |

```bash
./train.sh beauty Qwen/Qwen3-Embedding-0.6B 50
./eval.sh beauty
```

---

<a id="exp-beauty-qwen3-embedding-0.6b-aug-3"></a>

### EXP-Q3 · Qwen3-0.6B + Augmented (window=3) 8 neg · Beauty

| Thuộc tính | Giá trị |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-0.6B` |
| LoRA rank / alpha | 16 / 64 |
| LoRA target modules | q, k, v, o, gate, up, down |
| Dataset training | Beauty **augmented** (~154k samples, ~7 samples/user) |
| Augmentation | Sliding window, window\_size=3 |
| Epochs | 3 |
| train_group_size | 8 (1 positive + 7 negatives) |
| Negative sampling | Random từ pre-mined pool |
| per_device_batch | 4 |
| gradient_accumulation | 8 (effective batch = 32 queries) |
| Learning rate | 1e-4 |
| Context size | 3 items (= window\_size) |
| Cải tiến | Data augmentation — nhiều training samples/user hơn |
| Best checkpoint | checkpoint-11000 (valid NDCG@10 = ?) |

```bash
# Bước 1: tạo augmented data
cd dataset && python export_tevatron.py beauty --window_size 3 && cd ..

# Bước 2: train
./train.sh beauty Qwen/Qwen3-Embedding-0.6B 8 aug-3

# Bước 3: eval
./eval.sh beauty best Qwen/Qwen3-Embedding-0.6B test aug-3
```

---

<a id="exp-sports-qwen3-embedding-0.6b"></a>

### EXP-Q4 · Qwen3-0.6B fine-tuned 50 neg · Sports

| Thuộc tính | Giá trị |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-0.6B` |
| Cấu hình | Giống [EXP-Q2](#exp-q2--qwen3-06b-fine-tuned-50-neg--beauty) |
| Dataset training | Sports standard (35,598 samples) |
| Trạng thái | Training xong, eval test chưa hoàn thành |

```bash
./train.sh sports Qwen/Qwen3-Embedding-0.6B 50
./eval.sh sports
```

---

<a id="exp-ml-1m-qwen3-embedding-0.6b"></a>

### EXP-Q5 · Qwen3-0.6B fine-tuned · ML-1M

| Thuộc tính | Giá trị |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-0.6B` |
| LoRA rank / alpha | 16 / 64 |
| LoRA target modules | q, k, v, o, gate, up, down |
| Dataset training | MovieLens 1M (6,040 samples, 1 sample/user) |
| Epochs | 3 |
| train_group_size | 8 (1 positive + 7 negatives) |
| Negative sampling | Random từ pre-mined pool |
| per_device_batch | 4 |
| gradient_accumulation | 8 |
| Learning rate | 1e-4 |
| Context size | 3 items |
| Augmentation | Không |
| Ghi chú | Avg seq length = 165 → context 3/165 items |

```bash
./train.sh ml-1m
./eval.sh ml-1m
```

---

<a id="exp-beauty-llama-3.2-1b-zeroshot"></a>
<a id="exp-sports-llama-3.2-1b-zeroshot"></a>

### EXP-L1 · Llama-3.2-1B zero-shot

| Thuộc tính | Giá trị |
|---|---|
| Base model | `meta-llama/Llama-3.2-1B` |
| Fine-tuning | Không (zero-shot) |
| Query format | `"Query: <item1>, <item2>, <item3> </s>"` |
| Pooling | Last token |
| Dataset | Beauty · Sports |
| Ghi chú | So sánh với Qwen3-0.6B zero-shot — Llama-3.2-1B không phải embedding model nên zero-shot rất kém |

```bash
./eval.sh beauty base meta-llama/Llama-3.2-1B
./eval.sh sports base meta-llama/Llama-3.2-1B
```

---

<a id="exp-beauty-llama-3.2-1b"></a>

### EXP-L2 · Llama-3.2-1B fine-tuned · Beauty

| Thuộc tính | Giá trị |
|---|---|
| Base model | `meta-llama/Llama-3.2-1B` |
| LoRA rank / alpha | 16 / 64 |
| LoRA target modules | q, k, v, o, gate, up, down |
| Dataset training | Beauty standard (22,363 samples) |
| Epochs | 3 |
| train_group_size | 8 (1 positive + 7 negatives) |
| Negative sampling | Random từ pre-mined pool |
| per_device_batch | 1 (Llama branch trong train.sh) |
| gradient_accumulation | 32 |
| Learning rate | 1e-4 |
| Context size | 3 items |
| Augmentation | Không |
| Cải tiến | — |

```bash
./train.sh beauty meta-llama/Llama-3.2-1B
./eval.sh beauty best meta-llama/Llama-3.2-1B
```

---

<a id="exp-sports-llama-3.2-1b"></a>

### EXP-L3 · Llama-3.2-1B fine-tuned · Sports

| Thuộc tính | Giá trị |
|---|---|
| Base model | `meta-llama/Llama-3.2-1B` |
| Cấu hình | Giống [EXP-L2](#exp-l2--llama-321b-fine-tuned--beauty) |
| Dataset training | Sports standard (35,598 samples) |
| Trạng thái | ❌ Thất bại — valid NDCG@10 = 0.0000 (model không học được) |
| Ghi chú | Cần điều tra nguyên nhân: LR quá cao? Dữ liệu Sports khó hơn? |

```bash
./train.sh sports meta-llama/Llama-3.2-1B
```
