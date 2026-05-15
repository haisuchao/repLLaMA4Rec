# Kết quả thực nghiệm

> Cập nhật lần cuối: 2026-05-13
>
> **Ký hiệu trạng thái:** ✅ Hoàn thành · ⏳ Đang chạy / chưa eval · ❌ Thất bại

---

## Bảng tổng hợp kết quả

> Cập nhật tự động bằng: `python show_results.py --update-experiments`

<!-- RESULTS_START -->
### BEAUTY

| Model | NDCG@5 | HR@5 | NDCG@10 | HR@10 | NDCG@20 | HR@20 | MRR@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-0.6B · zero-shot | 0.0092 | 0.0182 | 0.0135 | 0.0315 | 0.0179 | 0.0488 | 0.0081 |
| Qwen3-0.6B · FT | 0.0251 | 0.0485 | 0.0372 | 0.0861 | 0.0487 | 0.1318 | 0.0224 |
| Qwen3-0.6B · Aug3-8neg | 0.0263 | 0.0498 | 0.0378 | 0.0856 | 0.0498 | 0.1333 | 0.0233 |
| Qwen3-0.6B · Aug5-8neg | 0.0264 | 0.0509 | 0.0390 | 0.0905 | 0.0518 | 0.1409 | 0.0235 |
| Llama-3.2-1B · zero-shot | 0.0012 | 0.0021 | 0.0015 | 0.0030 | 0.0020 | 0.0048 | 0.0011 |
| Llama-3.2-1B · FT-8neg | 0.0209 | 0.0414 | 0.0313 | 0.0736 | 0.0416 | 0.1148 | 0.0185 |

### SPORTS

| Model | NDCG@5 | HR@5 | NDCG@10 | HR@10 | NDCG@20 | HR@20 | MRR@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-0.6B · zero-shot | 0.0045 | 0.0090 | 0.0071 | 0.0170 | 0.0100 | 0.0289 | 0.0041 |
| Llama-3.2-1B · zero-shot | 0.0005 | 0.0008 | 0.0006 | 0.0013 | 0.0009 | 0.0023 | 0.0004 |

### ML-1M

| Model | NDCG@5 | HR@5 | NDCG@10 | HR@10 | NDCG@20 | HR@20 | MRR@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-0.6B · FT | 0.0259 | 0.0472 | 0.0396 | 0.0899 | 0.0543 | 0.1485 | 0.0245 |

<!-- RESULTS_END -->

---

## Mô tả thực nghiệm (tự động)

> Sinh tự động từ `train_config.json` bằng: `python show_results.py --update-experiments`

<!-- AUTO_EXP_START -->
_Chưa có experiment nào có train_config.json. Chạy train.sh để tự sinh._
<!-- AUTO_EXP_END -->

---

## Mô tả chi tiết các thực nghiệm

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
