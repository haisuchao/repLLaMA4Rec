# Experiment Guideline — repLLaMA Beauty Ablation Study

> **Dataset chính:** Beauty (chạy hết 6 phase, sau đó replicate trên Sports & ML-1M)  
> **Model mặc định:** `Qwen/Qwen3-Embedding-0.6B`  
> **Activate env trước mỗi phiên:** `source tevatron-env/bin/activate`

---

## 0. Tổng quan

### 0.1 Mục tiêu

| Metric | Hiện tại (baseline) | Mục tiêu |
|---|---|---|
| NDCG@5 | 0.0251 | > **0.0450** |
| NDCG@10 | 0.0372 | > **0.0550** |

Tổng cải thiện cần đạt: **+79%** NDCG@5, **+48%** NDCG@10. Với 6 phase, trung bình mỗi phase đóng góp ~8% relative NDCG@10.

### 0.2 Config Tracker — cập nhật sau mỗi phase

Điền kết quả tốt nhất vào bảng này để dùng làm carry-forward input cho phase tiếp theo.

| Phase hoàn thành | Tham số được chọn | Giá trị | NDCG@10 sau phase |
|---|---|---|---|
| Baseline | group-size | 8 | 0.0372 |
| Phase 1 — Group Size | group-size | **\_\_\_** | **\_\_\_** |
| Phase 2 — Epochs | epochs | **\_\_\_** | **\_\_\_** |
| Phase 3 — History Length | context-size, query-max-len | **\_\_\_** | **\_\_\_** |
| Phase 4 — Model Size | model | **\_\_\_** | **\_\_\_** |
| Phase 5 — Augmentation | augment | on/off | **\_\_\_** |
| Phase 6 — Neg Sampling | neg-strategy | **\_\_\_** | **\_\_\_** |

### 0.3 Decision Framework

**Carry-forward một setting khi:**

| NDCG@10 relative improvement | Time overhead so với config hiện tại | Quyết định |
|---|---|---|
| ≥ 5% | bất kỳ | ✅ Carry forward |
| 2–5% | ≤ 1.5× | ✅ Carry forward |
| 2–5% | > 1.5× | ⚠️ Cân nhắc: ghi note, ưu tiên không chọn nếu cần tiết kiệm thời gian |
| < 2% | bất kỳ | ❌ Không carry forward, giữ config cũ |

**Early stop trong một phase:** Nếu option đầu không cải thiện (< 2%), thử ít nhất **1 option tiếp theo** trước khi bỏ hẳn phase. Nếu cả 2 đều không cải thiện → dừng phase.

### 0.4 Quy trình sau mỗi experiment

```bash
# 1. Đợi eval.sh chạy xong
# 2. Cập nhật bảng kết quả trong guideline.md
# 3. Sync kết quả vào experiments.md
python show_results.py --update-experiments
# 4. Điền vào Config Tracker (mục 0.2) nếu phase hoàn thành
```

---

## Phase 1 — Group Size

### Mục tiêu
Tăng số negatives mỗi query để training signal mạnh hơn. Config này **không cần chuẩn bị lại data** và có **time overhead gần bằng 0** (cùng số gradient updates: ~698/epoch).

> **Lưu ý kỹ thuật:**  
> gs=16 và gs=32 cần chỉnh batch/accum để giữ effective batch = 32. train.sh **không** tự điều chỉnh cho group-size, phải truyền tay qua environment hoặc override trong lệnh deepspeed.  
> gs=50: có 0.6× gradient updates so với baseline — ít updates hơn có thể ảnh hưởng đến convergence.

### Config hiện tại (carry-forward input)
```
group-size : 8   (baseline)
epochs     : 3
data-variant: "" (beauty/)
model      : Qwen/Qwen3-Embedding-0.6B
```

### Các thực nghiệm

```bash
# Thực nghiệm 1.1 — gs=16
./train.sh beauty --group-size 16 --tag gs16
./eval.sh beauty --tag gs16

# Thực nghiệm 1.2 — gs=32 (chạy nếu 1.1 cải thiện HOẶC theo early-stop rule)
./train.sh beauty --group-size 32 --tag gs32
./eval.sh beauty --tag gs32

# Thực nghiệm 1.3 — gs=50  (chạy nếu 1.2 cải thiện)
./train.sh beauty --group-size 50 --tag gs50
./eval.sh beauty --tag gs50
```

### Bảng kết quả

| Config | Tag | NDCG@5 | NDCG@10 | HR@10 | MRR@10 | Time vs baseline | Δ NDCG@10 (rel.) | Quyết định |
|---|---|---|---|---|---|---|---|---|
| gs=8 *(baseline)* | gs8 | 0.0217 | 0.0329 | 0.0775 | 0.0195 | 1.0× | — | ✅ baseline |
| gs=16 | gs16 | 0.0226 | 0.0344 | 0.0803 | 0.0206 | ~2.0× | | |
| gs=32 | gs32 | 0.0234 | 0.0360 | 0.0840 | 0.0215 | ~3.0× | | |
| gs=50 | — | 0.0251 | 0.0372 | 0.0861 | 0.0224 | ~5.0× | — | ✅ baseline |

### Quy tắc quyết định

1. Chọn gs có NDCG@10 cao nhất với cải thiện ≥ 2% relative.
2. Nếu gs=50 tốt nhất nhưng cải thiện < 2% → không chọn (ít gradient updates hơn làm training kém stable).
3. Nếu không có gs nào cải thiện ≥ 2% → giữ gs=8, ghi nhận và chuyển Phase 2.

### Carry-forward → Phase 2
```
BEST_GS = 50   # điền giá trị được chọn
BEST_TAG = gs{BEST_GS}   # hoặc "" nếu giữ baseline
```

---

## Phase 2 — Epochs

### Mục tiêu
Xem liệu model có cần train lâu hơn để hội tụ tốt hơn không. Lưu ý risk overfitting khi train quá nhiều epoch.

> **Lưu ý:** Số epoch ảnh hưởng linear đến thời gian training.  
> ep=5 → 1.67× baseline time · ep=10 → 3.33× baseline time

### Config hiện tại (carry-forward input)
```
group-size : <BEST_GS từ Phase 1>
epochs     : 3  (baseline)
data-variant: ""
model      : Qwen/Qwen3-Embedding-0.6B
```

### Các thực nghiệm

```bash
# Thực nghiệm 2.1 — ep=5
./train.sh beauty --epochs 5 --group-size <BEST_GS> --tag <BEST_TAG>-ep5
./eval.sh beauty --tag <BEST_TAG>-ep5

# Thực nghiệm 2.2 — ep=10 (chạy nếu 2.1 cải thiện)
./train.sh beauty --epochs 10 --group-size <BEST_GS> --tag <BEST_TAG>-ep10
./eval.sh beauty --tag <BEST_TAG>-ep10
```

### Bảng kết quả

| Config | Tag | NDCG@5 | NDCG@10 | HR@10 | MRR@10 | Time vs baseline | Δ NDCG@10 (rel.) | Quyết định |
|---|---|---|---|---|---|---|---|---|
| ep=3 *(best từ P1)* | — | | | | | 1.0× | — | ✅ input |
| ep=5 | …-ep5 | | | | | 1.67× | | |
| ep=10 | …-ep10 | | | | | 3.33× | | |

### Quy tắc quyết định

- ep=5 cải thiện ≥ 2% → carry forward.
- ep=10: nếu cải thiện **không đáng kể so với ep=5** (< 2% trên ep=5) → không chọn ep=10 (time cost 3.33× không xứng).
- Cảnh báo overfitting: nếu valid NDCG@10 tiếp tục tăng nhưng tỷ lệ valid/test gap tăng lớn → dừng ở ep=5.

### Carry-forward → Phase 3
```
BEST_GS = ___
BEST_EP = ___
BEST_TAG = ___
```

---

## Phase 3 — History Length (Context Size)

### Mục tiêu
Tăng số item lịch sử trong query từ 3 lên 5 hoặc 10, giúp model nắm bắt pattern dài hơn. **Cần chuẩn bị lại data.**

> **Lưu ý token length (Beauty):**  
> cs=5 → P95 ≈ 94 tokens → `--query-max-len 128` vẫn đủ  
> cs=10 → P95 ≈ 146 tokens → **cần** `--query-max-len 160`

### Config hiện tại (carry-forward input)
```
group-size  : <BEST_GS>
epochs      : <BEST_EP>
context-size: 3  (baseline)
```

### Chuẩn bị dữ liệu

```bash
cd dataset

# cs=5
python export_tevatron.py beauty --context_size 5      # → beauty-cs5/

# cs=10
python export_tevatron.py beauty --context_size 10     # → beauty-cs10/

cd ..
```

### Các thực nghiệm

```bash
# Thực nghiệm 3.1 — cs=5
./train.sh beauty --data-variant cs5 \
    --group-size <BEST_GS> --epochs <BEST_EP> \
    --tag cs5-<BEST_TAG>
./eval.sh beauty --tag cs5-<BEST_TAG>

# Thực nghiệm 3.2 — cs=10 (chạy nếu 3.1 cải thiện)
./train.sh beauty --data-variant cs10 \
    --group-size <BEST_GS> --epochs <BEST_EP> \
    --query-max-len 160 \
    --tag cs10-<BEST_TAG>
./eval.sh beauty --tag cs10-<BEST_TAG>
```

### Bảng kết quả

| Config | Tag | NDCG@5 | NDCG@10 | HR@10 | MRR@10 | Time vs baseline | Δ NDCG@10 (rel.) | Quyết định |
|---|---|---|---|---|---|---|---|---|
| cs=3 *(best từ P2)* | — | | | | | 1.0× | — | ✅ input |
| cs=5 | cs5-… | | | | | ~1.0× | | |
| cs=10 | cs10-… | | | | | ~1.05× | | |

### Quy tắc quyết định

- Time overhead của context_size gần như bằng 0 (chỉ query encoding dài hơn chút) → **quyết định thuần túy theo metric**.
- Chọn cs tốt nhất với cải thiện ≥ 2%.
- Nếu cs=10 không tốt hơn cs=5 → chọn cs=5.

### Carry-forward → Phase 4
```
BEST_GS  = ___
BEST_EP  = ___
BEST_CS  = ___
BEST_QML = ___   # query-max-len tương ứng (128 hoặc 160)
BEST_TAG = ___
```

---

## Phase 4 — Model Size

### Mục tiêu
Đánh giá xem model lớn hơn (Qwen3-4B, 4B params) có tạo ra improvement đáng kể không. Model 4B đã cache sẵn.

> **Lưu ý thời gian:**  
> Qwen3-4B với batch=1/accum=32 tốn ước tính **4–6× thời gian** so với 0.6B. Cân nhắc kỹ cost/benefit.  
> Ngưỡng carry-forward cho phase này: cải thiện **≥ 5% relative** NDCG@10 để justify time cost.

### Config hiện tại (carry-forward input)
```
group-size   : <BEST_GS>
epochs       : <BEST_EP>
data-variant : <BEST_CS_TAG>  (hoặc "" nếu cs=3)
query-max-len: <BEST_QML>
model        : Qwen/Qwen3-Embedding-0.6B  (baseline phase này)
```

### Các thực nghiệm

```bash
# Thực nghiệm 4.1 — Qwen3-4B
./train.sh beauty \
    --model Qwen/Qwen3-Embedding-4B \
    --data-variant <BEST_CS_TAG> \
    --epochs <BEST_EP> \
    --group-size <BEST_GS> \
    --query-max-len <BEST_QML> \
    --tag 4b-<BEST_TAG>
./eval.sh beauty --model Qwen/Qwen3-Embedding-4B --tag 4b-<BEST_TAG>
```

> **Nếu OOM:** Giảm `--group-size` (thử 8 hoặc 4) và ghi chú. So sánh với 0.6B cùng group-size để so sánh công bằng.

### Bảng kết quả

| Config | Tag | NDCG@5 | NDCG@10 | HR@10 | MRR@10 | Time vs baseline | Δ NDCG@10 (rel.) | Quyết định |
|---|---|---|---|---|---|---|---|---|
| 0.6B *(best từ P3)* | — | | | | | 1.0× | — | ✅ input |
| Qwen3-4B | 4b-… | | | | | ~4–6× | | |

### Quy tắc quyết định

- 4B cải thiện **≥ 5%** relative NDCG@10 → carry forward (time justified).
- 4B cải thiện **2–5%** relative → không carry forward (4–6× time không xứng với gain nhỏ).
- 4B cải thiện < 2% → không carry forward, tiếp tục với 0.6B.

### Carry-forward → Phase 5
```
BEST_MODEL = ___   # Qwen3-Embedding-0.6B hoặc Qwen3-Embedding-4B
BEST_GS    = ___
BEST_EP    = ___
BEST_CS    = ___
BEST_QML   = ___
BEST_TAG   = ___
```

---

## Phase 5 — Query Augmentation

### Mục tiêu
Sinh thêm training samples bằng sliding window (~5.9× data trên Beauty → **~5.9× thời gian training**). Mỗi user đóng góp N-3 samples thay vì 1. Cần chuẩn bị lại data.

> **Lưu ý thời gian:**  
> Beauty: 22,363 users → 131,413 aug samples (5.9×) → ~5.9× training time.  
> Threshold carry-forward phase này: **≥ 5% relative** NDCG@10 (vì time cost rất cao).  
> Nếu improvement dưới ngưỡng, ghi nhận và bỏ qua augmentation.

### Config hiện tại (carry-forward input)
```
model        : <BEST_MODEL>
group-size   : <BEST_GS>
epochs       : <BEST_EP>
context-size : <BEST_CS>
query-max-len: <BEST_QML>
augment      : off  (baseline phase này)
```

### Chuẩn bị dữ liệu

```bash
cd dataset

# Augmented, context=<BEST_CS>
python export_tevatron.py beauty \
    --augment \
    --context_size <BEST_CS>
# → beauty-aug/       nếu cs=3
# → beauty-cs5-aug/   nếu cs=5

cd ..
```

### Các thực nghiệm

```bash
# Thực nghiệm 5.1 — augmentation với best context_size
./train.sh beauty \
    --model <BEST_MODEL> \
    --data-variant <AUG_TAG> \
    --group-size <BEST_GS> \
    --epochs <BEST_EP> \
    --query-max-len <BEST_QML> \
    --tag aug-<BEST_TAG>
./eval.sh beauty --model <BEST_MODEL> --tag aug-<BEST_TAG>
```

### Bảng kết quả

| Config | Tag | NDCG@5 | NDCG@10 | HR@10 | MRR@10 | Time vs baseline | Δ NDCG@10 (rel.) | Quyết định |
|---|---|---|---|---|---|---|---|---|
| no-aug *(best từ P4)* | — | | | | | 1.0× | — | ✅ input |
| augment | aug-… | | | | | ~5.9× | | |

### Quy tắc quyết định

- Cải thiện **≥ 5%** relative → carry forward (time overhead 5.9× cần improvement đủ lớn).
- Cải thiện **< 5%** → không carry forward (thời gian train quá cao so với gain).

### Carry-forward → Phase 6
```
BEST_MODEL   = ___
BEST_GS      = ___
BEST_EP      = ___
BEST_CS      = ___
BEST_QML     = ___
BEST_AUGMENT = on/off
BEST_TAG     = ___
```

---

## Phase 6 — Negative Sampling

### Mục tiêu
Thay random negatives bằng BM25 hard negatives (items giống query về mặt lexical nhưng không phải target). Data generation nhanh (~vài phút), training time **không đổi**.

> **Chiến lược thử:**  
> Bắt đầu với `mixed` (10 BM25 hard + 40 random) — ổn định hơn `bm25` all-hard.  
> Nếu `mixed` không cải thiện → thử `bm25` all-hard.

### Config hiện tại (carry-forward input)
```
model        : <BEST_MODEL>
group-size   : <BEST_GS>
epochs       : <BEST_EP>
context-size : <BEST_CS>
query-max-len: <BEST_QML>
augment      : <BEST_AUGMENT>
neg-strategy : random  (baseline phase này)
```

### Chuẩn bị dữ liệu

```bash
cd dataset

# mixed — 10 BM25 hard + 40 random
python export_tevatron.py beauty \
    --context_size <BEST_CS> \
    [--augment]  \           # thêm nếu BEST_AUGMENT=on
    --neg_strategy mixed
# → beauty-mixed/       hoặc beauty-cs5-mixed/   hoặc beauty-aug-mixed/ ...

# bm25 — 50 BM25 hard (chuẩn bị sẵn để dùng nếu mixed không đủ tốt)
python export_tevatron.py beauty \
    --context_size <BEST_CS> \
    [--augment]  \
    --neg_strategy bm25

cd ..
```

### Các thực nghiệm

```bash
# Thực nghiệm 6.1 — mixed negatives
./train.sh beauty \
    --model <BEST_MODEL> \
    --data-variant <MIXED_TAG> \
    --group-size <BEST_GS> \
    --epochs <BEST_EP> \
    --query-max-len <BEST_QML> \
    --tag mixed-<BEST_TAG>
./eval.sh beauty --model <BEST_MODEL> --tag mixed-<BEST_TAG>

# Thực nghiệm 6.2 — bm25 all-hard (chạy nếu 6.1 không cải thiện hoặc theo early-stop rule)
./train.sh beauty \
    --model <BEST_MODEL> \
    --data-variant <BM25_TAG> \
    --group-size <BEST_GS> \
    --epochs <BEST_EP> \
    --query-max-len <BEST_QML> \
    --tag bm25-<BEST_TAG>
./eval.sh beauty --model <BEST_MODEL> --tag bm25-<BEST_TAG>
```

### Bảng kết quả

| Config | Tag | NDCG@5 | NDCG@10 | HR@10 | MRR@10 | Time vs baseline | Δ NDCG@10 (rel.) | Quyết định |
|---|---|---|---|---|---|---|---|---|
| random *(best từ P5)* | — | | | | | 1.0× | — | ✅ input |
| mixed (10h+40r) | mixed-… | | | | | ~1.0× | | |
| bm25 (50h) | bm25-… | | | | | ~1.0× | | |

### Quy tắc quyết định

- `mixed` cải thiện ≥ 2% → carry forward `mixed` (time overhead = 0, không có lý do không chọn).
- `bm25` tốt hơn `mixed` → chọn `bm25`.
- Không có cái nào cải thiện → giữ `random`.

### Kết quả cuối cùng (Beauty)
```
FINAL_MODEL      = ___
FINAL_GS         = ___
FINAL_EP         = ___
FINAL_CS         = ___
FINAL_QML        = ___
FINAL_AUGMENT    = ___
FINAL_NEG        = ___
FINAL_TAG        = ___
FINAL_NDCG5      = ___
FINAL_NDCG10     = ___
Mục tiêu đạt được: NDCG@5 > 0.0450? ___   NDCG@10 > 0.0550? ___
```

---

## Phase 7 — Replicate trên Sports & ML-1M

Sau khi xác định `FINAL_CONFIG` từ Beauty, replicate trên 2 dataset còn lại.

### 7.1 Chuẩn bị dữ liệu

```bash
cd dataset

# Sports
python export_tevatron.py sports \
    --context_size <FINAL_CS> \
    [--augment --max_aug_per_user 100]  \   # ML-1M: --max_aug_per_user 20
    [--neg_strategy <FINAL_NEG>]
# ML-1M
python export_tevatron.py ml-1m \
    --context_size <FINAL_CS> \
    [--augment --max_aug_per_user 20] \
    [--neg_strategy <FINAL_NEG>]

cd ..
```

> **Lưu ý Sports vs ML-1M:**  
> Sports: avg seq len = 8.3 — tương tự Beauty, không cần `--max_aug_per_user`  
> ML-1M: avg seq len = 165.5 — **bắt buộc** `--max_aug_per_user 20` nếu augment

### 7.2 Training

```bash
# Sports
./train.sh sports \
    --model <FINAL_MODEL> \
    --data-variant <FINAL_TAG_DATA> \
    --group-size <FINAL_GS> \
    --epochs <FINAL_EP> \
    --query-max-len <FINAL_QML> \
    --tag <FINAL_TAG>
./eval.sh sports --model <FINAL_MODEL> --tag <FINAL_TAG>

# ML-1M
./train.sh ml-1m \
    --model <FINAL_MODEL> \
    --data-variant <FINAL_TAG_DATA> \
    --group-size <FINAL_GS> \
    --epochs <FINAL_EP> \
    --query-max-len 128 \             # ML-1M không cần tăng (max P95=68 tokens)
    --tag <FINAL_TAG>
./eval.sh ml-1m --model <FINAL_MODEL> --tag <FINAL_TAG>
```

### 7.3 Bảng kết quả đa dataset

| Dataset | NDCG@5 | NDCG@10 | HR@10 | MRR@10 | SASRec NDCG@10 | Δ vs SASRec |
|---|---|---|---|---|---|---|
| Beauty | | | | | | |
| Sports | | | | | | |
| ML-1M | | | | | | |

---

## Tổng kết

### Tóm tắt từng phase

| Phase | Setting thử | Được chọn | NDCG@10 | Δ (abs.) | Δ (rel.) | Time overhead |
|---|---|---|---|---|---|---|
| Baseline | gs=8, ep=3, cs=3 | — | 0.0372 | — | — | 1.0× |
| P1 — Group Size | gs=16,32,50 | | | | | |
| P2 — Epochs | ep=5,10 | | | | | |
| P3 — History Len | cs=5,10 | | | | | |
| P4 — Model Size | Qwen3-4B | | | | | |
| P5 — Augmentation | --augment | | | | | |
| P6 — Neg Sampling | mixed, bm25 | | | | | |

### Cập nhật experiments.md

```bash
python show_results.py --update-experiments
```
