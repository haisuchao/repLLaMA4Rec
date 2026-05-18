# Data Preprocessing Pipeline

Pipeline tiền xử lý dữ liệu cho bài toán **Sequential Recommendation**, hỗ trợ xuất song song hai định dạng để huấn luyện và so sánh hai mô hình:

- **repLLaMA** (thư viện Tevatron v2) — mô hình retrieval dựa trên LLM
- **SASRec** (thư viện RecBole) — mô hình sequential recommendation dựa trên Self-Attention

---

## Mục lục

1. [Cấu trúc thư mục](#1-cấu-trúc-thư-mục)
2. [Cài đặt](#2-cài-đặt)
3. [Chuẩn bị dữ liệu thô](#3-chuẩn-bị-dữ-liệu-thô)
4. [Quy trình tiền xử lý](#4-quy-trình-tiền-xử-lý)
5. [Định dạng output](#5-định-dạng-output)
6. [Cách chạy](#6-cách-chạy)
7. [Data Augmentation — Sliding Window](#7-data-augmentation--sliding-window)
8. [History Length — Context Size](#8-history-length--context-size)
9. [BM25 Hard Negative Mining](#10-bm25-hard-negative-mining)
10. [So sánh hai định dạng](#11-so-sánh-hai-định-dạng)
11. [Cấu hình nâng cao](#12-cấu-hình-nâng-cao)

---

## 1. Cấu trúc thư mục

```
dataset/
│
├── preprocess.py            # Module tiền xử lý chung
│                            #   - Load raw data (Amazon, MovieLens, Steam)
│                            #   - 5-core filter
│                            #   - Build user sequences
│                            #   - Leave-one-out split
│
├── export_tevatron.py       # Xuất định dạng Tevatron v2 (unified)
│                            #   - Context size: --context_size (cả 2 mode)
│                            #   - Query augmentation: --augment
│                            #   - Negative sampling: --neg_strategy (random/bm25/mixed)
├── export_recbole.py        # Xuất định dạng RecBole (SASRec)
├── run_all.py               # Entry point — chạy pipeline standard (export_tevatron + export_recbole)
│
├── raw/                     # Thư mục chứa dữ liệu thô (tự tạo)
│   ├── reviews_Beauty_5.json.gz
│   ├── meta_Beauty.json.gz
│   └── ...
│
└── dataset/                 # Thư mục output (tự sinh)
    ├── tevatron/
    │   ├── <dataset>/           # standard — 1 sample/user, random negatives
    │   ├── <dataset>-w3/        # aug window=3, random negatives
    │   ├── <dataset>-mixed/     # 1 sample/user, BM25 hard + random negatives
    │   ├── <dataset>-w3-mixed/  # aug window=3, BM25 hard + random negatives
    │   └── ...                  # tag tự sinh từ tham số CLI
    └── recbole/
        └── <dataset>/
            ├── <dataset>.inter
            ├── <dataset>.item
            └── sasrec_<dataset>.yaml
```

---

## 2. Cài đặt

```bash
pip install pandas tqdm
```

---

## 3. Chuẩn bị dữ liệu thô

Pipeline hỗ trợ 4 dataset. Tải file về và đặt vào thư mục `raw/` theo cấu trúc dưới đây.

### Amazon Beauty / Sports

Tải tại: http://jmcauley.ucsd.edu/data/amazon/

```
raw/
├── reviews_Beauty_5.json.gz
├── meta_Beauty.json.gz
├── reviews_Sports_and_Outdoors_5.json.gz
└── meta_Sports_and_Outdoors.json.gz
```

Mỗi dòng trong file review là một JSON object:
```json
{
  "reviewerID": "A1RSDE90N6RSZF",
  "asin": "B00006L9LC",
  "unixReviewTime": 1393632000
}
```

### MovieLens 1M

Tải tại: https://grouplens.org/datasets/movielens/1m/

```
raw/ml-1m/
├── ratings.dat      # UserID::MovieID::Rating::Timestamp
└── movies.dat       # MovieID::Title::Genres
```

### Steam

Tải tại: https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data

```
raw/
├── steam_reviews.json.gz
└── steam_games.json.gz
```

---

## 4. Quy trình tiền xử lý

Toàn bộ pipeline được thực hiện theo 5 bước, tuân theo chuẩn của **SASRec paper** (Kang & McAuley, 2018).

### Bước 1 — Load dữ liệu thô

Đọc file raw và chuẩn hoá về cấu trúc thống nhất gồm 3 trường:

| Trường | Kiểu | Mô tả |
|---|---|---|
| `user_id` | string | Định danh người dùng |
| `item_id` | string | Định danh sản phẩm / phim / game |
| `timestamp` | int | Thời điểm tương tác (Unix timestamp) |

Ngoài ra, metadata của item (title) cũng được load để tạo text biểu diễn cho Tevatron.

### Bước 2 — 5-core filter

Lọc lặp đến khi tất cả **user** và **item** đều có tối thiểu 5 tương tác. Quá trình lặp vì sau khi loại item hiếm, một số user có thể mất đi tương tác và ngược lại.

```
Vòng lặp:
  1. Loại item có < 5 tương tác
  2. Loại user có < 5 tương tác
  3. Lặp lại đến khi không còn thay đổi
```

> **Lưu ý:** Sau khi filter, một số user vẫn có thể có ít item hơn `CONTEXT_SIZE` do bỏ duplicate liên tiếp ở bước 3. Đây là các **cold-start users** và vẫn được giữ lại nếu sequence còn đủ `MIN_SEQ_LEN = 3` item.

### Bước 3 — Xây dựng chuỗi tương tác

Với mỗi user, sắp xếp các item đã tương tác theo thứ tự thời gian và loại bỏ các item trùng liên tiếp:

```
Raw interactions (chưa sort):
  user_A: [(item3, t=5), (item1, t=2), (item1, t=3), (item2, t=8)]

Sau sort theo timestamp:
  user_A: [(item1, t=2), (item1, t=3), (item3, t=5), (item2, t=8)]

Sau bỏ duplicate liên tiếp:
  user_A: [item1, item3, item2]
```

Chỉ giữ lại user có sequence dài tối thiểu `MIN_SEQ_LEN = 3` item.

### Bước 4 — Leave-one-out split

Chiến lược chia dữ liệu theo **Leave-one-out** chuẩn của SASRec paper:

```
Sequence đầy đủ: [item_1, item_2, ..., item_{N-2}, item_{N-1}, item_N]
                                             ↑              ↑         ↑
                                          train           valid     test
                                         positive        positive positive
```

| Split | Positive item | Context Tevatron | Full history RecBole |
|---|---|---|---|
| `train` | `item_{N-2}` | tối đa `CONTEXT_SIZE` item trước `item_{N-2}` | `item_1` → `item_{N-3}` |
| `valid` | `item_{N-1}` | tối đa `CONTEXT_SIZE` item trước `item_{N-1}` | `item_1` → `item_{N-2}` |
| `test`  | `item_N`     | tối đa `CONTEXT_SIZE` item trước `item_N`     | `item_1` → `item_{N-1}` |

**Xử lý cold-start** (user có ít hơn `CONTEXT_SIZE + 3` item):

```
Sequence: [i1, i2, i3]  (N=3)
  train → context = []            positive = i1  ← không có history
  valid → context = [i1]          positive = i2
  test  → context = [i1, i2]      positive = i3

Sequence: [i1, i2, i3, i4]  (N=4)
  train → context = [i1]          positive = i2
  valid → context = [i1, i2]      positive = i3
  test  → context = [i1, i2, i3]  positive = i4

Sequence: [i1..i7]  (N=7, đủ history với CONTEXT_SIZE=3)
  train → context = [i3, i4]      positive = i5
  valid → context = [i4, i5]      positive = i6
  test  → context = [i5, i6]      positive = i7
```

### Bước 5 — Xuất dữ liệu

Từ cùng một kết quả tiền xử lý, xuất song song sang 2 định dạng:

```
Preprocessing result
      ├──→ export_tevatron.py  →  dataset/tevatron/<dataset>[-<tag>]/
      │         tag tự sinh theo tham số CLI:
      │           (mặc định)                      → beauty/            1 sample/user, cs=3, random
      │           --context_size 5                → beauty-cs5/        1 sample/user, cs=5, random
      │           --augment                       → beauty-aug/        sliding window, cs=3, random
      │           --context_size 5 --augment      → beauty-cs5-aug/    sliding window, cs=5, random
      │           --neg_strategy mixed             → beauty-mixed/      1 sample/user, cs=3, hard+random
      │           --augment --neg_strategy mixed   → beauty-aug-mixed/  sliding window, cs=3, hard+random
      └──→ export_recbole.py   →  dataset/recbole/<dataset>/
```

---

## 5. Định dạng output

### 5.1 Tevatron v2 — repLLaMA

**`corpus.jsonl`** — mỗi dòng là một item trong toàn bộ dataset:

```json
{"docid": "B00006L9LC", "text": "Passage: Citre Shine Moisture Burst Shampoo"}
```

**`train.jsonl` / `valid.jsonl` / `test.jsonl`** — mỗi dòng là một query tương ứng với một user:

```json
{
  "query_id": "A1RSDE90N6RSZF_train",
  "query": "Query: Citre Shine Shampoo, Dove Body Wash, Bonne Bell Lotion </s>",
  "positive_passages": [
    {"docid": "B0012Y0ZG2", "title": "", "text": "Passage: Neutrogena T/Gel Shampoo"}
  ],
  "negative_passages": [
    {"docid": "B00021DJ32", "title": "", "text": "Passage: Pantene Pro-V Shampoo"},
    {"docid": "B0009RF9DW", "title": "", "text": "Passage: Head & Shoulders Shampoo"}
  ]
}
```

Quy tắc tạo query:
- Prefix `"Query: "` và suffix `" </s>"` theo chuẩn repLLaMA
- Các item trong history được nối bằng dấu `", "`
- Context chỉ dùng tối đa `CONTEXT_SIZE = 3` item gần nhất
- Cold-start không có history: `"Query: </s>"`

Quy tắc negative sampling (chỉ áp dụng cho `train.jsonl`, valid/test không có negatives):

| `--neg_strategy` | Pool negatives | Số lượng mặc định |
|---|---|---|
| `random` (mặc định) | Random từ toàn bộ corpus | 50/query (`--num_hard_neg 10` + `--num_random_neg 40`) |
| `bm25` | Top BM25 scored items | 50/query (toàn bộ là hard) |
| `mixed` | BM25 hard + random | 10 hard + 40 random = 50/query |

Tất cả strategies đều loại trừ **toàn bộ sequence** của user (gồm cả valid/test items) khỏi negative pool. Cold-start (context rỗng) tự động fallback về random dù chọn strategy nào.

### 5.2 RecBole — SASRec

**`<dataset>.inter`** — toàn bộ lịch sử tương tác theo thứ tự thời gian (tab-separated):

```
user_id:token   item_id:token   timestamp:float
A1RSDE90N6RSZF  B00006L9LC      0.0
A1RSDE90N6RSZF  B0012Y0ZG2      1.0
A1RSDE90N6RSZF  B00021DJ32      2.0
```

**`<dataset>.item`** — metadata của item (tab-separated):

```
item_id:token   item_title:token_seq        item_brand:token  item_category:token_seq
B00006L9LC      Citre Shine Shampoo         Unilever          Beauty Hair Care
B0012Y0ZG2      Neutrogena T/Gel Shampoo    Neutrogena        Beauty Hair Care
```

**`sasrec_<dataset>.yaml`** — config SASRec tự sinh với `MAX_ITEM_LIST_LENGTH` bằng độ dài sequence lớn nhất, cho phép SASRec dùng **toàn bộ history**:

```yaml
MAX_ITEM_LIST_LENGTH: 52   # = max sequence length trong dataset

eval_args:
  split: {LS: valid_and_test}  # item cuối = test, áp chót = valid
  order: TO                    # Time Order — giữ thứ tự thời gian
  mode: full                   # Rank trên toàn bộ items
```

---

## 6. Cách chạy

```bash
# Chạy 1 dataset
python run_all.py beauty

# Chạy nhiều dataset cùng lúc
python run_all.py beauty sports ml-1m

# Chạy tất cả dataset được hỗ trợ
python run_all.py --all
```

Output mẫu:

```
############################################################
#  DATASET: BEAUTY
############################################################

── Tevatron export ──────────────────────────────────────
  corpus:  100%|████████| 12101/12101 [00:02<00:00, 5234item/s]
  train :  100%|████████|  2819/2819  [00:05<00:00,  521query/s]
  valid :  100%|████████|  2819/2819  [00:04<00:00,  618query/s]
  test  :  100%|████████|  2819/2819  [00:04<00:00,  601query/s]

  Context window : up to 3 items (shorter for cold-start)
  Cold-start     : 142 train queries with empty context
  Negatives/query: 50 (sampled on-the-fly, no RAM overhead)
  Output dir     : dataset/tevatron/beauty/

── RecBole export ───────────────────────────────────────
  .inter:  100%|████████|  2819/2819  [00:01<00:00, 1823user/s]
  .item :  100%|████████| 12101/12101 [00:00<00:00, 9841item/s]

  Full history   : YES (avg 8.3 items/user)
  Output dir     : dataset/recbole/beauty/

✓ beauty hoàn tất (45.2s)
```

---

## 7. Data Augmentation — Sliding Window

Script tạo thêm training samples bằng cách trượt cửa sổ context qua toàn bộ history, giúp so sánh công bằng hơn với SASRec (vốn train trên tất cả positions).

### Nguyên lý

Với sequence `[i1, i2, i3, i4, i5, i6, i7, i8, i9]` (valid=i8, test=i9), `window_size=3`:

```
Train samples (augmented):
  ({i1},           i2)   ← cold-start
  ({i1, i2},       i3)
  ({i1, i2, i3},   i4)
  ({i2, i3, i4},   i5)   ← cửa sổ bắt đầu trượt
  ({i3, i4, i5},   i6)
  ({i4, i5, i6},   i7)   ← last allowed (positive = i_{N-2})

Valid : ({i5, i6, i7}, i8)   ← không đổi (1 query/user)
Test  : ({i6, i7, i8}, i9)   ← không đổi (1 query/user)
```

**Các ràng buộc quan trọng:**
- Train positives dừng tại `i_{N-2}` — không bao giờ dùng `i_{N-1}` (valid) hay `i_N` (test) làm positive training → không có data leakage
- Negatives: loại trừ **toàn bộ sequence** của user (gồm cả valid/test positives)
- Valid và test: giữ nguyên 1 query/user như `export_tevatron.py` — protocol evaluation không thay đổi

### Cách chạy

```bash
cd dataset/

# context=3 (default), output → beauty-aug/
python export_tevatron.py beauty --augment

# context=5, output → beauty-cs5-aug/
python export_tevatron.py beauty --augment --context_size 5

# ML-1M: giới hạn 20 samples/user (avg=165 items), output → ml-1m-aug/
python export_tevatron.py ml-1m --augment --max_aug_per_user 20
```

Train sau khi export (tag tự động `aug`):

```bash
./train.sh beauty --data-variant aug
./eval.sh beauty --tag aug
```

### Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--context_size N` | `3` | Số item gần nhất trong query (áp dụng cả hai mode) |
| `--augment` | `False` | Bật sliding window — sinh N-3 samples/user |
| `--max_aug_per_user N` | `None` (unlimited) | Giới hạn samples/user, lấy K vị trí **gần nhất** |
| `--tag TAG` | tự động | Override hậu tố thư mục output |

### Ước tính số samples

| Dataset | Users | Avg seq len | Avg title words | P95 title words | Samples (standard) | Samples (aug, unlimited) |
|---|---|---|---|---|---|---|
| Beauty | 22,363 | 8.9 | 10.9 | 18 | ~22k | ~154k (~7×) |
| Sports | 35,598 | 8.3 | 8.2 | 16 | ~36k | ~224k (~6×) |
| ML-1M | 6,040 | 165.5 | 4.0 | 7 | ~6k | ~980k (~163×) — dùng `--max_aug_per_user 20` |

---

## 8. History Length — Context Size

Mặc định mỗi query chỉ dùng `CONTEXT_SIZE=3` item gần nhất từ lịch sử user. Tăng `--context_size N` để cung cấp thêm context, giúp model nắm bắt pattern dài hạn hơn.

### Cách chạy

```bash
cd dataset/

# context_size=5, output → beauty-cs5/
python export_tevatron.py beauty --context_size 5

# context_size=10, output → beauty-cs10/
python export_tevatron.py beauty --context_size 10

# Kết hợp context_size + mixed negatives
python export_tevatron.py beauty --context_size 5 --neg_strategy mixed
```

Train sau khi export — nhớ tăng `--query-max-len` cho Beauty với cs=10:

```bash
./train.sh beauty --data-variant cs5               # 128 tokens vẫn đủ
./train.sh beauty --data-variant cs10 --query-max-len 160   # Beauty cần tăng
./train.sh sports --data-variant cs10              # Sports 128 vẫn đủ
```

### Query token estimate theo context_size (P95)

Dựa trên số liệu thực tế từ `compute_stats.py` (hệ số tokens/words ≈ 1.3):

| `context_size` | Beauty P95 tokens | Sports P95 tokens | ML-1M P95 tokens | `--query-max-len` |
|---|---|---|---|---|
| 3 *(mặc định)* | 66 | 49 | 25 | 128 |
| 5 | 94 | 70 | 38 | 128 |
| 10 | 146 | 113 | 68 | 160 *(chỉ Beauty)* |

> Chạy `python compute_stats.py beauty sports ml-1m` để xem bảng đầy đủ với avg/P95 words và token estimates.

---

## 10. BM25 Hard Negative Mining

Thay thế hoặc bổ sung random negatives bằng **hard negatives** — các items có title gần giống query (cùng brand, cùng category) nhưng không phải target. Hard negatives khó phân biệt hơn → tín hiệu training mạnh hơn.

### Cơ chế

BM25 index toàn bộ item titles trong corpus. Với mỗi query (title của 3 items gần nhất ghép lại), BM25 trả về top-K items có keyword overlap cao nhất → loại positive → lấy làm hard negatives.

Ví dụ với query `"Ardell LashGrip, Maybelline Great Lash, L'Oreal Voluminous"`:

```
BM25 rank cao → hard negatives:
  Ardell False Lashes Wispies          ← cùng brand "Ardell"
  Maybelline Lash Sensational          ← cùng brand "Maybelline"
  L'Oreal Voluminous Million Lashes    ← cùng brand + từ khóa "Voluminous"
  KISS Lash Couture                    ← cùng từ khóa "Lash"

BM25 rank thấp → tương đương random:
  Neutrogena Sunscreen SPF 50          ← không có keyword nào match
  Revlon Lipstick Red
```

### Ba strategies

| `--neg_strategy` | Pool negatives | Số lượng mặc định |
|---|---|---|
| `random` (mặc định) | Toàn bộ random | 50/query |
| `bm25` | Toàn bộ BM25 hard | 50/query |
| `mixed` | BM25 hard + random | `--num_hard_neg` hard + `--num_random_neg` random (mặc định 10+40=50) |

`mixed` là lựa chọn khuyến nghị: hard negatives tạo tín hiệu mạnh, random negatives giúp training ổn định hơn so với toàn hard.

### Cách chạy

```bash
cd dataset/

# Mixed (mặc định: 10 hard + 40 random), output → beauty-mixed/
python export_tevatron.py beauty --neg_strategy mixed

# Tùy chỉnh ratio: 20 hard + 30 random, output → beauty-mixed/
python export_tevatron.py beauty --neg_strategy mixed --num_hard_neg 20 --num_random_neg 30

# Toàn BM25 hard, output → beauty-bm25/
python export_tevatron.py beauty --neg_strategy bm25

# Kết hợp augmentation + mixed negatives, output → beauty-aug-mixed/
python export_tevatron.py beauty --augment --neg_strategy mixed

# ML-1M: augmentation + mixed, output → ml-1m-aug-mixed/
python export_tevatron.py ml-1m --augment --max_aug_per_user 20 --neg_strategy mixed
```

Train sau khi export (tag tự động `mixed`):

```bash
./train.sh beauty --data-variant mixed
./eval.sh beauty --tag mixed
```

### Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--neg_strategy` | `random` | `random` / `bm25` / `mixed` |
| `--num_hard_neg N` | `10` | Số BM25 hard negatives mỗi query |
| `--num_random_neg N` | `40` | Số random negatives mỗi query |
| `--tag TAG` | tự động | Override hậu tố thư mục output |

> **Lưu ý cold-start:** User không có context (history rỗng) không thể tạo BM25 query có nghĩa. Mọi strategy đều tự động fallback về random negatives cho cold-start samples.

---

## 11. So sánh hai định dạng

| | Tevatron (repLLaMA) | RecBole (SASRec) |
|---|---|---|
| **Biểu diễn user** | `CONTEXT_SIZE = 3` item gần nhất | Toàn bộ history |
| **Lý do giới hạn** | Context length của LLM | Không giới hạn |
| **Loại thông tin** | Text (title của item) | ID của item |
| **Train/valid/test** | 3 file riêng biệt | RecBole tự chia từ `.inter` |
| **Cách chia** | Leave-one-out thủ công | `LS: valid_and_test` |
| **Evaluation** | FAISS search + TREC eval | RecBole built-in |
| **Điểm mạnh** | Hiểu ngữ nghĩa qua text | Tận dụng full history |

> **Đảm bảo so sánh công bằng:** Cả hai mô hình dùng cùng bộ dữ liệu sau filter, cùng chiến lược leave-one-out split, và cùng tập items để rank. Sự khác biệt duy nhất nằm ở cách biểu diễn user và kiến trúc mô hình — đây chính là điều cần so sánh.

---

## 12. Cấu hình nâng cao

Các tham số chỉnh trong `preprocess.py`:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `MIN_INTERACTIONS` | `5` | Ngưỡng k-core filter |
| `CONTEXT_SIZE` | `3` | Số item gần nhất dùng làm query trong Tevatron |
| `MIN_SEQ_LEN` | `3` | Độ dài sequence tối thiểu để giữ user |

Tham số CLI của `export_tevatron.py` (tất cả tùy chọn):

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--context_size N` | `3` | Số item gần nhất trong query — áp dụng cả hai mode |
| `--augment` | `False` | Bật sliding window augmentation |
| `--max_aug_per_user N` | `None` (unlimited) | Giới hạn samples/user khi augment |
| `--neg_strategy` | `random` | `random` / `bm25` / `mixed` |
| `--num_hard_neg N` | `10` | Số BM25 hard negatives mỗi query |
| `--num_random_neg N` | `40` | Số random negatives mỗi query |
| `--tag TAG` | tự động | Override hậu tố thư mục output |

Thêm dataset mới trong `preprocess.py`:

```python
DATASETS["my_dataset"] = {
    "type":        "amazon",            # hoặc "movielens", "steam"
    "review_file": "raw/reviews_My.json.gz",
    "meta_file":   "raw/meta_My.json.gz",
    "name":        "My Dataset",
}
```