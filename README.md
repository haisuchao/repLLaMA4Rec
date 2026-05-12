# repLLaMA for Sequential Recommendation

Áp dụng mô hình **repLLaMA** (dense retrieval dựa trên LLM) vào bài toán **Sequential Recommendation**, so sánh với **SASRec** (Self-Attentive Sequential Recommendation) trên cùng bộ dữ liệu.

---

## Ý tưởng

Bài toán recommendation được reformulate thành bài toán **retrieval**:
- **Query** = lịch sử tương tác gần nhất của user (dưới dạng text)
- **Corpus** = toàn bộ catalog item (mỗi item là một passage)
- **Mục tiêu** = retrieve được item user sẽ tương tác tiếp theo

repLLaMA được fine-tune bằng thư viện **Tevatron v2**, sử dụng **Qwen3-Embedding-0.6B + LoRA** làm backbone thay vì LLaMA-2-7B gốc để phù hợp với phần cứng hạn chế.

---

## Cấu trúc thư mục

```
repLLaMA/
├── dataset/                    # Pipeline tiền xử lý dữ liệu
│   ├── preprocess.py           # 5-core filter, build sequences, leave-one-out split
│   ├── export_tevatron.py      # Xuất định dạng Tevatron (repLLaMA)
│   ├── export_recbole.py       # Xuất định dạng RecBole (SASRec)
│   ├── run_all.py              # Entry point chạy toàn bộ pipeline
│   ├── raw/                    # Dữ liệu thô (tự tải về)
│   ├── export_tevatron_aug.py  # Xuất training data với sliding window augmentation
│   ├── dataset/
│   │   ├── tevatron/           # Output cho repLLaMA
│   │   │   ├── <dataset>/      # Data gốc (1 sample/user)
│   │   │   │   ├── corpus.jsonl
│   │   │   │   ├── train.jsonl
│   │   │   │   ├── valid.jsonl
│   │   │   │   └── test.jsonl
│   │   │   └── <dataset>-aug/  # Data augmented (N-3 samples/user)
│   │   │       └── train.jsonl (augmented), valid/test/corpus (giống gốc)
│   │   └── recbole/            # Output cho SASRec
│   │       └── <dataset>/
│   │           ├── <dataset>.inter
│   │           ├── <dataset>.item
│   │           └── sasrec_<dataset>.yaml
│   └── README.md               # Mô tả chi tiết pipeline tiền xử lý
│
├── tevatron/                   # Thư viện Tevatron v2 (clone từ GitHub)
├── tevatron-env/               # Virtual environment (chứa cả Tevatron và RecBole)
│
├── train.sh                    # Fine-tune repLLaMA
├── eval.sh                     # Đánh giá model: best / latest / base / checkpoint-N
├── run_recbole.py              # Chạy bất kỳ model RecBole (SASRec, GRU4Rec, custom, ...)
├── show_results.py             # Tổng hợp và hiển thị bảng kết quả tất cả experiments
│
├── ds_config.json              # DeepSpeed ZeRO-2 config
└── output/                     # Kết quả training và embedding (tự sinh)
    └── <dataset>/
        └── <model_tag[-variant]>/   # ví dụ: qwen3-embedding-0.6b, qwen3-embedding-0.6b-aug
            ├── checkpoint-*/        # LoRA checkpoints (lưu mỗi save_steps)
            ├── adapter_model.safetensors  # Final model (= checkpoint cuối)
            └── embeddings/
                ├── corpus/          # Dense vectors của corpus
                ├── queries/         # Dense vectors của queries
                └── results/         # Kết quả search và evaluation
```

---

## Yêu cầu phần cứng

| | Tối thiểu | Thực tế sử dụng |
|---|---|---|
| GPU | 8 GB VRAM | NVIDIA RTX 3060 12 GB |
| RAM | 16 GB | 32 GB |
| Disk | 50 GB | ~100 GB (với Sports dataset) |
| CUDA | 11.8+ | CUDA 11.8, Driver 575 |

---

## 1. Cài đặt môi trường

### 1.1 Môi trường repLLaMA (Tevatron)

Virtual environment tại `tevatron-env/` đã được cấu hình sẵn với Python 3.11.7.

```bash
cd /path/to/repLLaMA
source tevatron-env/bin/activate
```

Nếu cần tạo lại từ đầu:

```bash
python3.11 -m venv tevatron-env
source tevatron-env/bin/activate

pip install torch==2.7.1+cu126 torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cu126

pip install nvidia-nccl-cu11          # bắt buộc — torch yêu cầu NCCL
pip install deepspeed==0.18.9
pip install transformers==5.7.0
pip install peft==0.19.1
pip install accelerate==1.13.0
pip install datasets faiss-cpu pyserini pytrec_eval sentencepiece tqdm

# Cài tevatron ở chế độ editable
pip install -e tevatron/
```

> **Lưu ý quan trọng — CUDA không nhận GPU:**
> Driver mới đặt `libcuda.so` tại `/usr/lib/x86_64-linux-gnu/` nhưng path này
> không có trong `LD_LIBRARY_PATH` mặc định. Script `activate` đã được vá để
> tự động thêm path này khi activate env. Nếu tạo lại env từ đầu, thêm vào
> `tevatron-env/bin/activate` sau dòng `export PATH`:
>
> ```bash
> _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
> export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
> ```
>
> Và thêm vào hàm `deactivate()`:
>
> ```bash
> if [ -n "${_OLD_LD_LIBRARY_PATH:-}" ]; then
>     LD_LIBRARY_PATH="${_OLD_LD_LIBRARY_PATH:-}"
>     export LD_LIBRARY_PATH
>     unset _OLD_LD_LIBRARY_PATH
> else
>     unset LD_LIBRARY_PATH
>     unset _OLD_LD_LIBRARY_PATH
> fi
> ```

Kiểm tra môi trường:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → True NVIDIA GeForce RTX 3060
```

### 1.2 Môi trường RecBole (SASRec và các model khác)

RecBole được cài **chung trong `tevatron-env`** — không cần conda env riêng.

```bash
source tevatron-env/bin/activate
python -c "import recbole; print(recbole.__version__)"
# → 1.2.1
```

Nếu cần cài lại:

```bash
source tevatron-env/bin/activate
pip install recbole==1.2.1
```

> **Lưu ý — numpy 2.0 compatibility:** RecBole 1.2.x dùng các numpy aliases đã bị xóa
> trong numpy 2.0 (`np.bool8`, `np.float_`, `np.unicode_`, ...). Script `run_recbole.py`
> đã tích hợp sẵn compatibility patch — không cần làm gì thêm.

---

## 2. Chuẩn bị dữ liệu thô

> **Quan trọng:** Phải tải đúng phiên bản dataset. Amazon có hai phiên bản
> khác nhau — sai phiên bản sẽ cho kết quả sai lệch hoàn toàn.

### Amazon Beauty & Sports — Phiên bản 2014 (chuẩn paper)

Tải từ: **http://jmcauley.ucsd.edu/data/amazon/**

| File | Dataset | ~Kích thước |
|---|---|---|
| `reviews_Beauty_5.json.gz` | Beauty (5-core) | ~198k reviews |
| `meta_Beauty.json.gz` | Beauty metadata | — |
| `reviews_Sports_and_Outdoors_5.json.gz` | Sports (5-core) | ~296k reviews |
| `meta_Sports_and_Outdoors.json.gz` | Sports metadata | ~3 GB |

Đặt vào `dataset/raw/` và đảm bảo tên file khớp với config trong `preprocess.py`:

```python
DATASETS = {
    "beauty": {
        "review_file": "raw/reviews_Beauty_5.json.gz",
        "meta_file":   "raw/meta_Beauty.json.gz",
    },
    "sports": {
        "review_file": "raw/reviews_Sports_and_Outdoors_5.json.gz",
        "meta_file":   "raw/meta_Sports_and_Outdoors.json.gz",
    },
}
```

> ⚠️ **Không dùng** `All_Beauty_5.json.gz` hay dataset Amazon 2018 — đây là
> category khác (niche), chỉ có ~85 items sau filter, không phải bộ "Beauty"
> chuẩn dùng trong các paper (12,101 items).

### MovieLens 1M

Tải từ: https://grouplens.org/datasets/movielens/1m/

Giải nén vào `dataset/raw/ml-1m/`:

```
dataset/raw/ml-1m/
├── ratings.dat    # UserID::MovieID::Rating::Timestamp
└── movies.dat     # MovieID::Title::Genres
```

### Thống kê dataset sau tiền xử lý

| Dataset | Users | Items | Interactions | Avg seq len | Max seq len |
|---|---|---|---|---|---|
| Amazon Beauty | 22,363 | 12,101 | 198,502 | 8.9 | 204 |
| Amazon Sports | 35,598 | 18,357 | 296,337 | 8.3 | 296 |
| MovieLens 1M | 6,040 | 3,416 | 999,611 | 165.5 | 2277 |

---

## 3. Tiền xử lý dữ liệu

```bash
cd dataset/

# Chạy 1 dataset
python run_all.py beauty

# Chạy nhiều dataset
python run_all.py beauty sports ml-1m

# Chạy tất cả
python run_all.py --all
```

Pipeline 5 bước (xem `dataset/README.md` để biết chi tiết):
1. Load dữ liệu thô
2. 5-core filter (iterative — lọc cả user và item)
3. Build sequences (sort theo timestamp, bỏ duplicate liên tiếp)
4. Leave-one-out split (train/valid/test = item N-2, N-1, N)
5. Xuất song song: Tevatron format + RecBole format

**Về evaluation protocol:**
- `valid.jsonl` dùng để chọn checkpoint tốt nhất trong quá trình train
- `test.jsonl` chỉ dùng một lần duy nhất để báo cáo kết quả cuối cùng
- Cả hai model đều dùng cùng protocol leave-one-out → so sánh công bằng

---

## 4. repLLaMA — Training & Evaluation

Activate môi trường trước khi chạy bất kỳ script nào:

```bash
source tevatron-env/bin/activate
```

### 4.1 Tham số của các scripts

**`train.sh`** — Fine-tune model:

```
./train.sh <dataset> [model] [train_group_size] [variant]
```

| Tham số | Giá trị | Mặc định |
|---|---|---|
| `dataset` | `beauty` \| `sports` \| `ml-1m` \| `steam` | bắt buộc |
| `model` | HuggingFace model ID | `Qwen/Qwen3-Embedding-0.6B` |
| `train_group_size` | số nguyên ≥ 2 (1 positive + N negatives) | `8` |
| `variant` | hậu tố experiment, ví dụ `aug` | rỗng |

**`eval.sh`** — Đánh giá model (fine-tuned hoặc base):

```
./eval.sh <dataset> [checkpoint] [model] [split] [variant]
```

| `checkpoint` | Hành động | Output |
|---|---|---|
| `best` *(mặc định)* | Sweep checkpoints trên valid → chọn tốt nhất theo `ndcg_10` → evaluate | `eval_<split>_best.txt` |
| `latest` | Final model sau training, không sweep (nhanh) | `eval_<split>_latest.txt` |
| `base` | Base model **không fine-tune** (zero-shot baseline) — bỏ qua `variant` | `eval_<split>.txt` trong thư mục `-zeroshot` |
| `checkpoint-N` | Checkpoint cụ thể (debug) | `eval_<split>_checkpoint-N.txt` |

| Tham số | Giá trị | Mặc định |
|---|---|---|
| `split` | `valid` \| `test` | `test` |
| `variant` | hậu tố model dir | rỗng |

Khi truyền `variant`, cả `train.sh` và `eval.sh` đều dùng:
- `train.sh`: đọc data từ `dataset/tevatron/<dataset>-<variant>/`, lưu model vào `output/<dataset>/<model_tag>-<variant>/`
- `eval.sh`: tìm model tại `output/<dataset>/<model_tag>-<variant>/`

### 4.2 Workflow khuyến nghị

```bash
source tevatron-env/bin/activate

# Bước 1: Fine-tune
./train.sh beauty

# Bước 2: Tìm best checkpoint → kết quả test (workflow chính)
./eval.sh beauty
```

### 4.3 Các cách chạy eval.sh

```bash
# Tìm best checkpoint tự động (khuyến nghị cho kết quả chính thức)
./eval.sh beauty

# Evaluate final model nhanh (không sweep checkpoints)
./eval.sh beauty latest

# Zero-shot baseline — base model không fine-tune
./eval.sh beauty base

# Debug một checkpoint cụ thể
./eval.sh beauty checkpoint-1000
./eval.sh beauty checkpoint-1000 Qwen/Qwen3-Embedding-0.6B valid

# Augmented experiment
./eval.sh beauty best Qwen/Qwen3-Embedding-0.6B test aug
./eval.sh beauty latest Qwen/Qwen3-Embedding-0.6B test aug
```

Khi checkpoint không tồn tại, `eval.sh` tự liệt kê các checkpoint hiện có.

### 4.4 Ví dụ chạy đầy đủ cho Beauty

```bash
source tevatron-env/bin/activate

./train.sh beauty Qwen/Qwen3-Embedding-0.6B 32
./eval.sh beauty
```

Output:
```
output/beauty/qwen3-embedding-0.6b/
├── checkpoint-*/                          # LoRA checkpoints
└── embeddings/results/
    ├── checkpoint_selection.log           # Bảng valid metric của từng checkpoint
    ├── eval_test_best.txt                 # Kết quả test của best checkpoint ← show_results.py đọc file này
    └── eval_test_latest.txt              # Kết quả test của final model (nếu chạy --latest)
```

### 4.5 Cấu hình training

| Tham số | Giá trị mặc định | Ghi chú |
|---|---|---|
| Base model | Qwen3-Embedding-0.6B | Nhỏ, phù hợp 12 GB VRAM |
| LoRA target | q,k,v,o,down,up,gate | Toàn bộ attention + FFN |
| `per_device_train_batch_size` | 4 | |
| `gradient_accumulation_steps` | 8 | Effective batch size = 32 |
| `train_group_size` | 8 | 1 positive + 7 negatives, điều chỉnh qua tham số CLI |
| Learning rate | 1e-4 | |
| Epochs | 3 | |
| Save steps | 200 | Tần suất lưu checkpoint |
| Query max len | 128 tokens | |
| Passage max len | 196 tokens | |

**Điều chỉnh `train_group_size` và VRAM:**

`train_group_size` = 1 positive + N negatives. Data hiện có 50 negatives/query nên tối đa là 51.
Tăng `train_group_size` làm tăng VRAM tỷ lệ thuận — nếu OOM, giảm `per_device_train_batch_size`
và tăng `gradient_accumulation_steps` để giữ nguyên effective batch size 32:

| `train_group_size` | Negatives | `per_device_train_batch_size` | `gradient_accumulation_steps` |
|---|---|---|---|
| 8 *(mặc định)* | 7 | 4 | 8 |
| 16 | 15 | 2 | 16 |
| 32 | 31 | 1 | 32 |

### 4.6 Zero-shot evaluation (không fine-tune, không training)

Dùng `checkpoint=base` trong `eval.sh` — không cần training, không cần checkpoint:

```bash
./eval.sh beauty base
./eval.sh beauty base Qwen/Qwen3-Embedding-0.6B valid
```

Kết quả lưu tại `output/<dataset>/<model_tag>-zeroshot/embeddings/results/eval_test.txt`.  
Corpus được **cache**: lần đầu encode xong, chạy lại với split khác sẽ bỏ qua bước encode corpus.

### 4.7 Data augmentation — sliding window training

Tạo nhiều training samples/user bằng cách trượt cửa sổ context qua toàn bộ history:

```bash
# Bước 1: Tạo augmented data (window_size mặc định = 3)
cd dataset
python export_tevatron_aug.py beauty
python export_tevatron_aug.py sports
python export_tevatron_aug.py ml-1m --max_aug_per_user 20  # giới hạn với ML-1M (avg=165)
cd ..

# Bước 2: Train với augmented data
./train.sh beauty Qwen/Qwen3-Embedding-0.6B 8 aug

# Bước 3: Evaluate
./select_best.sh beauty Qwen/Qwen3-Embedding-0.6B ndcg_10 aug
```

So sánh với training thông thường (1 sample/user):

| | Standard | Augmented (window=3) |
|---|---|---|
| Samples/user | 1 | N−3 (trung bình ~7 với Beauty) |
| Training samples (Beauty) | ~22k | ~154k |
| Thư mục data | `beauty/` | `beauty-aug/` |
| Thư mục model | `qwen3-embedding-0.6b/` | `qwen3-embedding-0.6b-aug/` |

**Tùy chọn window size:** Thử window lớn hơn để tăng context length:
```bash
cd dataset
python export_tevatron_aug.py beauty --window_size 5 --tag aug-w5
cd ..
./train.sh beauty Qwen/Qwen3-Embedding-0.6B 8 aug-w5
```

> **Lưu ý ML-1M:** avg sequence length = 165, nên dùng `--max_aug_per_user 20` để giới hạn ~20 samples/user và tránh training quá lâu.

### 4.8 Về `--append_eos_token` và `</s>` trong data

Query format trong jsonl: `"Query: item1, item2, item3 </s>"`

Chuỗi `</s>` là legacy từ LLaMA-2 (nơi đây là EOS token thực). Với **Qwen3**, EOS token là
`<|im_end|>` — chuỗi `</s>` trong text chỉ là 3 token thường, không phải EOS.

Flag `--append_eos_token` trong các scripts mới là thứ thực sự thêm `<|im_end|>` vào cuối
sequence sau khi tokenize, đảm bảo `last` pooling lấy đúng token đại diện cho toàn bộ sequence.
**Không được bỏ flag này** — nó áp dụng cho cả query lẫn corpus encoding để nhất quán với training.

---

## 5. RecBole — Training & Evaluation

Dùng script `run_recbole.py` từ thư mục root của project:

```bash
source tevatron-env/bin/activate

# Chạy SASRec (sử dụng config yaml tự sinh)
python run_recbole.py SASRec beauty
python run_recbole.py SASRec sports
python run_recbole.py SASRec ml-1m

# Override config bất kỳ từ CLI
python run_recbole.py SASRec beauty epochs=50 learning_rate=0.0005

# Chạy model khác trong RecBole (dùng default config của RecBole)
python run_recbole.py GRU4Rec beauty
python run_recbole.py BERT4Rec beauty

# Chạy model custom (đặt file trong recbole/, kế thừa SequentialRecommender)
python run_recbole.py recbole.my_model.MyModel beauty
```

Script tự động:
- Load file `dataset/dataset/recbole/<dataset>/sasrec_<dataset>.yaml` nếu tồn tại
- Set đúng `data_path` (tránh lỗi path khi chạy từ project root)
- Patch numpy 2.0 compatibility trước khi import RecBole

### Model custom

Đặt file `.py` trong thư mục `recbole/` (đã có `__init__.py`), kế thừa class RecBole:

```python
# recbole/my_model.py
from recbole.model.sequential_recommender import SASRec

class MyModel(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # override hoặc thêm logic
```

```bash
python run_recbole.py recbole.my_model.MyModel beauty epochs=100
```

### Ghi chú kỹ thuật

> **`MAX_ITEM_LIST_LENGTH: 200`** trong tất cả yaml. SASRec chỉ thấy tối đa 200 item
> gần nhất, tránh OOM với self-attention O(n²) trên sequences dài (ML-1M max=2277).

> **`train_neg_sample_args: ~`** bắt buộc khi dùng `loss_type: CE`. CE loss tự xử lý
> negatives nội bộ (softmax trên toàn bộ items) — không được set negative sampling.

> **`TIME_FIELD: timestamp` + `load_col`** bắt buộc để RecBole sort đúng thứ tự thời
> gian. Thiếu hai dòng này sẽ gây `ValueError: [timestamp] is not exist in interaction`.

---

## 6. So sánh kết quả

Cả hai model đều dùng:
- Cùng bộ dữ liệu sau 5-core filter
- Cùng leave-one-out split (item cuối = test, áp chót = valid)
- Cùng tập items để rank
- Metrics: Recall@10, Recall@20, NDCG@10, NDCG@20, MRR

Điểm khác biệt duy nhất:

| | repLLaMA (Tevatron) | SASRec (RecBole) |
|---|---|---|
| Biểu diễn user | 3 item gần nhất (text) | Toàn bộ history (ID) |
| Biểu diễn item | Title text | Item ID |
| Kiến trúc | LLM encoder + LoRA | Self-Attention |
| Evaluation | FAISS dense retrieval | RecBole full ranking |

---

## 7. Kết quả thực nghiệm

Cập nhật bảng bằng lệnh:

```bash
python show_results.py                 # Chỉ in ra terminal
python show_results.py --update-readme # In + ghi vào README này
```

Thêm kết quả SASRec thủ công: tạo file `output/<dataset>/sasrec/eval_test.txt` với format `<metric>  all  <value>`.

<!-- RESULTS_START -->
### BEAUTY

| Model | NDCG@5 | HR@5 | NDCG@10 | HR@10 | NDCG@20 | HR@20 | MRR@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SASRec | 0.0323 | 0.0548 | 0.0412 | 0.0824 | 0.0506 | 0.1198 | 0.0081 |
| Qwen3-0.6B (zero-shot) | 0.0092 | 0.0182 | 0.0135 | 0.0315 | 0.0179 | 0.0488 | 0.0081 |
| Qwen3-0.6B (fine-tuned) | 0.0251 | 0.0485 | 0.0372 | 0.0861 | 0.0487 | 0.1318 | 0.0224 |
| Llama-3.2-1B (zero-shot) | 0.0012 | 0.0021 | 0.0015 | 0.0030 | 0.0020 | 0.0048 | 0.0011 |
| Llama-3.2-1B (fine-tuned) | 0.0209 | 0.0414 | 0.0313 | 0.0736 | 0.0416 | 0.1148 | 0.0185 |

### SPORTS

| Model | NDCG@5 | HR@5 | NDCG@10 | HR@10 | NDCG@20 | HR@20 | MRR@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-0.6B (zero-shot) | 0.0045 | 0.0090 | 0.0071 | 0.0170 | 0.0100 | 0.0289 | 0.0041 |
| Llama-3.2-1B (zero-shot) | 0.0005 | 0.0008 | 0.0006 | 0.0013 | 0.0009 | 0.0023 | 0.0004 |

### ML-1M

| Model | NDCG@5 | HR@5 | NDCG@10 | HR@10 | NDCG@20 | HR@20 | MRR@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-0.6B (fine-tuned) | 0.0259 | 0.0472 | 0.0396 | 0.0899 | 0.0543 | 0.1485 | 0.0245 |

<!-- RESULTS_END -->

---

## 8. Tình trạng hiện tại

### Đã hoàn thành
- [x] Pipeline tiền xử lý dữ liệu (`dataset/`)
- [x] Export Tevatron format (train/valid/test/corpus)
- [x] Export RecBole format (.inter, .item, .yaml)
- [x] Cài đặt và fix `tevatron-env` (NCCL + CUDA path, torch cu126)
- [x] Scripts gọn: `train.sh` và `eval.sh` (best/latest/base/checkpoint-N)
- [x] Tất cả scripts hỗ trợ tham số `dataset`, `model`, `split`, `variant`
- [x] `train.sh` hỗ trợ tham số `train_group_size` và `variant`
- [x] `--append_eos_token` đồng nhất trên tất cả scripts (corpus + queries)
- [x] Tất cả 3 dataset đã sẵn sàng (Beauty, Sports phiên bản 2014; ML-1M)
- [x] `negative_passages` đã loại bỏ khỏi `valid.jsonl` và `test.jsonl` — tiết kiệm disk
- [x] `MAX_ITEM_LIST_LENGTH` cố định ở 200 trong tất cả yaml và `export_recbole.py`
- [x] Fix parser Amazon 2014: meta file dùng Python dict format (nháy đơn) — thêm fallback `ast.literal_eval` trong `preprocess.py`
- [x] `eval.sh` mode `base` — đánh giá base model không fine-tune (zero-shot)
- [x] `export_tevatron_aug.py` — data augmentation với sliding window, hỗ trợ `--window_size` và `--max_aug_per_user`
- [x] `show_results.py` — tổng hợp kết quả tất cả experiments tự động, cập nhật README
- [x] RecBole cài trong `tevatron-env` (không cần conda riêng)
- [x] `run_recbole.py` — chạy bất kỳ model RecBole, hỗ trợ model custom qua `module.ClassName`
- [x] Fix numpy 2.0 compatibility cho RecBole (`bool8`, `float_`, `unicode_`, ...)
- [x] Fix yaml RecBole: `train_neg_sample_args: ~`, `TIME_FIELD`, `load_col`, `order: TO`

### Cần làm
- [ ] Chạy thực nghiệm đầy đủ (augmented, window size ablation) và ghi lại kết quả so sánh

### Cải tiến tiềm năng
- [ ] Hard negative mining (BM25 hoặc từ top retrieved items) để cải thiện chất lượng training
- [ ] **Ý tưởng 3: Thứ tự items trong query** — xem chi tiết bên dưới

---

## Ý tưởng 3: Thứ tự items trong query (chưa implement)

Đây là ablation thú vị và ít được nghiên cứu. Hiện tại query được format theo thứ tự **chronological** (`i_{n-3}, i_{n-2}, i_{n-1}` → most recent ở cuối). Cần thử thêm:

| Variant | Format query | Thư mục data |
|---|---|---|
| `chrono` *(hiện tại)* | `i_{n-3}, i_{n-2}, i_{n-1}` | `beauty/` |
| `reversed` | `i_{n-1}, i_{n-2}, i_{n-3}` | `beauty-reversed/` |
| `random` | shuffle mỗi lần | `beauty-random/` |

### Cơ chế ảnh hưởng (last pooling + left-padding)

Với Qwen3 dùng **causal attention + last pooling**, token cuối `<|im_end|>` attend đến toàn bộ token trước nó. Về lý thuyết có thể học từ bất kể thứ tự, nhưng thực tế causal LLMs bị ảnh hưởng bởi **"lost in the middle"** — token ở giữa sequence ít được chú ý hơn. Với `CONTEXT_SIZE=3` (rất ngắn), effect này có thể không đáng kể nhưng vẫn đáng kiểm tra.

### Dự đoán

- **Reversed ≈ Chronological**: Model dùng global attention qua EOS token → thứ tự ít quan trọng
- **Reversed tốt hơn Chronological**: Model có recency bias, item gần nhất ở đầu được chú ý hơn → hint để thiết kế query format tốt hơn
- **Random ≈ Chronological/Reversed**: Model làm "bag of items" retrieval, không học sequential pattern
- **Random tệ hơn nhiều**: Thứ tự thực sự quan trọng, model học được sequential patterns từ LLM pre-training

### Implement

Chỉ cần sửa hàm `build_query()` trong `export_tevatron.py` và `export_tevatron_aug.py`:

```python
# Chronological (hiện tại)
"Query: " + ", ".join(context_texts) + " </s>"

# Reversed
"Query: " + ", ".join(reversed(context_texts)) + " </s>"

# Random (dùng random.shuffle — nhớ set seed cho reproducibility)
random.shuffle(context_texts)
"Query: " + ", ".join(context_texts) + " </s>"
```

Export ra thư mục riêng (ví dụ `beauty-reversed/`, `beauty-random/`) → train với `variant=reversed` hoặc `variant=random` → so sánh qua `show_results.py`.

---

## 9. Ghi chú kỹ thuật

### Thư mục gốc `output/` chứa model gì?

HuggingFace Trainer lưu model vào hai nơi khi kết thúc training:
- `checkpoint-{step}/` → checkpoint theo lịch `--save_steps`
- Thư mục gốc `output_dir/` → **bản copy của checkpoint cuối cùng**

Hai bộ trọng số này giống nhau. Thư mục gốc **không phải model tốt nhất** — đó là lý do cần `select_best.sh`.

### Tại sao chỉ 3 item làm query?

Tevatron/repLLaMA encode query thành một vector thông qua LLM. Context length của LLM có giới hạn, và mỗi item title có thể dài. `CONTEXT_SIZE=3` là cân bằng giữa thông tin lịch sử và giới hạn token. SASRec không bị giới hạn này vì dùng item ID (1 token/item) và self-attention.

### Training data chỉ dùng 1 example/user

Mỗi user đóng góp **1 training example** (positive = item N-2). SASRec dùng **tất cả positions** trong sequence (~7× training data). Đây là điểm bất đối xứng cần lưu ý khi so sánh kết quả. Giải pháp tiềm năng: sửa `export_tevatron.py` để sinh training examples từ tất cả positions (1 đến N-2), đảm bảo item N-1 và N không xuất hiện làm positive trong training.

### Amazon 2014 meta file dùng Python dict format

File `meta_Beauty.json.gz` và `meta_Sports_and_Outdoors.json.gz` từ nguồn McAuley 2014 không phải JSON chuẩn — mỗi dòng là một Python dict với dấu nháy đơn:

```python
{'asin': 'B000RG2YFQ', 'title': 'Ardell LashGrip Adhesive', ...}  # Python dict
{"asin": "B000RG2YFQ", "title": "Ardell LashGrip Adhesive", ...}  # JSON chuẩn (2018)
```

`json.loads()` thất bại im lặng trên format này, khiến `item_meta` rỗng và toàn bộ text trong corpus/queries chỉ hiển thị item ID. Fix: hàm `parse_line()` trong `preprocess.py` thử `json.loads` trước, fallback sang `ast.literal_eval` nếu thất bại.

### Vá `trainer.py` của Tevatron

File `tevatron/src/tevatron/retriever/trainer.py` đã được vá để tương thích với `transformers>=5.0` (API `processing_class` thay `tokenizer`, tham số `save_safetensors`). Không cần vá lại khi dùng môi trường đã cài sẵn.
