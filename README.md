# repLLaMA for Sequential Recommendation

Áp dụng mô hình **repLLaMA** (dense retrieval dựa trên LLM) vào bài toán **Sequential Recommendation**, so sánh với **SASRec** (Self-Attentive Sequential Recommendation) trên cùng bộ dữ liệu.

---

## Mục lục

- [Ý tưởng](#ý-tưởng)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu phần cứng](#yêu-cầu-phần-cứng)
- [1. Cài đặt môi trường](#1-cài-đặt-môi-trường)
  - [1.1 Môi trường repLLaMA (Tevatron)](#11-môi-trường-repllama-tevatron)
  - [1.2 Môi trường RecBole](#12-môi-trường-recbole-sasrec-và-các-model-khác)
- [2. Chuẩn bị dữ liệu thô](#2-chuẩn-bị-dữ-liệu-thô)
- [3. Tiền xử lý dữ liệu](#3-tiền-xử-lý-dữ-liệu)
- [4. repLLaMA — Training & Evaluation](#4-repllama--training--evaluation)
  - [4.0 Lựa chọn Base Model](#40-lựa-chọn-base-model)
  - [4.1 Tham số của các scripts](#41-tham-số-của-các-scripts)
  - [4.2 Workflow khuyến nghị](#42-workflow-khuyến-nghị)
  - [4.3 Các cách chạy](#43-các-cách-chạy)
  - [4.4 Ví dụ chạy đầy đủ cho Beauty](#44-ví-dụ-chạy-đầy-đủ-cho-beauty)
  - [4.5 Cấu hình training](#45-cấu-hình-training)
  - [4.6 Zero-shot evaluation](#46-zero-shot-evaluation-không-fine-tune-không-training)
  - [4.7 Tùy chỉnh data export](#47-tùy-chỉnh-data-export)
- [5. Reranker — rankLLaMA](#5-reranker--rankllama)
  - [Ví dụ minh họa — Beauty dataset](#ví-dụ-minh-họa--beauty-dataset)
  - [5.1 Cấu trúc thư mục output](#51-cấu-trúc-thư-mục-output)
  - [5.2 Scripts](#52-scripts)
  - [5.3 Tham số](#53-tham-số)
  - [5.4 Workflow khuyến nghị](#54-workflow-khuyến-nghị)
  - [5.5 Cấu hình training](#55-cấu-hình-training)
  - [5.6 Ghi chú kỹ thuật](#56-ghi-chú-kỹ-thuật)
- [6. RecBole — Training & Evaluation](#6-recbole--training--evaluation)
- [7. So sánh kết quả](#7-so-sánh-kết-quả)
- [8. Kết quả thực nghiệm](#8-kết-quả-thực-nghiệm)
- [9. Tình trạng hiện tại](#9-tình-trạng-hiện-tại)
- [Ý tưởng 3: Thứ tự items trong query](#ý-tưởng-3-thứ-tự-items-trong-query-chưa-implement)
- [10. Ghi chú kỹ thuật](#10-ghi-chú-kỹ-thuật)

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
│   ├── export_recbole.py       # Xuất định dạng RecBole — data → dataset/recbole/, config → recbole/props/
│   ├── compute_stats.py        # Thống kê title words, query token estimates
│   ├── run_all.py              # Entry point chạy toàn bộ pipeline
│   ├── raw/                    # Dữ liệu thô (tự tải về)
│   └── dataset/
│       ├── tevatron/           # Output cho repLLaMA
│       │   └── <dataset[-tag]>/    # tag tự sinh: cs5, aug, mixed, cs5-aug-mixed, ...
│       │       ├── corpus.jsonl
│       │       ├── train.jsonl
│       │       ├── valid.jsonl
│       │       └── test.jsonl
│       └── recbole/            # Data files cho RecBole (.inter, .item)
│           └── <dataset>/
│               ├── <dataset>.inter
│               └── <dataset>.item
│
├── recbole/                    # RecBole workspace (như MiaSRec)
│   ├── recbole/                # Thư viện RecBole (vendored — copy từ MiaSRec)
│   ├── cmamba4rec.py           # Custom model (CMamba4Rec)
│   ├── props/                  # YAML configs
│   │   ├── beauty/
│   │   │   ├── sasrec.yaml     # Auto-generated bởi export_recbole.py
│   │   │   └── cmamba4rec.yaml # Config custom model
│   │   ├── sports/
│   │   │   ├── sasrec.yaml
│   │   │   └── cmamba4rec.yaml
│   │   └── ml-1m/
│   │       ├── sasrec.yaml
│   │       └── cmamba4rec.yaml
│   └── output/                 # Kết quả training RecBole (tự sinh)
│       ├── saved/              # Model checkpoints (.pth)
│       ├── log/                # Training logs
│       └── log_tensorboard/    # Tensorboard logs
│
├── tevatron/                   # Thư viện Tevatron v2 (clone từ GitHub)
├── tevatron-env/               # Virtual environment
│
├── train.sh                    # Fine-tune repLLaMA
├── eval.sh                     # Đánh giá model: best / latest / base / checkpoint-N
├── run_recbole.py              # Entry point RecBole (SASRec, GRU4Rec, custom model, ...)
├── show_results.py             # Tổng hợp và hiển thị bảng kết quả tất cả experiments
│
├── ds_config.json              # DeepSpeed ZeRO-2 config
└── output/                     # Kết quả training và embedding repLLaMA (tự sinh)
    └── <dataset>/
        └── <model_tag[-variant]>/
            ├── train_config.json    # Config lưu bởi train.sh (dùng bởi show_results.py)
            ├── checkpoint-*/        # LoRA checkpoints
            ├── adapter_model.safetensors
            └── embeddings/results/
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
pip install datasets faiss-cpu pyserini pytrec_eval sentencepiece qwen_omni_utils tqdm

# Flash Attention 2 — bắt buộc (xem lưu ý bên dưới)
pip install flash-attn==2.8.3 --no-build-isolation

# Cài tevatron ở chế độ editable
pip install -e tevatron/
```

> **Lưu ý quan trọng — Flash Attention 2:**
>
> Tevatron mặc định dùng `attn_implementation="flash_attention_2"` cho cả training lẫn eval.
> Nếu không cài `flash-attn`, **cả `train.sh` và `eval.sh` đều báo lỗi ngay khi khởi động.**
>
> **Yêu cầu để cài flash-attn:**
> - GPU có CUDA Compute Capability ≥ 8.0 (Ampere trở lên — RTX 30xx, 40xx, A100, H100)
> - PyTorch đã cài với CUDA support (`torch.cuda.is_available() == True`)
> - BF16 hoặc FP16 (không hỗ trợ FP32)
>
> **Nếu GPU không hỗ trợ flash-attn** (Compute Capability < 8.0 — GTX 10xx, 20xx):
> Thêm flag `--attn_implementation sdpa` vào lệnh trong `train.sh` và `eval.sh`,
> hoặc sửa default trong `tevatron/src/tevatron/retriever/arguments.py`:
> ```python
> attn_implementation: Optional[str] = field(
>     default="sdpa",   # thay "flash_attention_2" bằng "sdpa"
>     ...
> )
> ```
>
> Kiểm tra flash-attn đã cài đúng:
> ```bash
> python -c "import flash_attn; print('flash-attn version:', flash_attn.__version__)"
> # → flash-attn version: 2.8.3
> ```

> **Lưu ý quan trọng — CUDA không nhận GPU:**
> Có hai nguyên nhân phổ biến:
>
> **1. `libcuda.so` không nằm trong `LD_LIBRARY_PATH`** — Driver mới đặt file này tại
> `/usr/lib/x86_64-linux-gnu/`. Script `activate` và các script `train.sh`/`eval.sh`
> đã được vá để set đúng path này.
>
> **2. CUDA toolkit cũ (11.x) xung đột với torch cu126** — Nếu shell profile có
> `LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:...`, các thư viện CUDA 11.8 sẽ được
> load trước CUDA 12.6 mà torch tự bundle → `torch.cuda.is_available()` trả về False.
>
> Fix (đã áp dụng trong `activate`, `train.sh`, `eval.sh`): set `LD_LIBRARY_PATH`
> chỉ chứa path driver, **không** append CUDA toolkit cũ:
>
> ```bash
> _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
> export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"
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

| Dataset | Users | Items | Interactions | Avg seq len | Max seq len | Avg title words | P95 title words |
|---|---|---|---|---|---|---|---|
| Amazon Beauty | 22,363 | 12,101 | 198,502 | 8.9 | 204 | 10.9 | 18 |
| Amazon Sports | 35,598 | 18,357 | 296,337 | 8.3 | 296 | 8.2 | 16 |
| MovieLens 1M | 6,040 | 3,416 | 999,611 | 165.5 | 2277 | 4.0 | 7 |

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

Pipeline 5 bước (xem **[dataset/README.md](dataset/README.md)** để biết chi tiết):
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

### 4.0 Lựa chọn Base Model

Bảng dưới liệt kê các model có thể dùng làm backbone cho `train.sh`. Ký hiệu **✓ cache** = đã download sẵn, không cần tải thêm.

| Model | Base | Params | Mục đích gốc | Max seq len | Hidden dim | Ưu điểm | Nhược điểm | Phù hợp |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- | :--- | :---: |
| `Qwen/Qwen3-Embedding-0.6B` ✓ cache | Qwen3 decoder | 0.6B | Dense retrieval · Embedding | 32 768 | 1 024 | Embedding-specific, nhẹ, zero-shot tốt | Nhỏ nhất trong dòng Embedding | ⭐⭐⭐ |
| `Qwen/Qwen3-Embedding-4B` ✓ cache | Qwen3 decoder | 4B | Dense retrieval · Embedding | 32 768 | 2 560 | Cùng dòng Embedding nhưng lớn hơn, zero-shot tốt hơn đáng kể | VRAM sát giới hạn 12 GB (batch=1) | ⭐⭐⭐ |
| `meta-llama/Llama-3.2-1B` ✓ cache | Llama 3.2 decoder | 1.24B | Text generation | 131 072 | 2 048 | Context rất dài, nhẹ, đã thử nghiệm | Không phải embedding model → zero-shot kém | ⭐⭐ |
| `Qwen/Qwen2.5-1.5B` ✓ cache | Qwen2.5 decoder | 1.5B | Text generation | 32 768 | 1 536 | Nhẹ, đa ngôn ngữ tốt | Không phải embedding model | ⭐⭐ |
| `Qwen/Qwen2.5-3B` ✓ cache | Qwen2.5 decoder | 3B | Text generation | 32 768 | 2 048 | Cân bằng size/performance tốt | Không phải embedding model | ⭐⭐ |
| `meta-llama/Llama-3.2-3B` *(cần tải)* | Llama 3.2 decoder | 3.21B | Text generation | 131 072 | 3 072 | Context rất dài, mạnh hơn 1B | Cần download (~6 GB), VRAM sát giới hạn | ⭐⭐ |
| `microsoft/Phi-3-mini-4k-instruct` ✓ cache | Phi-3 decoder | 3.8B | Text generation | 4 096 | 3 072 | Mạnh so với size | Max seq len ngắn (4 096) — không đủ cho history dài | ⭐ |
| `mistralai/Mistral-7B-v0.3` ✓ cache | Mistral decoder | 7.24B | Text generation | 32 768 | 4 096 | Mạnh, nhiều embedding model build trên đây | **OOM** — 16 GB BF16, vượt 12 GB VRAM | ✗ |

**Ghi chú về mức độ phù hợp:**
- ⭐⭐⭐ **Tốt nhất** — Embedding-specific, fit VRAM, khuyến nghị dùng trước
- ⭐⭐ **Tốt** — Text generation model, cần fine-tune nhiều hơn để đạt chất lượng embedding tương đương; fit VRAM với batch=1
- ⭐ **Dùng được nhưng có hạn chế** — Vấn đề kỹ thuật cụ thể cần lưu ý
- ✗ **Không khả thi** với phần cứng hiện tại

**Cách chọn nhanh:**
```bash
# Tốt nhất — embedding model, đã có sẵn
./train.sh beauty --model Qwen/Qwen3-Embedding-0.6B          # nhẹ, nhanh
./train.sh beauty --model Qwen/Qwen3-Embedding-4B --tag 4b   # mạnh hơn, sát VRAM

# Thay thế — general LLM, cần fine-tune nhiều hơn
./train.sh beauty --model meta-llama/Llama-3.2-1B --tag llama1b
./train.sh beauty --model Qwen/Qwen2.5-3B --tag qwen3b
```

### 4.1 Tham số của các scripts

**`train.sh`** — Fine-tune model:

```
./train.sh <dataset> [--model MODEL] [--group-size N] [--epochs N] [--query-max-len N] [--data-variant TAG] [--tag TAG]
```

| Tham số | Giá trị | Mặc định |
|---|---|---|
| `dataset` | `beauty` \| `sports` \| `ml-1m` \| `steam` | bắt buộc |
| `--model` | HuggingFace model ID | `Qwen/Qwen3-Embedding-0.6B` |
| `--group-size` | số nguyên ≥ 2 (1 positive + N-1 negatives) | `8` |
| `--epochs` | số training epochs | `3` |
| `--query-max-len` | số token tối đa của query; tăng khi dùng `context_size` lớn | `128` |
| `--data-variant` | đọc training data từ `dataset/tevatron/<dataset>-<TAG>/` | rỗng |
| `--tag` | hậu tố output dir (`output/<dataset>/<model_tag>-<TAG>/`); mặc định lấy từ `--data-variant` | rỗng |

**`eval.sh`** — Đánh giá model (fine-tuned hoặc base):

```
./eval.sh <dataset> [checkpoint] [--model MODEL] [--split SPLIT] [--tag TAG] [--metric METRIC]
```

| `checkpoint` (positional) | Hành động | Output |
|---|---|---|
| `best` *(mặc định)* | Sweep checkpoints trên valid → chọn tốt nhất theo `--metric` → evaluate trên `--split` | `eval_<split>_best.txt` |
| `latest` | Final model sau training, không sweep (nhanh) | `eval_<split>_latest.txt` |
| `base` | Base model **không fine-tune** (zero-shot baseline) — bỏ qua `--tag` | `eval_<split>.txt` trong thư mục `-zeroshot` |
| `checkpoint-N` | Checkpoint cụ thể (debug) | `eval_<split>_checkpoint-N.txt` |

| Flag | Giá trị | Mặc định |
|---|---|---|
| `--model` | HuggingFace model ID | `Qwen/Qwen3-Embedding-0.6B` |
| `--split` | `valid` \| `test` | `test` |
| `--tag` | hậu tố model dir — khớp với `--tag` khi train | rỗng |
| `--metric` | metric chọn best checkpoint: `ndcg_5/10/20`, `hr_5/10/20`, `mrr_5/10/20` | `ndcg_10` |

### 4.2 Workflow khuyến nghị

```bash
source tevatron-env/bin/activate

# Bước 1: Fine-tune (default model, default settings)
./train.sh beauty

# Bước 2: Tìm best checkpoint → kết quả test (workflow chính)
./eval.sh beauty
```

### 4.3 Các cách chạy

```bash
# ── train.sh ──────────────────────────────────────────────────
./train.sh beauty                                         # all defaults
./train.sh beauty --model Qwen/Qwen3-Embedding-4B        # model khác
./train.sh beauty --group-size 32 --tag gs32             # nhiều negatives hơn
./train.sh beauty --epochs 5 --tag ep5                   # train lâu hơn
./train.sh beauty --data-variant aug                     # augmented data (tag = aug)
./train.sh beauty --data-variant mixed                   # BM25 hard negatives
./train.sh beauty --data-variant cs5                     # history length = 5 items
./train.sh beauty --data-variant cs10 --query-max-len 160    # history length = 10 items (Beauty cần tăng len)
./train.sh beauty --data-variant w3 --tag w3-gs16 \
    --group-size 16                                      # data-variant + output tag riêng
./train.sh beauty --model Qwen/Qwen3-Embedding-4B --tag 4b   # model khác, tag tùy chỉnh

# ── eval.sh ───────────────────────────────────────────────────
./eval.sh beauty                          # best checkpoint → test (ndcg_10)
./eval.sh beauty latest                   # final model → test
./eval.sh beauty base                     # zero-shot baseline
./eval.sh beauty checkpoint-1000          # debug checkpoint cụ thể

./eval.sh beauty --split valid            # evaluate trên valid set
./eval.sh beauty --tag aug                # model train với augmented data
./eval.sh beauty latest --tag aug         # augmented, final model
./eval.sh beauty --tag cs5                # history length = 5 items
./eval.sh beauty --metric hr_10           # chọn best theo HR@10 thay vì NDCG@10

./eval.sh beauty checkpoint-1000 \
    --model Qwen/Qwen3-Embedding-0.6B \
    --split valid                         # đầy đủ tham số
./eval.sh beauty --model Qwen/Qwen3-Embedding-4B --tag 4b  # model 4B đã train
```

Khi checkpoint không tồn tại, `eval.sh` tự liệt kê các checkpoint hiện có.

### 4.4 Ví dụ chạy đầy đủ cho Beauty

```bash
source tevatron-env/bin/activate

# Default: Qwen3-0.6B, group-size=8
./train.sh beauty
./eval.sh beauty

# Group size lớn hơn (31 negatives thay vì 7)
./train.sh beauty --group-size 32 --tag gs32
./eval.sh beauty --tag gs32
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
| Save steps | 2000 | Tần suất lưu checkpoint |
| Query max len | 128 tokens | |
| Passage max len | 196 tokens | |
| `--append_eos_token` | bật | Thêm `tokenizer.eos_token_id` vào cuối mỗi sequence sau tokenize — đảm bảo `last` pooling lấy đúng token EOS của model. EOS khác nhau theo model: Qwen3=`<\|im_end\|>`, LLaMA-3=`<\|eot_id\|>`. **Không được bỏ flag này.** |
| `--pooling last` | last | Last token pooling — lấy vector của EOS token làm embedding đại diện |
| `--normalize` | bật | L2-normalize embedding trước khi tính similarity |

**Auto-adjust batch size theo model:**

`train.sh` tự động chỉnh `per_device_train_batch_size` và `gradient_accumulation_steps` dựa trên model để giữ effective batch size = 32 và tránh OOM:

| Model | `per_device_batch` | `grad_accum` | `eval_batch` |
|---|---|---|---|
| Qwen3-0.6B *(mặc định)* | 4 | 8 | 64 |
| Qwen3-4B, Qwen2.5-3B, Llama-3.2-1B/3B | 1 | 32 | 16 |

**Điều chỉnh `--group-size` và VRAM:**

`--group-size` = 1 positive + N negatives. Data hiện có 50 negatives/query nên tối đa là 51.
Tăng `--group-size` làm tăng VRAM tỷ lệ thuận. Với Qwen3-0.6B, các mức điều chỉnh thủ công:

| `--group-size` | Negatives | `per_device_train_batch_size` | `gradient_accumulation_steps` |
|---|---|---|---|
| 8 *(mặc định)* | 7 | 4 | 8 |
| 16 | 15 | 2 | 16 |
| 32 | 31 | 1 | 32 |

Với model lớn (4B, 1B), `train.sh` đã set sẵn batch=1, accum=32 nên `--group-size` có thể tăng tự do đến khi OOM.

**Chọn `--query-max-len` theo `context_size`:**

`context_size` là số item gần nhất dùng làm query, được set khi export data bằng `--context_size N`.
Bảng dưới dựa trên P95 token estimate (×1.3 từ word count) thực tế của từng dataset:

| `context_size` | Beauty P95 tokens | Sports P95 tokens | ML-1M P95 tokens | `--query-max-len` |
|---|---|---|---|---|
| 3 *(mặc định)* | 66 | 49 | 25 | 128 *(mặc định, đủ cho cả 3 dataset)* |
| 5 | 94 | 70 | 38 | 128 *(vẫn đủ)* |
| 10 | 146 | 113 | 68 | 160 *(cần tăng với Beauty)* |

```bash
# context_size=3 — không cần thay đổi gì
./train.sh beauty --data-variant cs3   # mặc định đã đúng

# context_size=5 — 128 vẫn đủ
./train.sh beauty --data-variant cs5

# context_size=10 — cần tăng query_max_len cho Beauty
./train.sh beauty --data-variant cs10 --query-max-len 160
./train.sh sports --data-variant cs10             # 128 vẫn đủ với Sports
```

### 4.6 Zero-shot evaluation (không fine-tune, không training)

Dùng `checkpoint=base` trong `eval.sh` — không cần training, không cần checkpoint:

```bash
./eval.sh beauty base
./eval.sh beauty base --model Qwen/Qwen3-Embedding-0.6B --split valid
./eval.sh beauty base --model Qwen/Qwen3-Embedding-4B
```

Kết quả lưu tại `output/<dataset>/<model_tag>-zeroshot/embeddings/results/eval_test.txt`.  
Corpus được **cache**: lần đầu encode xong, chạy lại với split khác sẽ bỏ qua bước encode corpus.

### 4.7 Tùy chỉnh data export

`export_tevatron.py` là script thống nhất điều khiển 3 chiều độc lập: **history length**, **query augmentation**, và **negative sampling**. Ba chiều này có thể kết hợp tự do.

**History Length — tăng số item trong query:**

```bash
cd dataset

# context_size=5 — 5 items gần nhất (tag tự động: cs5)
python export_tevatron.py beauty --context_size 5
python export_tevatron.py sports --context_size 5

# context_size=10 — 10 items gần nhất (tag tự động: cs10)
python export_tevatron.py beauty --context_size 10
cd ..

# Train — nhớ tăng --query-max-len cho Beauty với cs=10
./train.sh beauty --data-variant cs5
./train.sh beauty --data-variant cs10 --query-max-len 160
./eval.sh beauty --tag cs5
./eval.sh beauty --tag cs10
```

**Query augmentation — sliding window:**

```bash
cd dataset

# context=3 (default), ~7 samples/user với Beauty (tag tự động: aug)
python export_tevatron.py beauty --augment
python export_tevatron.py sports --augment

# ML-1M: giới hạn 20 samples/user (avg seq len = 165)
python export_tevatron.py ml-1m --augment --max_aug_per_user 20

# context=5, augmented (tag tự động: cs5-aug)
python export_tevatron.py beauty --augment --context_size 5
cd ..

./train.sh beauty --data-variant aug
./eval.sh beauty --tag aug
```

**BM25 Hard Negative Mining:**

```bash
cd dataset
python export_tevatron.py beauty --neg_strategy mixed        # 10 hard + 40 random (tag: mixed)
python export_tevatron.py beauty --neg_strategy bm25         # 50 BM25 hard (tag: bm25)
python export_tevatron.py beauty --neg_strategy mixed \
    --num_hard_neg 20 --num_random_neg 30                    # tùy chỉnh ratio
cd ..

./train.sh beauty --data-variant mixed
./eval.sh beauty --tag mixed
```

**Kết hợp tự do — ví dụ:**

```bash
cd dataset
# context_size=5 + BM25 mixed negatives → tag: cs5-mixed
python export_tevatron.py beauty --context_size 5 --neg_strategy mixed

# augmentation + BM25 mixed → tag: aug-mixed
python export_tevatron.py beauty --augment --neg_strategy mixed

# context_size=5 + augmentation + BM25 mixed → tag: cs5-aug-mixed
python export_tevatron.py beauty --context_size 5 --augment --neg_strategy mixed
cd ..

./train.sh beauty --data-variant cs5-mixed
./train.sh beauty --data-variant aug-mixed
./train.sh beauty --data-variant cs5-aug-mixed
```

**Bảng tổng hợp tag tự động:**

| Tùy chọn | Auto tag | Data dir | Train command |
|---|---|---|---|
| mặc định | `""` | `beauty/` | `./train.sh beauty` |
| `--context_size 5` | `cs5` | `beauty-cs5/` | `./train.sh beauty --data-variant cs5` |
| `--context_size 10` | `cs10` | `beauty-cs10/` | `./train.sh beauty --data-variant cs10 --query-max-len 160` |
| `--augment` | `aug` | `beauty-aug/` | `./train.sh beauty --data-variant aug` |
| `--context_size 5 --augment` | `cs5-aug` | `beauty-cs5-aug/` | `./train.sh beauty --data-variant cs5-aug` |
| `--neg_strategy mixed` | `mixed` | `beauty-mixed/` | `./train.sh beauty --data-variant mixed` |
| `--augment --neg_strategy mixed` | `aug-mixed` | `beauty-aug-mixed/` | `./train.sh beauty --data-variant aug-mixed` |
| `--context_size 5 --augment --neg_strategy mixed` | `cs5-aug-mixed` | `beauty-cs5-aug-mixed/` | `./train.sh beauty --data-variant cs5-aug-mixed` |

> **Lưu ý ML-1M:** avg sequence length = 165, nên dùng `--max_aug_per_user 20` khi augmentation để tránh training quá lâu.
>
> **Lưu ý `--query-max-len`:** Chỉ cần tăng với Beauty + `context_size=10` (P95=146 tokens). Sports và ML-1M vẫn đủ với 128 ở mọi context_size ≤ 10.

---

## 5. Reranker — rankLLaMA

Reranker là **stage 2** của two-stage pipeline: sau khi repLLaMA retrieve top-K candidates, một cross-encoder rerank lại để cải thiện ranking quality.

```
Stage 1 — repLLaMA (bi-encoder)
  query_embedding = repLLaMA(user_history)
  top_K = FAISS.search(query_embedding, K=100)

Stage 2 — rankLLaMA (cross-encoder)
  score(i) = CrossEncoder([user_history; item_title_i])  for i in top_K
  final_ranking = sort(top_K, by=score)
```

**Tại sao cross-encoder mạnh hơn bi-encoder ở reranking?**

Bi-encoder encode query và item **độc lập** → similarity qua dot product. Cross-encoder encode `[query; item]` **chung một lần** → full attention giữa mọi token của cả hai → bắt được fine-grained matching signal mà bi-encoder không có.

**Ceiling analysis (Beauty, qwen3-0.6b):**

| Depth retriever | Recall@K | Tiềm năng HR@10 nếu rerank hoàn hảo |
|---|---|---|
| 100 (mặc định) | 0.2673 | **3.1×** (từ 0.0861) |

---

### Ví dụ minh họa — Beauty dataset

Dưới đây là ví dụ thực từ Beauty test set, minh họa đầu vào/đầu ra của từng stage.

#### Stage 1 — repLLaMA (Retriever)

User đã mua 3 items gần nhất. Hệ thống cần dự đoán item tiếp theo.

**Đầu vào — Query (text):**
```
Query: Pineapple Pumpkin Enzyme Skin Peel- Enhanced with Papaya Extract
       & Alpha Hydroxy Acids (Professional Chemical Peel),
       LAVANILA The Healthy Deodorant Vanilla Lavender 2.0 oz,
       Waxelene 2oz jar </s>
```

repLLaMA encode query thành 1 dense vector, sau đó tìm kiếm FAISS trên toàn bộ corpus (12,101 items).

**Đầu ra — Top-6 candidates (similarity score):**

```
Rank  Score   Item
────  ──────  ────────────────────────────────────────────────────────────────
 [1]  0.9365  Waxelene 2oz jar                                  ← item trong history!
 [2]  0.9241  Pineapple Pumpkin Enzyme Skin Peel (...)           ← item trong history!
 [3]  0.9076  Aztec Secrets: Indian Healing Bentonite Clay, 2 lbs
 [4]  0.9075  Aztec Secrets: Indian Healing Bentonite Clay, 2 lbs  (duplicate)
 [5]  0.9068  Salicylic Acid 20% Gel Peel - Enhanced with Tea Tree Oil (...)  ★ POSITIVE
 [6]  0.9043  TCA 15% Gel Peel - Salicylic Acid 5% Enhanced with Botanical Extracts (...)
```

> **Vấn đề của bi-encoder:** Retriever xếp hạng cao các item **đã có trong history** của user (rank 1, 2) vì embedding của chúng tương tự query. Nó không phân biệt được "items user đã mua" vs "items user muốn mua tiếp". Item đúng (★) bị đẩy xuống rank 5.

---

#### Stage 2 — rankLLaMA (Reranker)

Với mỗi candidate trong top-K, reranker nhận một cặp **(query, candidate)** ghép lại và tính relevance score.

**Đầu vào — Cặp (query; candidate) cho từng item:**

```
# Candidate rank [1] — Waxelene 2oz jar
Query: Pineapple Pumpkin Enzyme Skin Peel (...), LAVANILA (...), Waxelene 2oz jar </s>
Passage: Waxelene 2oz jar
→ score = ?   [cross-encoder xử lý cả 2 cùng lúc, full attention]

# Candidate rank [5] — Salicylic Acid Chemical Peel (POSITIVE)
Query: Pineapple Pumpkin Enzyme Skin Peel (...), LAVANILA (...), Waxelene 2oz jar </s>
Passage: Salicylic Acid 20% Gel Peel - Enhanced with Tea Tree Oil & Green Tea Extract
→ score = ?
```

Tokenized input thực tế (sau khi ghép, truncate 256 tokens, append EOS):
```
[token_1][token_2]...[query_tokens]...[</s>][passage_tokens]...[<|im_end|>]
                                                                ↑ EOS — classification head dùng hidden state ở đây
```

**Đầu ra — Reranked list (kỳ vọng sau reranking):**

```
Rank  Score    Item
────  ───────  ────────────────────────────────────────────────────────────────
 [1]  (cao)    Salicylic Acid 20% Gel Peel - Enhanced with Tea Tree Oil (...)  ★ POSITIVE
 [2]  (cao)    TCA 15% Gel Peel - Salicylic Acid 5% Enhanced with Botanical Extracts
 [3]  (thấp)   Aztec Secrets: Indian Healing Bentonite Clay, 2 lbs
 [4]  (thấp)   Aztec Secrets: Indian Healing Bentonite Clay, 2 lbs
 [5]  (thấp)   Pineapple Pumpkin Enzyme Skin Peel (...)           ← item trong history → downrank
 [6]  (thấp)   Waxelene 2oz jar                                   ← item trong history → downrank
```

> **Lợi thế của cross-encoder:** Full attention cho phép model nhận biết rằng Waxelene và Pineapple Peel **đã xuất hiện trong query** (lịch sử của user) → downrank. Đồng thời nhận ra Salicylic Acid Chemical Peel là *loại sản phẩm tiếp theo hợp lý* dựa trên context mua sắm → uprank lên vị trí 1.

---

### 5.1 Cấu trúc thư mục output

```
output/<dataset>/<model_tag>-reranker/
├── train_data/
│   ├── train_queries.pkl        # Encoded train queries (cache, reuse được)
│   ├── train_rank.txt           # FAISS search output thô
│   ├── train_rank.trec          # Converted TREC format (deduplicated)
│   └── reranker_train.jsonl     # Training data: query + pos + hard negs
├── checkpoint-500/              # Reranker LoRA checkpoints
├── adapter_model.safetensors    # Final reranker model (= checkpoint cuối)
└── inference/
    ├── test_pairs.jsonl         # Flat (query, candidate) pairs để chạy reranker
    ├── test_reranked.txt        # Reranker output thô: qid\tdocid\tscore
    ├── test_reranked.trec       # TREC format (có rank)
    └── eval_test_reranked.txt   # Metrics + bảng so sánh Retriever vs Reranker
```

---

### 5.2 Scripts

| Script | Vai trò |
|---|---|
| `prepare_rerank_data.py` | Chuyển đổi dữ liệu từ retriever output sang format reranker |
| `train_reranker.sh` | Train cross-encoder reranker (5 bước, idempotent) |
| `rerank.sh` | Rerank top-K + evaluate + in bảng so sánh |

**`prepare_rerank_data.py`** — hai mode:

| Mode | Input | Output |
|---|---|---|
| `train` | `train.jsonl` + `train_rank.trec` + `corpus.jsonl` | `reranker_train.jsonl` (query + pos + hard negs) |
| `infer` | `test.jsonl` + `test_rank.trec` + `corpus.jsonl` | flat pairs jsonl (1 dòng / (query, candidate)) |

---

### 5.3 Tham số

**`train_reranker.sh`:**

```
./train_reranker.sh <dataset> [--model MODEL] [--tag RTAG]
                               [--depth N] [--group-size N] [--epochs N]
```

| Tham số | Mặc định | Ghi chú |
|---|---|---|
| `--model` | `Qwen/Qwen3-Embedding-0.6B` | Base model (khuyến nghị dùng cùng với retriever) |
| `--tag RTAG` | `""` | Tag retriever đã train (ví dụ: `aug-5`); để trống = standard model |
| `--depth N` | `100` | Số candidates/query để mine hard negatives |
| `--group-size N` | `8` | 1 pos + (N-1) hard negs mỗi query khi train |
| `--epochs N` | `3` | Số training epochs |

**`rerank.sh`:**

```
./rerank.sh <dataset> [--model MODEL] [--tag RTAG] [--split SPLIT]
            [--retriever-trec FILE] [--reranker-ckpt DIR]
```

| Tham số | Mặc định | Ghi chú |
|---|---|---|
| `--tag RTAG` | `""` | Phải khớp với `--tag` khi chạy `train_reranker.sh` |
| `--split` | `test` | `valid` hoặc `test` |
| `--retriever-trec FILE` | auto | Override trec file (mặc định tự tìm từ `eval_<split>_best.txt`) |
| `--reranker-ckpt DIR` | final model | Override checkpoint cụ thể (ví dụ: debug checkpoint-500) |

---

### 5.4 Workflow khuyến nghị

```bash
source tevatron-env/bin/activate

# Điều kiện tiên quyết: retriever đã train và eval xong
./train.sh beauty
./eval.sh beauty

# Bước 1: Train reranker (~30-40 phút)
./train_reranker.sh beauty

# Bước 2: Rerank + evaluate
./rerank.sh beauty
```

Có thể chạy lại `train_reranker.sh` an toàn sau khi fix lỗi — các bước đã hoàn thành (corpus cache, train query embeddings, train_rank.trec) sẽ bị skip tự động.

---

### 5.5 Cấu hình training

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Kiến trúc | `AutoModelForSequenceClassification` | Cross-encoder, output 1 score/pair |
| LoRA rank / alpha | 16 / 64 | Giống retriever |
| LoRA target | q,k,v,o,down,up,gate | Toàn bộ attention + FFN |
| `per_device_train_batch_size` | 2 | Số queries per GPU per step |
| `gradient_accumulation_steps` | 16 | Effective batch = 32 queries/step |
| `train_group_size` | 8 | 1 pos + 7 hard negs mỗi query |
| Input max length | 256 tokens | `[query_history; item_title]` ghép lại |
| Learning rate | 1e-4 | |
| Warmup steps | 100 | |
| Save steps | 500 | |
| `--append_eos_token` | bật | Thêm EOS cuối chuỗi làm anchor cho classification head |
| `num_labels` | 1 | Single regression score — **bắt buộc** để loss tính đúng |

**Auto-adjust batch size theo model:**

| Model | `per_device_batch` | `grad_accum` |
|---|---|---|
| Qwen3-0.6B *(mặc định)* | 2 | 16 |
| Qwen3-4B, Qwen2.5-3B, Llama-3.2-* | 1 | 32 |

---

### 5.6 Ghi chú kỹ thuật

**Hard negatives từ retriever:** Training data của reranker dùng top-K retrieved items (trừ positive) làm hard negatives. Đây là những items retriever nghĩ là relevant nhưng không phải ground truth — reranker phải học phân biệt fine-grained, khó hơn random negatives nhiều.

**`num_labels=1` trong `build()`:** Tevatron's `RerankerModel.build()` gốc không pass `num_labels`, gây `AutoModelForSequenceClassification` mặc định 2 nhãn → reshape sai (`[batch, group×2]` thay vì `[batch, group]`) → loss sai. Đã fix tại `tevatron/src/tevatron/reranker/driver/train.py`.

**Padding side:** Retriever dùng **left padding** (causal LLM + last-token pooling trên query/passage riêng lẻ). Reranker dùng **right padding** (sequence classification — HF tự tìm last non-padding token để classify).

**Corpus cache:** `train_reranker.sh` tự tìm và reuse corpus embedding `.pkl` đã được `eval.sh` tạo ra — không encode lại corpus (~12K items Beauty).

**Transformers 5.x compatibility fixes** áp dụng cho reranker:
- `transformers.deepspeed` → `transformers.integrations.deepspeed` (với fallback `False`)
- `compute_loss()` thêm `num_items_in_batch=None, **kwargs` để tương thích API mới
- `--overwrite_output_dir` bị xóa → bỏ flag này
- `--warmup_ratio` deprecated → dùng `--warmup_steps`

---

## 6. RecBole — Training & Evaluation

Dùng script `run_recbole.py` từ thư mục root của project:

```bash
source tevatron-env/bin/activate

# Train + eval (mặc định)
python run_recbole.py SASRec beauty
python run_recbole.py SASRec sports
python run_recbole.py SASRec ml-1m

# Override config bất kỳ từ CLI
python run_recbole.py SASRec beauty epochs=50 learning_rate=0.0005

# Chạy model khác trong RecBole (dùng default config của RecBole)
python run_recbole.py GRU4Rec beauty
python run_recbole.py BERT4Rec beauty

# Chạy model custom (đặt file trong recbole/, truyền tên module.ClassName)
python run_recbole.py cmamba4rec.CMamba4Rec beauty

# Eval only — load checkpoint đã lưu, không train lại
python run_recbole.py eval recbole/output/saved/SASRec-May-12-2026_09-28-11.pth
```

Output lưu tại:
- `recbole/output/saved/` — model checkpoints (`.pth`)
- `recbole/output/log/` — training logs
- `recbole/output/log_tensorboard/` — tensorboard logs

Script tự động:
- Thêm `recbole/` vào `sys.path` để tìm thư viện (`recbole/recbole/`) và custom models
- Load YAML config từ `recbole/props/<dataset>/` nếu tồn tại, fallback về config mặc định RecBole
- Set đúng `data_path` (tránh lỗi path khi chạy từ project root)
- Patch numpy 2.0 compatibility trước khi import RecBole

### Model custom

Đặt file `.py` trong thư mục `recbole/`, kế thừa class RecBole:

```python
# recbole/my_model.py
from recbole.model.sequential_recommender import SASRec

class MyModel(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # override hoặc thêm logic
```

```bash
python run_recbole.py my_model.MyModel beauty epochs=100
```

### Ghi chú kỹ thuật

> **`MAX_ITEM_LIST_LENGTH: 200`** trong tất cả yaml. SASRec chỉ thấy tối đa 200 item
> gần nhất, tránh OOM với self-attention O(n²) trên sequences dài (ML-1M max=2277).

> **`train_neg_sample_args: ~`** bắt buộc khi dùng `loss_type: CE`. CE loss tự xử lý
> negatives nội bộ (softmax trên toàn bộ items) — không được set negative sampling.

> **`TIME_FIELD: timestamp` + `load_col`** bắt buộc để RecBole sort đúng thứ tự thời
> gian. Thiếu hai dòng này sẽ gây `ValueError: [timestamp] is not exist in interaction`.

---

## 7. So sánh kết quả

Cả hai model đều dùng:
- Cùng bộ dữ liệu sau 5-core filter
- Cùng leave-one-out split (item cuối = test, áp chót = valid)
- Cùng tập items để rank
- Metrics: NDCG@5, HR@5, NDCG@10, HR@10, NDCG@20, HR@20, MRR@10

Điểm khác biệt duy nhất:

| | repLLaMA (Tevatron) | SASRec (RecBole) |
|---|---|---|
| Biểu diễn user | 3 item gần nhất (text) | Toàn bộ history (ID) |
| Biểu diễn item | Title text | Item ID |
| Kiến trúc | LLM encoder + LoRA | Self-Attention |
| Evaluation | FAISS dense retrieval | RecBole full ranking |

---

## 8. Kết quả thực nghiệm

Xem chi tiết tại **[experiments.md](experiments.md)** — bao gồm bảng kết quả tổng hợp và mô tả từng thực nghiệm.

Cập nhật bảng kết quả tự động:

```bash
python show_results.py                      # Chỉ in ra terminal
python show_results.py --update-experiments # In + ghi vào experiments.md
```

Thêm kết quả SASRec thủ công: tạo file `output/<dataset>/sasrec/eval_test.txt` với format `<metric>  all  <value>`.

---

## 9. Tình trạng hiện tại

### Đã hoàn thành
- [x] Pipeline tiền xử lý dữ liệu (`dataset/`)
- [x] Export Tevatron format (train/valid/test/corpus)
- [x] Export RecBole format (.inter, .item, .yaml)
- [x] Cài đặt và fix `tevatron-env` (NCCL + CUDA path, torch cu126)
- [x] Scripts gọn: `train.sh` và `eval.sh` (best/latest/base/checkpoint-N)
- [x] `train.sh`: named flags `--model`, `--group-size`, `--data-variant`, `--tag`; auto batch-size theo model size
- [x] `eval.sh`: named flags `--model`, `--split`, `--tag`, `--metric`; hỗ trợ chọn best checkpoint theo metric tùy chọn
- [x] `--append_eos_token` đồng nhất trên tất cả scripts (corpus + queries)
- [x] Tất cả 3 dataset đã sẵn sàng (Beauty, Sports phiên bản 2014; ML-1M)
- [x] `negative_passages` đã loại bỏ khỏi `valid.jsonl` và `test.jsonl` — tiết kiệm disk
- [x] `MAX_ITEM_LIST_LENGTH` cố định ở 200 trong tất cả yaml và `export_recbole.py`
- [x] Fix parser Amazon 2014: meta file dùng Python dict format (nháy đơn) — thêm fallback `ast.literal_eval` trong `preprocess.py`
- [x] `eval.sh` mode `base` — đánh giá base model không fine-tune (zero-shot)
- [x] `export_tevatron.py` (unified) — query augmentation (sliding window) + BM25/mixed negative sampling, tất cả trong 1 script
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

Chỉ cần sửa hàm `build_query()` trong `export_tevatron.py`:

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

## 10. Ghi chú kỹ thuật

### Thư mục gốc `output/` chứa model gì?

HuggingFace Trainer lưu model vào hai nơi khi kết thúc training:
- `checkpoint-{step}/` → checkpoint theo lịch `--save_steps`
- Thư mục gốc `output_dir/` → **bản copy của checkpoint cuối cùng**

Hai bộ trọng số này giống nhau. Thư mục gốc **không phải model tốt nhất** — đó là lý do cần dùng `eval.sh` với `checkpoint=best` để sweep và chọn đúng checkpoint.

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

### Tự động dọn DeepSpeed states sau mỗi checkpoint (`train.py`)

File `tevatron/src/tevatron/retriever/driver/train.py` đã được thêm `CheckpointCleanupCallback` — hook `on_save` chạy ngay sau khi DeepSpeed lưu mỗi checkpoint, xóa các file chỉ cần để resume training:

| File/thư mục bị xóa | Dung lượng | Lý do |
|---|---|---|
| `global_step*/` | ~2.3 GB | DeepSpeed model + optimizer states |
| `rng_state.pth` | ~15 KB | Trạng thái random number generator |
| `scheduler.pt` | ~1.4 KB | Trạng thái learning rate scheduler |
| `training_args.bin` | ~6.5 KB | Config training |
| `zero_to_fp32.py` | ~33 KB | Script chuyển đổi ZeRO weights |

Mỗi `checkpoint-N/` sau cleanup chỉ còn `adapter_config.json` + `adapter_model.safetensors` (~20 MB) — đủ để `eval.sh` chạy bình thường.

> **Lưu ý:** Sau cleanup không thể resume training từ checkpoint đó nữa. Đây là trade-off chấp nhận được vì workflow luôn là train từ đầu.
>
> Để revert về bản gốc Tevatron: `cd tevatron && git checkout src/tevatron/retriever/driver/train.py`
