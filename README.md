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
│   ├── dataset/
│   │   ├── tevatron/           # Output cho repLLaMA
│   │   │   └── <dataset>/
│   │   │       ├── corpus.jsonl
│   │   │       ├── train.jsonl
│   │   │       ├── valid.jsonl
│   │   │       └── test.jsonl
│   │   └── recbole/            # Output cho SASRec
│   │       └── <dataset>/
│   │           ├── <dataset>.inter
│   │           ├── <dataset>.item
│   │           └── sasrec_<dataset>.yaml
│   └── README.md               # Mô tả chi tiết pipeline tiền xử lý
│
├── tevatron/                   # Thư viện Tevatron v2 (clone từ GitHub)
├── tevatron-env/               # Virtual environment cho repLLaMA
│
├── train.sh                    # Fine-tune repLLaMA
├── encode_corpus.sh            # Encode toàn bộ corpus thành dense vectors
├── encode_queries.sh           # Encode queries từ một split
├── search.sh                   # FAISS search: query vs corpus
├── evaluate.sh                 # Tính metrics (Recall, NDCG, MRR, MAP)
├── select_best.sh              # Sweep checkpoints trên valid → evaluate test
├── eval_one.sh                 # Evaluate một checkpoint bất kỳ
│
├── ds_config.json              # DeepSpeed ZeRO-2 config
└── output/                     # Kết quả training và embedding (tự sinh)
    └── <dataset>/
        └── <model_tag>/
            ├── checkpoint-*/   # LoRA checkpoints (lưu mỗi save_steps)
            ├── adapter_model.safetensors  # Final model (= checkpoint cuối)
            └── embeddings/
                ├── corpus/     # Dense vectors của corpus
                ├── queries/    # Dense vectors của queries
                └── results/    # Kết quả search và evaluation
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

pip install torch==2.7.1+cu118 torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cu118

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

### 1.2 Môi trường SASRec (RecBole)

Dùng conda env `serec` đã có sẵn trên máy:

```bash
conda activate serec
python -c "import recbole; print(recbole.__version__)"
# → 1.2.0
```

Nếu cần tạo lại:

```bash
conda create -n serec python=3.9
conda activate serec
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install recbole==1.2.0
```

> **Lưu ý:** Nếu `torch.cuda.is_available()` trả về `False` trong conda env,
> chạy lại: `pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118`

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

### 4.1 Tham số chung của các scripts

| Tham số | Scripts | Giá trị | Mặc định |
|---|---|---|---|
| `dataset` | tất cả | `beauty` \| `sports` \| `ml-1m` \| `steam` | bắt buộc |
| `model` | tất cả | HuggingFace model ID | `Qwen/Qwen3-Embedding-0.6B` |
| `train_group_size` | `train.sh` | số nguyên ≥ 2 | `8` |
| `split` | `encode_queries`, `search`, `evaluate`, `eval_one` | `train` \| `valid` \| `test` | `test` |
| `checkpoint` | `eval_one.sh` | `checkpoint-N` hoặc `final` | bắt buộc |
| `metric` | `select_best.sh` | `ndcg_cut_10` \| `recall_10` \| ... | `ndcg_cut_10` |

### 4.2 Workflow khuyến nghị

```bash
# Bước 1: Fine-tune
./train.sh <dataset> [model] [train_group_size]

# Bước 2: Chọn checkpoint tốt nhất trên valid → evaluate test tự động
./select_best.sh <dataset> [model] [metric]
```

### 4.3 Evaluate thủ công

```bash
# Evaluate một checkpoint cụ thể (nhanh, tiện khi debug)
./eval_one.sh <dataset> <checkpoint> [model] [split]

# Ví dụ:
./eval_one.sh beauty checkpoint-600
./eval_one.sh beauty checkpoint-600 Qwen/Qwen3-Embedding-0.6B valid
./eval_one.sh beauty final

# Pipeline thủ công từng bước:
./encode_corpus.sh  <dataset> [model]
./encode_queries.sh <dataset> [model] [split]
./search.sh         <dataset> [model] [split]
./evaluate.sh       <dataset> [model] [split]
```

Khi nhập checkpoint không đúng, `eval_one.sh` tự liệt kê các checkpoint hiện có:
```
Lỗi: Không tìm thấy checkpoint tại ./output/beauty/qwen3-embedding-0.6b/checkpoint-999

Các checkpoint hiện có:
  checkpoint-200
  checkpoint-400
  ...
  final
```

### 4.4 Ví dụ chạy đầy đủ cho Beauty

```bash
source tevatron-env/bin/activate

# Train với 31 negatives/query thay vì 7 mặc định
./train.sh beauty Qwen/Qwen3-Embedding-0.6B 32

# Tìm best checkpoint → kết quả test
./select_best.sh beauty
```

Output:
```
output/beauty/qwen3-embedding-0.6b/
├── checkpoint-200/ ... checkpoint-2097/   # LoRA checkpoints
└── embeddings/results/
    ├── checkpoint_selection.log           # Bảng so sánh valid metric
    ├── eval_test_best.txt                 # Kết quả test của best checkpoint
    └── eval_test_checkpoint-600.txt       # Kết quả từ eval_one.sh (nếu có)
```

### 4.5 Cấu hình training (`train.sh`)

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

### 4.6 Về `--append_eos_token` và `</s>` trong data

Query format trong jsonl: `"Query: item1, item2, item3 </s>"`

Chuỗi `</s>` là legacy từ LLaMA-2 (nơi đây là EOS token thực). Với **Qwen3**, EOS token là
`<|im_end|>` — chuỗi `</s>` trong text chỉ là 3 token thường, không phải EOS.

Flag `--append_eos_token` trong các scripts mới là thứ thực sự thêm `<|im_end|>` vào cuối
sequence sau khi tokenize, đảm bảo `last` pooling lấy đúng token đại diện cho toàn bộ sequence.
**Không được bỏ flag này** — nó áp dụng cho cả query lẫn corpus encoding để nhất quán với training.

---

## 5. SASRec — Training & Evaluation (RecBole)

```bash
conda activate serec
cd /path/to/repLLaMA

python -c "
from recbole.quick_start import run_recbole
run_recbole(
    model='SASRec',
    config_file_list=['dataset/dataset/recbole/<dataset>/sasrec_<dataset>.yaml']
)
"
```

Thay `<dataset>` bằng `beauty`, `sports`, hoặc `ml-1m`.

> **`MAX_ITEM_LIST_LENGTH` đã được cố định ở 200** trong tất cả yaml files và trong
> `export_recbole.py`. SASRec chỉ nhìn thấy tối đa 200 item gần nhất của mỗi user,
> tránh OOM với self-attention O(n²) trên sequences dài (ML-1M max=2277).

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
- [x] Cài đặt và fix `tevatron-env` (NCCL + CUDA path)
- [x] Scripts đầy đủ: `train.sh`, `encode_corpus.sh`, `encode_queries.sh`, `search.sh`, `evaluate.sh`, `select_best.sh`, `eval_one.sh`
- [x] Tất cả scripts hỗ trợ tham số `dataset`, `model`; encode/evaluate hỗ trợ `split`
- [x] `train.sh` hỗ trợ tham số `train_group_size` để điều chỉnh số negatives
- [x] `--append_eos_token` đồng nhất trên tất cả scripts (corpus + queries)
- [x] Tất cả 3 dataset đã sẵn sàng (Beauty, Sports phiên bản 2014; ML-1M)
- [x] `negative_passages` đã loại bỏ khỏi `valid.jsonl` và `test.jsonl` — tiết kiệm disk
- [x] `MAX_ITEM_LIST_LENGTH` cố định ở 200 trong tất cả yaml và `export_recbole.py`
- [x] Fix parser Amazon 2014: meta file dùng Python dict format (nháy đơn), không phải JSON — thêm fallback `ast.literal_eval` trong `preprocess.py`; data đã được tái tạo với title đúng

### Cần làm
- [ ] Fix CUDA trong conda env `serec` để chạy SASRec trên GPU
- [ ] Chạy thực nghiệm đầy đủ và ghi lại kết quả so sánh

### Cải tiến tiềm năng
- [ ] Augment training data bằng cách dùng tất cả positions trong sequence (không chỉ N-2) để tạo nhiều training examples hơn và so sánh công bằng hơn với SASRec
- [ ] Hard negative mining (BM25 hoặc từ top retrieved items) để cải thiện chất lượng training

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
