# Hướng dẫn chạy song song trên nhiều máy

> Mục tiêu: lấp các gap thực nghiệm liệt kê ở `paper_report.md` §5 bằng cách phân phối sang nhiều máy có GPU,
> chạy song song, rồi gộp kết quả về máy chính. Không cần đồng bộ toàn bộ `output/` (61GB) — chỉ cần đồng bộ
> code + raw data + kết quả text nhỏ.

---

## 1. Những gì cần copy vs. những gì nên tự sinh lại

| Thành phần | Kích thước | Copy sang máy mới? | Ghi chú |
|---|---:|---|---|
| Source code (`.py`, `.sh`, `.md`, `recbole/props/`) | vài MB | **Có** — qua `git clone`/`rsync` | Repo có git, dùng `git clone` là sạch nhất |
| `dataset/raw/` (Amazon .json.gz + ML-1M .dat) | 407 MB | **Có** (nhanh hơn tải lại) hoặc tải lại theo README §2 | 4 file: `meta_Beauty.json.gz`, `meta_Sports_and_Outdoors.json.gz`, `reviews_Beauty_5.json.gz`, `reviews_Sports_and_Outdoors_5.json.gz`, + thư mục `ml-1m/` |
| `dataset/dataset/` (đã export Tevatron+RecBole) | 9.3 GB | **Không** — tự sinh lại bằng `run_all.py` (nhanh, vài phút, không cần GPU) | Chỉ export đúng variant máy đó cần (xem §3) |
| `tevatron-env/` (venv) | 12 GB | **Không** — tự cài lại theo README §1.1 | venv có path tuyệt đối, không portable giữa máy |
| `~/.cache/huggingface/` (model weights) | ~21 GB | Tùy — nếu máy mới có internet + quyền truy cập HF, để tự tải; nếu không, `rsync` cache dir sẽ nhanh hơn tải lại | Chỉ cần tải đúng model dùng (Qwen3-Embedding-0.6B bắt buộc; 4B chỉ cần cho thực nghiệm scale) |
| `output/` (checkpoints, embeddings, kết quả) | 61 GB | **Không** copy sang; **Có** copy kết quả TỪ máy mới VỀ (chỉ vài trăm KB — xem §4) | |
| `tevatron/` (thư viện Tevatron v2, đã patch) | 23 MB | **Có** | Đã patch `trainer.py` để tương thích transformers>=5.0 — copy nguyên, đừng `git clone` lại từ upstream |

---

## 2. Setup máy mới (từ đầu)

```bash
# 1. Lấy code
git clone <remote-url-của-repo> repLLaMA   # hoặc rsync nếu không có remote
cd repLLaMA

# 2. Copy raw data (từ máy hiện tại, qua rsync/scp)
rsync -avz --progress user@may-hien-tai:/media/administrator/Data1/Projects/Python/repLLaMA/dataset/raw/ ./dataset/raw/

# 3. Cài môi trường — làm đúng theo README.md §1.1, tóm tắt:
python3.11 -m venv tevatron-env
source tevatron-env/bin/activate
pip install torch==2.7.1+cu126 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126
pip install nvidia-nccl-cu11 deepspeed==0.18.9 transformers==5.7.0 peft==0.19.1 accelerate==1.13.0
pip install datasets faiss-cpu pyserini pytrec_eval sentencepiece qwen_omni_utils tqdm recbole==1.2.1
pip install flash-attn==2.8.3 --no-build-isolation   # cần GPU Compute Capability >= 8.0 (Ampere+)
pip install -e tevatron/

# Nếu GPU không hỗ trợ flash-attn (< Ampere): thêm --attn_implementation sdpa vào train.sh/eval.sh
# hoặc sửa default trong tevatron/src/tevatron/retriever/arguments.py (xem README §1.1)

# 4. Verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import recbole; print(recbole.__version__)"   # → 1.2.1

# 5. Preprocess + export data — CHỈ export đúng variant máy này cần (xem bảng phân công §3)
cd dataset
python run_all.py <dataset>              # 5-core filter, leave-one-out split, xuất bản standard
python export_tevatron.py <dataset> --augment                          # nếu cần augmented
python export_tevatron.py <dataset> --context_size 5 --augment         # nếu cần cs5-aug, v.v.
cd ..

# 6. Model weights — để train.sh/eval.sh tự tải qua HuggingFace (cần internet + có thể cần HF_TOKEN
# nếu model gated), hoặc rsync cache từ máy hiện tại:
rsync -avz --progress user@may-hien-tai:~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B ~/.cache/huggingface/hub/
# (đổi tên model nếu cần model khác, ví dụ models--Qwen--Qwen3-Embedding-4B cho thực nghiệm scale)
```

---

## 3. Phân công thực nghiệm theo máy (map với `paper_report.md` §5)

Chia theo dataset để mỗi máy chỉ cần export/preprocess 1 dataset, giảm trùng lặp:

> **Quyết định 2026-07-14**: dừng đầu tư vào v2 (instruction-based) format — so sánh trực tiếp cho thấy v2
> chỉ nhỉnh hơn v1 (title-only) ~1-2% NDCG@10 trong khi cần `query_max_len` dài hơn đáng kể (256-320 vs 128)
> → train/eval chậm hơn rõ rệt. Toàn bộ kế hoạch dưới đây dùng **v1** (`export_tevatron.py`, không phải
> `export_tevatron_v2.py`). Chi tiết so sánh: `paper_report.md` §1.4 và §8.

> **Cập nhật 2026-07-24**: Sports augmentation (`cs5-aug-gs32` + ablation `cs5-gs32-0.6b`) đã chạy xong —
> KHÔNG cần lặp lại. ML-1M 4B cũng đã có kết quả đầy đủ.
> **Cập nhật 2026-07-26**: SASRec đã train xong cả 3 dataset (Beauty/Sports/ML-1M) — KHÔNG cần lặp lại.
> GRU4Rec và SRGNN cũng đã xong cả 3 dataset. Baseline RecBole coi như hoàn tất. Còn lại: 4B Sports, 4B Beauty.

### Máy A — Sports: model scale 4B

```bash
source tevatron-env/bin/activate
cd dataset && python run_all.py sports && cd ..

# Model scale 4B cho Sports — LƯU Ý: lần chạy trước tag "cs5-gs50-4b" bị nhầm, train_config.json ghi
# model=0.6B (không phải 4B) và không có checkpoint nào được lưu. Đảm bảo dùng đúng --model:
nvidia-smi --query-gpu=memory.total --format=csv   # kiểm tra VRAM trước khi chạy
./train.sh sports --model Qwen/Qwen3-Embedding-4B --data-variant cs5-aug --tag cs5-aug-gs32-4b --group-size 32
./eval.sh sports --model Qwen/Qwen3-Embedding-4B --tag cs5-aug-gs32-4b --data-variant cs5-aug
# Sau khi train xong, xác nhận checkpoint thật sự tồn tại trước khi coi là thành công (xem checklist §5):
ls -la output/sports/qwen3-embedding-4b-cs5-aug-gs32-4b/checkpoint-*/adapter_model.safetensors
```

### Máy B — (tuỳ chọn) hoàn tất `cs5-gs50-0.6b` cho Sports

```bash
source tevatron-env/bin/activate
cd dataset && python run_all.py sports && cd ..

# Job cũ dừng ở checkpoint-2000 (không có model cuối) — chỉ cần chạy nếu muốn thêm điểm so sánh group_size=50
# (cs5-aug-gs32/cs5-gs32-0.6b ở gs=32 đã đủ cho ablation chính, mục này không bắt buộc)
./train.sh sports --data-variant cs5-aug --tag cs5-aug-gs50 --group-size 50
./eval.sh sports --tag cs5-aug-gs50
./train.sh sports --data-variant cs5 --tag cs5-gs50-retry --group-size 50   # KHÔNG augmentation, đối chứng
./eval.sh sports --tag cs5-gs50-retry
```

### Máy C — Model scale lớn hơn (4B), cần VRAM cao nếu có

```bash
source tevatron-env/bin/activate
cd dataset && python run_all.py beauty && cd ..

# (4) Retrain Beauty 4B — lần trước KHÔNG để lại checkpoint (train_config.json tồn tại nhưng thư mục
# checkpoint rỗng → nghi ngờ OOM hoặc bị kill giữa chừng). Theo dõi kỹ VRAM, giảm --group-size nếu OOM.
nvidia-smi --query-gpu=memory.total --format=csv   # kiểm tra VRAM trước khi chạy, cần gần 12GB free
./train.sh beauty --model Qwen/Qwen3-Embedding-4B --tag 4b
./eval.sh beauty --model Qwen/Qwen3-Embedding-4B --tag 4b

# Nếu máy có VRAM > 12GB, có thể thử group-size cao hơn để so sánh công bằng với best 0.6B (gs8):
./train.sh beauty --model Qwen/Qwen3-Embedding-4B --tag 4b-aug --data-variant aug-3 --group-size 8

# ML-1M 4B đã có kết quả (cs5-gs50-aug-4b) — KHÔNG cần chạy lại. Sports 4B đã chuyển sang Máy A ở §3.
```

> **Lưu ý VRAM 4B**: README ghi nhận batch=1, accum=32 vẫn "sát giới hạn 12GB" trên RTX 3060. Nếu máy mới
> cũng chỉ có 12GB, cân nhắc theo dõi `nvidia-smi` trong lúc train (`watch -n 5 nvidia-smi`) để bắt OOM sớm
> thay vì để job chạy nhiều giờ rồi mới phát hiện không có checkpoint (đúng như đã xảy ra ở máy hiện tại).

---

## 4. Đưa kết quả về máy chính

Sau khi mỗi máy chạy xong, chỉ cần đồng bộ phần **kết quả text** (nhỏ), không cần đồng bộ checkpoint hay
embedding `.pkl` (nặng, không cần thiết cho việc viết paper):

```bash
# Từ máy chính, kéo về các file cần thiết (ví dụ Máy A):
rsync -avz --include='*/' \
  --include='train_config.json' \
  --include='checkpoint_selection.log' \
  --include='eval_test_best.txt' --include='eval_test_latest.txt' --include='eval_test*.txt' \
  --exclude='*' \
  user@may-A:/path/to/repLLaMA/output/sports/ ./output/sports/

# Với RecBole (SASRec/GRU4Rec/SRGNN), chỉ cần log + checkpoint nhỏ (.pth thường vài-15MB, copy nguyên):
rsync -avz user@may-A:/path/to/repLLaMA/recbole/output/saved/SRGNN-*.pth ./recbole/output/saved/
rsync -avz user@may-A:/path/to/repLLaMA/recbole/output/log/ ./recbole/output/log/
```

Sau khi đồng bộ xong, tại máy chính:

```bash
source tevatron-env/bin/activate
python show_results.py --update-experiments     # refresh bảng auto trong experiments.md

# Nếu có history filter cần chạy lại cho model mới (không cần GPU, chỉ cần rank_clean.trec + qrels_clean.txt
# đã đồng bộ về):
python eval_filter.py sports --tag cs5-aug-gs32-4b --filter-mode full   # sau khi có 4B Sports

# run_recbole.py đã fix bug weights_only (2026-07-25) — train xong sẽ tự in "test result", không cần chạy
# eval-only riêng nữa. Chỉ dùng eval-only khi muốn re-check checkpoint cũ đã có sẵn:
python run_recbole.py eval recbole/output/saved/SRGNN-<timestamp>.pth
```

Cuối cùng, cập nhật thủ công các bảng trong `paper_report.md` với số liệu mới — các dòng "N/A — chưa chạy"
chính là những chỗ cần điền.

---

## 5. Checklist nhanh trước khi coi một máy là "xong"

- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True`
- [ ] `./eval.sh <dataset> base` chạy được không lỗi (xác nhận model + data pipeline hoạt động) trước khi
      chạy các job dài hàng giờ
- [ ] Kiểm tra `nvidia-smi` không có process nào khác tranh VRAM trước khi launch train.sh
- [ ] Sau khi train xong, xác nhận `output/<dataset>/<tag>/checkpoint-*/adapter_model.safetensors` **tồn
      tại và có kích thước > 0** trước khi coi là thành công (bài học từ 4B Beauty: train_config.json được
      ghi trước, nhưng checkpoint có thể không bao giờ được lưu nếu job chết giữa chừng)
