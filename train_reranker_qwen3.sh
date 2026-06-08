#!/bin/bash
#
# train_reranker_qwen3.sh — Fine-tune Qwen3-Reranker cho sequential recommendation.
#
# Cách sử dụng:
#   ./train_reranker_qwen3.sh <dataset> [--retriever-model MODEL] [--retriever-tag TAG]
#                                        [--reranker-model MODEL]
#                                        [--depth N] [--group-size N] [--epochs N] [--lr LR]
#
#   <dataset>              : beauty | sports | ml-1m
#   --retriever-model M    : retriever base model (mặc định: Qwen/Qwen3-Embedding-0.6B)
#   --retriever-tag TAG    : tag của retriever đã train (ví dụ: aug-5); để trống = standard
#   --reranker-model M     : Qwen3-Reranker model (mặc định: Qwen/Qwen3-Reranker-0.6B)
#   --depth N              : số candidates/query để mine hard negatives (mặc định: 100)
#   --group-size N         : 1 pos + (N-1) hard negs mỗi query khi train (mặc định: 8)
#   --epochs N             : số training epochs (mặc định: 3)
#   --lr LR                : learning rate (mặc định: 1e-4)
#
# Pipeline:
#   Bước 1-4: Giống train_reranker.sh — reuse training data nếu đã có
#     1. Reuse corpus embedding từ eval.sh (hoặc encode mới)
#     2. Encode train queries với fine-tuned retriever
#     3. FAISS search → train_rank.trec (hard negatives)
#     4. prepare_rerank_data.py → reranker_train.jsonl
#   Bước 5: Fine-tune Qwen3-Reranker với InfoNCE loss trên yes-logits
#
# Output:
#   LoRA weights: output/<dataset>/<retriever_tag>-<reranker_tag>/
#   Training data (shared với train_reranker.sh): output/<dataset>/<retriever_tag>-reranker/train_data/
#
# Ví dụ:
#   ./train_reranker_qwen3.sh beauty
#   ./train_reranker_qwen3.sh beauty --retriever-tag aug-5 --epochs 5

set -e
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"

# ── Parse args ────────────────────────────────────────────────────────────────

if [ -z "$1" ]; then
  echo "Lỗi: Cần nhập dataset!"
  echo "Dùng: ./train_reranker_qwen3.sh <dataset> [options]"
  exit 1
fi

dataset=$1; shift

retriever_model="Qwen/Qwen3-Embedding-0.6B"
retriever_tag=""
reranker_model="Qwen/Qwen3-Reranker-0.6B"
depth=100
group_size=8
epochs=3
lr="1e-4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --retriever-model) retriever_model="$2"; shift 2 ;;
    --retriever-tag)   retriever_tag="$2";   shift 2 ;;
    --reranker-model)  reranker_model="$2";  shift 2 ;;
    --depth)           depth="$2";           shift 2 ;;
    --group-size)      group_size="$2";      shift 2 ;;
    --epochs)          epochs="$2";          shift 2 ;;
    --lr)              lr="$2";              shift 2 ;;
    *) echo "Lỗi: Tham số không hợp lệ '$1'"; exit 1 ;;
  esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────

RETRIEVER_TAG=$(basename "${retriever_model}" | tr '[:upper:]' '[:lower:]')
[ -n "${retriever_tag}" ] && RETRIEVER_TAG="${RETRIEVER_TAG}-${retriever_tag}"

RERANKER_TAG=$(basename "${reranker_model}" | tr '[:upper:]' '[:lower:]')

RETRIEVER_DIR="./output/${dataset}/${RETRIEVER_TAG}"
DATA_DIR="./dataset/dataset/tevatron/${dataset}"

# Training data: dùng chung với train_reranker.sh (nếu đã chạy)
SHARED_TRAIN_DIR="${RETRIEVER_DIR}-reranker/train_data"

# Output LoRA weights cho Qwen3-Reranker
RERANKER_OUT="${RETRIEVER_DIR}-${RERANKER_TAG}"

# Các file trung gian trong SHARED_TRAIN_DIR
TRAIN_QUERY_EMB="${SHARED_TRAIN_DIR}/train_queries.pkl"
TRAIN_RANK_TXT="${SHARED_TRAIN_DIR}/train_rank.txt"
TRAIN_RANK_TREC="${SHARED_TRAIN_DIR}/train_rank.trec"
RERANKER_TRAIN_JSONL="${SHARED_TRAIN_DIR}/reranker_train.jsonl"

# Eval batch size theo retriever model size
case "${retriever_model}" in
  *4B*|*3B*|*1B*) eval_batch=16 ;;
  *)               eval_batch=64 ;;
esac

# ── Validation ────────────────────────────────────────────────────────────────

if [ ! -d "${RETRIEVER_DIR}" ]; then
  echo "Lỗi: Không tìm thấy retriever tại ${RETRIEVER_DIR}"
  echo "Hãy chạy ./train.sh ${dataset}$([ -n "${retriever_tag}" ] && echo " --tag ${retriever_tag}") trước."
  exit 1
fi
if [ ! -f "${RETRIEVER_DIR}/adapter_model.safetensors" ] && \
   [ ! -f "${RETRIEVER_DIR}/adapter_model.bin" ]; then
  echo "Lỗi: Không tìm thấy retriever adapter weights tại ${RETRIEVER_DIR}"
  exit 1
fi

mkdir -p "${SHARED_TRAIN_DIR}" "${RERANKER_OUT}"

echo "════════════════════════════════════════════════"
echo "  Train Qwen3-Reranker"
echo "════════════════════════════════════════════════"
echo "  Dataset          : ${dataset}"
echo "  Retriever        : ${RETRIEVER_DIR}"
echo "  Reranker model   : ${reranker_model}"
echo "  Reranker output  : ${RERANKER_OUT}"
echo "  Hard neg depth   : ${depth}"
echo "  Group size       : ${group_size}  (1 pos + $((group_size - 1)) hard negs)"
echo "  Epochs           : ${epochs}"
echo "  Learning rate    : ${lr}"
echo "════════════════════════════════════════════════"
echo ""

# ── Xác định best checkpoint của retriever ────────────────────────────────────

BEST_LABEL=""
BEST_TXT="${RETRIEVER_DIR}/embeddings/results/eval_test_best.txt"
if [ -f "${BEST_TXT}" ]; then
  BEST_LABEL=$(grep "^Best checkpoint" "${BEST_TXT}" | awk '{print $NF}')
fi

if [ -n "${BEST_LABEL}" ] && [ -d "${RETRIEVER_DIR}/${BEST_LABEL}" ]; then
  BEST_LORA_PATH="${RETRIEVER_DIR}/${BEST_LABEL}"
  echo "Best checkpoint : ${BEST_LABEL}"
else
  BEST_LORA_PATH="${RETRIEVER_DIR}"
  BEST_LABEL="final"
  echo "Best checkpoint : không tìm thấy → dùng final model"
fi
echo ""

# ── Step 1: Corpus embedding ─────────────────────────────────────────────────

BEST_CORPUS_PKL="${RETRIEVER_DIR}/embeddings/corpus/${BEST_LABEL}.pkl"
if [ -f "${BEST_CORPUS_PKL}" ]; then
  CORPUS_EMB="${BEST_CORPUS_PKL}"
  echo "[1/5] Reuse corpus embedding: $(basename "${CORPUS_EMB}")"
else
  CORPUS_EMB="${BEST_CORPUS_PKL}"
  mkdir -p "$(dirname "${CORPUS_EMB}")"
  echo "[1/5] Encoding corpus..."
  CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
    --output_dir temp \
    --model_name_or_path "${retriever_model}" \
    --lora --lora_name_or_path "${BEST_LORA_PATH}" \
    --passage_prefix "Passage: " \
    --bf16 --pooling last --padding_side left \
    --append_eos_token --normalize \
    --per_device_eval_batch_size ${eval_batch} \
    --passage_max_len 196 \
    --dataset_name json \
    --dataset_path "${DATA_DIR}/corpus.jsonl" \
    --encode_output_path "${CORPUS_EMB}"
fi

# ── Step 2: Encode train queries ──────────────────────────────────────────────

if [ ! -f "${TRAIN_QUERY_EMB}" ]; then
  echo "[2/5] Encoding train queries..."
  CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
    --output_dir temp \
    --model_name_or_path "${retriever_model}" \
    --lora --lora_name_or_path "${BEST_LORA_PATH}" \
    --query_prefix "Query: " \
    --bf16 --pooling last --padding_side left \
    --append_eos_token --normalize \
    --per_device_eval_batch_size ${eval_batch} \
    --query_max_len 128 \
    --dataset_name json \
    --dataset_path "${DATA_DIR}/train.jsonl" \
    --encode_is_query \
    --encode_output_path "${TRAIN_QUERY_EMB}"
else
  echo "[2/5] Train query embeddings đã có — bỏ qua."
fi

# ── Step 3: FAISS search → train_rank.trec ───────────────────────────────────

if [ ! -f "${TRAIN_RANK_TREC}" ]; then
  echo "[3/5] FAISS search (depth=${depth})..."
  python -m tevatron.retriever.driver.search \
    --query_reps "${TRAIN_QUERY_EMB}" \
    --passage_reps "${CORPUS_EMB}" \
    --depth ${depth} --batch_size 128 \
    --save_text --save_ranking_to "${TRAIN_RANK_TXT}"

  python -m tevatron.utils.format.convert_result_to_trec \
    --input "${TRAIN_RANK_TXT}" --output "${TRAIN_RANK_TREC}" --remove_query

  # Deduplicate
  TREC_FILE="${TRAIN_RANK_TREC}" python3 -c "
import os
f = os.environ['TREC_FILE']
seen = set(); lines = []
with open(f) as fi:
    for line in fi:
        parts = line.strip().split()
        if len(parts) < 3: continue
        k = parts[0] + '_' + parts[2]
        if k not in seen:
            seen.add(k); lines.append(line)
with open(f, 'w') as fo:
    fo.writelines(lines)
print(f'  {len(lines):,} entries sau dedup')
"
else
  echo "[3/5] Train rank trec đã có — bỏ qua."
fi

# ── Step 4: Prepare training data ─────────────────────────────────────────────

if [ ! -f "${RERANKER_TRAIN_JSONL}" ]; then
  echo "[4/5] Chuẩn bị training data..."
  python prepare_rerank_data.py \
    --mode  train \
    --queries "${DATA_DIR}/train.jsonl" \
    --corpus  "${DATA_DIR}/corpus.jsonl" \
    --trec    "${TRAIN_RANK_TREC}" \
    --output  "${RERANKER_TRAIN_JSONL}" \
    --depth   ${depth}
else
  echo "[4/5] Training data đã có — bỏ qua."
fi

# ── Step 5: Fine-tune Qwen3-Reranker ─────────────────────────────────────────

echo ""
echo "[5/5] Fine-tuning Qwen3-Reranker..."
echo ""

CUDA_VISIBLE_DEVICES=0 python train_reranker_qwen3.py \
  --train_data   "${RERANKER_TRAIN_JSONL}" \
  --output_dir   "${RERANKER_OUT}" \
  --model        "${reranker_model}" \
  --group_size   ${group_size} \
  --max_length   256 \
  --epochs       ${epochs} \
  --lr           ${lr} \
  --per_batch    1 \
  --grad_accum   32 \
  --warmup_steps 100 \
  --save_steps   500 \
  --lora_r       16 \
  --lora_alpha   64

echo ""
echo "✓ Fine-tuning xong!"
echo "✓ LoRA weights lưu tại: ${RERANKER_OUT}"
echo ""
echo "Bước tiếp theo — rerank + evaluate:"
echo "  ./rerank_qwen3.sh ${dataset} --retriever-tag ${retriever_tag} --lora-path ${RERANKER_OUT}"
