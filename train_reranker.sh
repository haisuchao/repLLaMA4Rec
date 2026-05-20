#!/bin/bash
#
# train_reranker.sh — Train cross-encoder reranker với hard negatives từ retriever.
#
# Cách sử dụng:
#   ./train_reranker.sh <dataset> [--model MODEL] [--tag RTAG]
#                                  [--depth N] [--group-size N] [--epochs N]
#
#   <dataset>      : beauty | sports | ml-1m
#   --model MODEL  : base model (mặc định: Qwen/Qwen3-Embedding-0.6B)
#   --tag RTAG     : tag của retriever đã train (để trống = model chuẩn)
#   --depth N      : số candidates/query để mine hard negatives (mặc định: 100)
#   --group-size N : 1 pos + (N-1) hard neg mỗi query khi train (mặc định: 8)
#   --epochs N     : số training epochs (mặc định: 3)
#
# Pipeline:
#   1. Reuse corpus embedding từ eval.sh (hoặc encode nếu chưa có)
#   2. Encode train queries với fine-tuned retriever
#   3. FAISS search → train_rank.trec (hard negatives)
#   4. prepare_rerank_data.py → reranker_train.jsonl
#   5. tevatron.reranker.driver.train → reranker LoRA weights
#
# Ví dụ:
#   ./train_reranker.sh beauty                    # standard model, depth=100, group=8
#   ./train_reranker.sh beauty --tag aug-5        # retriever là aug-5 model
#   ./train_reranker.sh beauty --group-size 16    # nhiều hard negatives hơn
#   ./train_reranker.sh sports --depth 50         # depth nhỏ hơn
#
# Output: output/<dataset>/<model_tag>[-tag]-reranker/

set -e
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"

# ── Parse args ────────────────────────────────────────────────────────────────

if [ -z "$1" ]; then
  echo "Lỗi: Cần nhập dataset!"
  echo "Dùng: ./train_reranker.sh <dataset> [--model MODEL] [--tag RTAG] [--depth N] [--group-size N] [--epochs N]"
  exit 1
fi

dataset=$1; shift

model="Qwen/Qwen3-Embedding-0.6B"
retriever_tag=""
depth=100
group_size=8
epochs=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)      model="$2";         shift 2 ;;
    --tag)        retriever_tag="$2"; shift 2 ;;
    --depth)      depth="$2";         shift 2 ;;
    --group-size) group_size="$2";    shift 2 ;;
    --epochs)     epochs="$2";        shift 2 ;;
    *) echo "Lỗi: Tham số không hợp lệ '$1'"; exit 1 ;;
  esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_TAG=$(basename "${model}" | tr '[:upper:]' '[:lower:]')
[ -n "${retriever_tag}" ] && MODEL_TAG="${MODEL_TAG}-${retriever_tag}"

RETRIEVER_DIR="./output/${dataset}/${MODEL_TAG}"
DATA_DIR="./dataset/dataset/tevatron/${dataset}"
RERANKER_DIR="${RETRIEVER_DIR}-reranker"
DATA_OUT="${RERANKER_DIR}/train_data"

TRAIN_QUERY_EMB="${DATA_OUT}/train_queries.pkl"
TRAIN_RANK_TXT="${DATA_OUT}/train_rank.txt"
TRAIN_RANK_TREC="${DATA_OUT}/train_rank.trec"
RERANKER_TRAIN_JSONL="${DATA_OUT}/reranker_train.jsonl"

# Eval batch size theo model size
case "${model}" in
  *4B*|*3B*|*1B*) eval_batch=16 ;;
  *)              eval_batch=64 ;;
esac

# Per-device batch cho training: cross-encoder, seq_len=256
# Effective batch = per_batch * grad_accum = 32 queries/step
case "${model}" in
  *4B*|*3B*|*1B*) per_batch=1; grad_accum=32 ;;
  *)              per_batch=2; grad_accum=16 ;;
esac

# ── Validation ────────────────────────────────────────────────────────────────

if [ ! -d "${RETRIEVER_DIR}" ]; then
  echo "Lỗi: Không tìm thấy retriever tại ${RETRIEVER_DIR}"
  echo "Hãy chạy ./train.sh ${dataset}$([ -n "${retriever_tag}" ] && echo " --tag ${retriever_tag}") trước."
  exit 1
fi
if [ ! -f "${RETRIEVER_DIR}/adapter_model.safetensors" ] && \
   [ ! -f "${RETRIEVER_DIR}/adapter_model.bin" ]; then
  echo "Lỗi: Không tìm thấy adapter weights tại ${RETRIEVER_DIR}"
  exit 1
fi

mkdir -p "${DATA_OUT}"

echo "════════════════════════════════════════════════"
echo "  Train Reranker"
echo "════════════════════════════════════════════════"
echo "  Dataset      : ${dataset}"
echo "  Base model   : ${model}"
echo "  Retriever    : ${RETRIEVER_DIR}"
echo "  Reranker out : ${RERANKER_DIR}"
echo "  Depth        : ${depth} candidates/query"
echo "  Group size   : ${group_size} (1 pos + $((group_size - 1)) hard negs)"
echo "  Epochs       : ${epochs}"
echo "  Batch        : ${per_batch} queries × ${group_size} passages × accum ${grad_accum}"
echo "════════════════════════════════════════════════"
echo ""

# ── Step 1: Corpus embedding — reuse từ eval.sh nếu có ───────────────────────

CORPUS_EMB=$(ls "${RETRIEVER_DIR}/embeddings/corpus/"*.pkl 2>/dev/null | head -1)

if [ -z "${CORPUS_EMB}" ]; then
  CORPUS_EMB="${RETRIEVER_DIR}/embeddings/corpus/corpus.pkl"
  mkdir -p "$(dirname "${CORPUS_EMB}")"
  echo "[1/5] Encoding corpus..."
  CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
    --output_dir temp \
    --model_name_or_path "${model}" \
    --lora --lora_name_or_path "${RETRIEVER_DIR}" \
    --passage_prefix "Passage: " \
    --bf16 --pooling last --padding_side left \
    --append_eos_token --normalize \
    --per_device_eval_batch_size ${eval_batch} \
    --passage_max_len 196 \
    --dataset_name json \
    --dataset_path "${DATA_DIR}/corpus.jsonl" \
    --encode_output_path "${CORPUS_EMB}"
else
  echo "[1/5] Reuse corpus embedding: $(basename "${CORPUS_EMB}")"
fi

# ── Step 2: Encode train queries ──────────────────────────────────────────────

if [ ! -f "${TRAIN_QUERY_EMB}" ]; then
  echo "[2/5] Encoding train queries..."
  CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
    --output_dir temp \
    --model_name_or_path "${model}" \
    --lora --lora_name_or_path "${RETRIEVER_DIR}" \
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
  echo "[3/5] Searching (depth=${depth})..."
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

# ── Step 4: Prepare reranker training data ────────────────────────────────────

if [ ! -f "${RERANKER_TRAIN_JSONL}" ]; then
  echo "[4/5] Chuẩn bị reranker training data..."
  python prepare_rerank_data.py \
    --mode  train \
    --queries "${DATA_DIR}/train.jsonl" \
    --corpus  "${DATA_DIR}/corpus.jsonl" \
    --trec    "${TRAIN_RANK_TREC}" \
    --output  "${RERANKER_TRAIN_JSONL}" \
    --depth   ${depth}
else
  echo "[4/5] Reranker training data đã có — bỏ qua."
fi

# ── Step 5: Train reranker ────────────────────────────────────────────────────

echo ""
echo "[5/5] Training reranker..."
echo ""

CUDA_VISIBLE_DEVICES=0 python -m tevatron.reranker.driver.train \
  --model_name_or_path "${model}" \
  --lora --lora_r 16 --lora_alpha 64 --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj" \
  --dataset_name json \
  --dataset_path "${RERANKER_TRAIN_JSONL}" \
  --dataset_split train \
  --train_group_size ${group_size} \
  --rerank_max_len 256 \
  --append_eos_token \
  --output_dir "${RERANKER_DIR}" \
  --bf16 \
  --num_train_epochs ${epochs} \
  --per_device_train_batch_size ${per_batch} \
  --gradient_accumulation_steps ${grad_accum} \
  --learning_rate 1e-4 \
  --warmup_steps 100 \
  --save_steps 500 \
  --logging_steps 50 \
  --gradient_checkpointing

echo ""
echo "✓ Reranker training xong!"
echo "✓ Weights lưu tại: ${RERANKER_DIR}"
echo ""
echo "Bước tiếp theo:"
if [ -n "${retriever_tag}" ]; then
  echo "  ./rerank.sh ${dataset} --tag ${retriever_tag}"
else
  echo "  ./rerank.sh ${dataset}"
fi
