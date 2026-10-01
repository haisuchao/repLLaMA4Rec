#!/bin/bash
#
# train_reranker.sh — Train cross-encoder reranker (rankLLaMA) với hard negatives từ retriever.
#
# Cách sử dụng:
#   ./train_reranker.sh <dataset> [--model MODEL] [--tag RTAG] [--query-variant V]
#                                  [--depth N] [--group-size N] [--epochs N] [--lr LR]
#                                  [--keep-out-of-pool] [--prep-only]
#
#   <dataset>          : beauty | sports | toys | ml-1m
#   --model MODEL      : base model của retriever (mặc định: Qwen/Qwen3-Embedding-0.6B)
#   --reranker-model M : backbone reranker (mặc định: = --model). Khác --model → output có hậu tố
#                        -<tên model>, dữ liệu train_data/ dùng chung với reranker cùng retriever
#   --tag RTAG         : tag của retriever đã train (để trống = model chuẩn)
#   --query-variant V  : nguồn query train reranker (mặc định: auto)
#                          auto → query augmented cùng format với retriever (vd. beauty-aug, sports-cs5-aug)
#                          none → cùng bộ dữ liệu retriever (1 query/user, không augmented)
#                          <V>  → dataset/dataset/tevatron/<dataset>-<V>/train.jsonl
#   --depth N          : số candidates/query làm pool hard negatives (mặc định: 100)
#   --group-size N     : 1 pos + (N-1) hard neg mỗi query khi train (mặc định: 8)
#   --epochs N         : số training epochs (mặc định: 3)
#   --lr LR            : learning rate (mặc định: 1e-4)
#   --keep-out-of-pool : giữ cả query có positive ngoài top-depth (hành vi cũ — KHÔNG khuyến nghị)
#   --prep-only        : chỉ chạy bước 1-4 (chuẩn bị dữ liệu), không train
#
# Pipeline (thiết kế + bằng chứng: diag/README.md):
#   1. Reuse corpus embedding từ eval.sh (hoặc encode nếu chưa có)
#   2. Encode train queries (augmented) với retriever
#   3. FAISS search → top-depth
#   4. prepare_rerank_data.py → chỉ giữ query có positive trong top-depth;
#      negatives = phần còn lại của top-depth (Tevatron lấy đều group-1 negatives mỗi epoch)
#   5. tevatron.reranker.driver.train → reranker LoRA weights + reranker_config.json (rerank.sh đọc)
#
# Ví dụ:
#   ./train_reranker.sh beauty --tag aug-5              # query augmented beauty-aug (mặc định)
#   ./train_reranker.sh sports --tag cs5-aug-gs32       # query augmented sports-cs5-aug
#   ./train_reranker.sh beauty --tag aug-5 --query-variant none   # không augmented
#   ./train_reranker.sh beauty --tag aug-5 --reranker-model Qwen/Qwen3-Embedding-4B
#
# Output: output/<dataset>/<model_tag>[-tag]-reranker[-<query_variant>][-<reranker_model>]/

set -e
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"

# ── Parse args ────────────────────────────────────────────────────────────────

if [ -z "$1" ]; then
  echo "Lỗi: Cần nhập dataset!"
  echo "Dùng: ./train_reranker.sh <dataset> [--model MODEL] [--tag RTAG] [--query-variant V] [--depth N] [--group-size N] [--epochs N] [--lr LR] [--keep-out-of-pool] [--prep-only]"
  exit 1
fi

dataset=$1; shift

model="Qwen/Qwen3-Embedding-0.6B"
reranker_model=""
retriever_tag=""
query_variant="auto"
depth=100
group_size=8
epochs=3
lr="1e-4"
keep_out_of_pool=false
prep_only=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)            model="$2";         shift 2 ;;
    --reranker-model)   reranker_model="$2"; shift 2 ;;
    --tag)              retriever_tag="$2"; shift 2 ;;
    --query-variant)    query_variant="$2"; shift 2 ;;
    --depth)            depth="$2";         shift 2 ;;
    --group-size)       group_size="$2";    shift 2 ;;
    --epochs)           epochs="$2";        shift 2 ;;
    --lr)               lr="$2";            shift 2 ;;
    --keep-out-of-pool) keep_out_of_pool=true; shift ;;
    --prep-only)        prep_only=true;     shift ;;
    *) echo "Lỗi: Tham số không hợp lệ '$1'"; exit 1 ;;
  esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_TAG=$(basename "${model}" | tr '[:upper:]' '[:lower:]')
[ -n "${retriever_tag}" ] && MODEL_TAG="${MODEL_TAG}-${retriever_tag}"

RETRIEVER_DIR="./output/${dataset}/${MODEL_TAG}"

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

# EVAL_VARIANT / QUERY_VARIANT / QUERY_MAX_LEN — cùng quy tắc với rerank.sh
eval "$(python rerank_pipeline.py resolve --dataset "${dataset}" --retriever-dir "${RETRIEVER_DIR}" \
          --query-variant "${query_variant}")"

DATA_DIR="./dataset/dataset/tevatron/${dataset}"                        # corpus (giống mọi variant)
TRAIN_QUERY_DIR="${DATA_DIR}${QUERY_VARIANT:+-${QUERY_VARIANT}}"
# train_data/ chỉ phụ thuộc retriever → dùng chung giữa các backbone reranker
BASE_RERANKER_DIR="${RETRIEVER_DIR}-reranker${QUERY_VARIANT:+-${QUERY_VARIANT}}"
reranker_model="${reranker_model:-${model}}"
RERANKER_DIR="${BASE_RERANKER_DIR}"
[ "${reranker_model}" != "${model}" ] && RERANKER_DIR="${BASE_RERANKER_DIR}-$(basename "${reranker_model}" | tr '[:upper:]' '[:lower:]')"
DATA_OUT="${BASE_RERANKER_DIR}/train_data"

TRAIN_QUERY_EMB="${DATA_OUT}/train_queries.pkl"
TRAIN_RANK_TXT="${DATA_OUT}/train_rank.txt"
TRAIN_RANK_TREC="${DATA_OUT}/train_rank.trec"
if [ "${keep_out_of_pool}" = "true" ]; then
  RERANKER_TRAIN_JSONL="${DATA_OUT}/reranker_train_all.jsonl"
  pool_flag="--keep-out-of-pool"
else
  RERANKER_TRAIN_JSONL="${DATA_OUT}/reranker_train_inpool.jsonl"
  pool_flag=""
fi

# Eval batch size theo model size
case "${model}" in
  *4B*|*3B*|*1B*) eval_batch=16 ;;
  *)              eval_batch=64 ;;
esac

# Per-device batch cho training: cross-encoder, seq_len=256, effective batch = 32 queries/step.
# GPU ≥ 20GB: batch 4, không gradient checkpointing (nhanh hơn ~33%, cùng kết quả toán học).
gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
case "${reranker_model}" in
  *4B*|*3B*|*1B*) per_batch=1; grad_accum=32; grad_ckpt=true ;;
  *)
    if [ "${gpu_mem:-0}" -ge 20000 ]; then
      per_batch=4; grad_accum=8;  grad_ckpt=false
    else
      per_batch=2; grad_accum=16; grad_ckpt=true
    fi
    ;;
esac

mkdir -p "${DATA_OUT}" "${RERANKER_DIR}"

echo "════════════════════════════════════════════════"
echo "  Train Reranker"
echo "════════════════════════════════════════════════"
echo "  Dataset      : ${dataset}"
echo "  Retriever model: ${model}"
echo "  Reranker model : ${reranker_model}"
echo "  Retriever    : ${RETRIEVER_DIR} (data variant: ${EVAL_VARIANT:-chuẩn})"
echo "  Train queries: ${TRAIN_QUERY_DIR}/train.jsonl"
echo "  Reranker out : ${RERANKER_DIR}"
echo "  Depth        : ${depth} candidates/query $([ "${keep_out_of_pool}" = "true" ] && echo '(giữ cả query có positive ngoài pool)' || echo '(chỉ query có positive trong pool)')"
echo "  Group size   : ${group_size} (1 pos + $((group_size - 1)) hard negs)"
echo "  Epochs       : ${epochs}"
echo "  Learning rate: ${lr}"
echo "  Batch        : ${per_batch} queries × ${group_size} passages × accum ${grad_accum}$([ "${grad_ckpt}" = "true" ] && echo ' (gradient checkpointing)')"
echo "════════════════════════════════════════════════"
echo ""

[ -f "${RERANKER_DIR}/INVALID_pad_bug.txt" ] && \
  echo "⚠️  ${RERANKER_DIR} chứa reranker cũ train khi còn pad bug — adapter sẽ bị ghi đè khi train xong." && echo ""

# ── Xác định best checkpoint để dùng nhất quán cho cả corpus lẫn query ───────
#
# Reranker cần mine hard negatives từ CÙNG model sẽ dùng lúc inference
# (best checkpoint, không phải last checkpoint hay checkpoint ngẫu nhiên).
# Đọc từ eval_test_best.txt do eval.sh tạo ra.

BEST_LABEL=""
BEST_TXT="${RETRIEVER_DIR}/embeddings/results/eval_test_best.txt"
if [ -f "${BEST_TXT}" ]; then
  BEST_LABEL=$(grep "^Best checkpoint" "${BEST_TXT}" | awk '{print $NF}')
fi

if [ -n "${BEST_LABEL}" ] && [ -d "${RETRIEVER_DIR}/${BEST_LABEL}" ]; then
  BEST_LORA_PATH="${RETRIEVER_DIR}/${BEST_LABEL}"
  echo "Best checkpoint   : ${BEST_LABEL} (lora: ${BEST_LORA_PATH})"
else
  BEST_LORA_PATH="${RETRIEVER_DIR}"
  echo "Best checkpoint   : không tìm thấy → dùng final model (${RETRIEVER_DIR})"
  BEST_LABEL="final"
fi
echo ""

# ── Step 1: Corpus embedding — reuse từ eval.sh nếu có ───────────────────────

CORPUS_EMB="${RETRIEVER_DIR}/embeddings/corpus/${BEST_LABEL}.pkl"
if [ -f "${CORPUS_EMB}" ]; then
  echo "[1/5] Reuse corpus embedding (${BEST_LABEL}): $(basename "${CORPUS_EMB}")"
else
  mkdir -p "$(dirname "${CORPUS_EMB}")"
  echo "[1/5] Encoding corpus với ${BEST_LABEL}..."
  CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
    --output_dir temp \
    --model_name_or_path "${model}" \
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
  echo "[2/5] Encoding train queries ($(wc -l < "${TRAIN_QUERY_DIR}/train.jsonl") queries) với ${BEST_LABEL}..."
  CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
    --output_dir temp \
    --model_name_or_path "${model}" \
    --lora --lora_name_or_path "${BEST_LORA_PATH}" \
    --query_prefix "Query: " \
    --bf16 --pooling last --padding_side left \
    --append_eos_token --normalize \
    --per_device_eval_batch_size ${eval_batch} \
    --query_max_len ${QUERY_MAX_LEN} \
    --dataset_name json \
    --dataset_path "${TRAIN_QUERY_DIR}/train.jsonl" \
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
    --depth ${depth} --batch_size 256 \
    --save_text --save_ranking_to "${TRAIN_RANK_TXT}"

  python -m tevatron.utils.format.convert_result_to_trec \
    --input "${TRAIN_RANK_TXT}" --output "${TRAIN_RANK_TREC}" --remove_query
else
  echo "[3/5] Train rank trec đã có — bỏ qua."
fi

# ── Step 4: Prepare reranker training data ────────────────────────────────────

if [ ! -f "${RERANKER_TRAIN_JSONL}" ]; then
  echo "[4/5] Chuẩn bị reranker training data..."
  python prepare_rerank_data.py \
    --mode  train \
    --queries "${TRAIN_QUERY_DIR}/train.jsonl" \
    --corpus  "${DATA_DIR}/corpus.jsonl" \
    --trec    "${TRAIN_RANK_TREC}" \
    --output  "${RERANKER_TRAIN_JSONL}" \
    --depth   ${depth} ${pool_flag}
else
  echo "[4/5] Reranker training data đã có — bỏ qua."
fi
n_train=$(wc -l < "${RERANKER_TRAIN_JSONL}")
echo "  ${n_train} training queries"

# Config cho rerank.sh: cùng retriever checkpoint + cùng data variant lúc inference
cat > "${RERANKER_DIR}/reranker_config.json" << EOF
{
  "dataset": "${dataset}",
  "model": "${model}",
  "reranker_model": "${reranker_model}",
  "retriever_dir": "${RETRIEVER_DIR}",
  "retriever_label": "${BEST_LABEL}",
  "retriever_lora": "${BEST_LORA_PATH}",
  "corpus_emb": "${CORPUS_EMB}",
  "eval_variant": "${EVAL_VARIANT}",
  "query_variant": "${QUERY_VARIANT}",
  "query_max_len": ${QUERY_MAX_LEN},
  "depth": ${depth},
  "keep_out_of_pool": ${keep_out_of_pool},
  "train_queries": ${n_train},
  "group_size": ${group_size},
  "epochs": ${epochs},
  "learning_rate": "${lr}",
  "per_device_batch": ${per_batch},
  "gradient_accumulation": ${grad_accum},
  "timestamp": "$(date -Iseconds)"
}
EOF

if [ "${prep_only}" = "true" ]; then
  echo ""
  echo "✓ --prep-only: dữ liệu tại ${RERANKER_TRAIN_JSONL}, bỏ qua training."
  exit 0
fi

# ── Step 5: Train reranker ────────────────────────────────────────────────────

echo ""
echo "[5/5] Training reranker..."
echo ""

ckpt_flag=""
[ "${grad_ckpt}" = "true" ] && ckpt_flag="--gradient_checkpointing"

CUDA_VISIBLE_DEVICES=0 python -m tevatron.reranker.driver.train \
  --model_name_or_path "${reranker_model}" \
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
  --learning_rate ${lr} \
  --warmup_steps 100 \
  --save_steps 500 \
  --logging_steps 50 \
  ${ckpt_flag}

rm -f "${RERANKER_DIR}/INVALID_pad_bug.txt"

echo ""
echo "✓ Reranker training xong!"
echo "✓ Weights lưu tại: ${RERANKER_DIR}"
echo ""
echo "Bước tiếp theo:"
echo "  ./rerank.sh ${dataset}$([ -n "${retriever_tag}" ] && echo " --tag ${retriever_tag}")$([ "${query_variant}" != "auto" ] && echo " --query-variant ${query_variant}")$([ "${reranker_model}" != "${model}" ] && echo " --reranker-model ${reranker_model}")"
