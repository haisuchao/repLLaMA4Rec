#!/bin/bash
#
# rerank.sh — Rerank top-K của retriever bằng cross-encoder đã train (rankLLaMA) + fuse với retriever.
#
# Cách sử dụng:
#   ./rerank.sh <dataset> [--model MODEL] [--tag RTAG] [--query-variant V] [--split SPLIT]
#               [--reranker-ckpt DIR] [--alpha A] [--valid-sample N] [--all-queries]
#               [--batch-size N] [--force]
#
#   <dataset>           : beauty | sports | toys | ml-1m
#   --model MODEL       : base model của retriever (mặc định: Qwen/Qwen3-Embedding-0.6B)
#   --reranker-model M  : backbone reranker (mặc định: = --model; khớp train_reranker.sh)
#   --tag RTAG          : tag của retriever (khớp train_reranker.sh --tag)
#   --query-variant V   : khớp train_reranker.sh --query-variant (mặc định: auto)
#   --split SPLIT       : valid | test (mặc định: test)
#   --reranker-ckpt DIR : checkpoint reranker cụ thể (mặc định: final model)
#   --alpha A           : trọng số retriever khi fuse; bỏ trống = chọn trên mẫu valid theo NDCG@10
#   --valid-sample N    : số query valid (có positive trong top-K) dùng để chọn α (mặc định: 2000)
#   --all-queries       : rerank mọi query. Mặc định chỉ rerank query có positive trong top-K — metric
#                         trùng với chạy đủ (các query còn lại mọi hệ thống đều miss) nhưng nhanh ~3.5×
#   --batch-size N      : batch inference (mặc định: 256; 64 với reranker 1B-4B)
#   --force             : tạo lại pairs + scores + fuse (vẫn giữ retriever trec cache)
#
# Pipeline (thiết kế + bằng chứng: diag/README.md):
#   1. Retriever (checkpoint + data variant ghi trong reranker_config.json) → top-K cho valid/test
#   2. Loại item lịch sử user (history filter, mode full)
#   3. Reranker score từng (query, candidate)
#   4. Fuse: α·z(retriever) + (1−α)·z(reranker), chuẩn hoá z-score trong từng query; α chọn trên valid
#   5. Bảng so sánh retriever raw / retriever filtered / reranker / fused
#
# Điều kiện tiên quyết: train_reranker.sh đã chạy (tạo reranker_config.json + weights)
#
# Ví dụ:
#   ./rerank.sh beauty --tag aug-5
#   ./rerank.sh sports --tag cs5-aug-gs32
#   ./rerank.sh beauty --tag aug-5 --alpha 0.3            # α cố định, không cần chạy valid
#   ./rerank.sh beauty --tag aug-5 --reranker-model Qwen/Qwen3-Embedding-4B
#   ./rerank.sh beauty --tag aug-5 --reranker-ckpt output/beauty/qwen3-embedding-0.6b-aug-5-reranker-aug/checkpoint-5000

set -e
shopt -s inherit_errexit   # lỗi trong $(retrieve ...)/$(score ...) cũng dừng script
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"

# ── Parse args ────────────────────────────────────────────────────────────────

if [ -z "$1" ]; then
  echo "Lỗi: Cần nhập dataset!"
  echo "Dùng: ./rerank.sh <dataset> [--model MODEL] [--tag RTAG] [--query-variant V] [--split SPLIT] [--reranker-ckpt DIR] [--alpha A] [--valid-sample N] [--all-queries] [--force]"
  exit 1
fi

dataset=$1; shift

model="Qwen/Qwen3-Embedding-0.6B"
reranker_model=""
retriever_tag=""
query_variant="auto"
split="test"
reranker_ckpt=""
alpha=""
valid_sample=2000
scope="inpool"
batch_size=""
force=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          model="$2";          shift 2 ;;
    --reranker-model) reranker_model="$2"; shift 2 ;;
    --tag)            retriever_tag="$2";  shift 2 ;;
    --query-variant)  query_variant="$2";  shift 2 ;;
    --split)          split="$2";          shift 2 ;;
    --reranker-ckpt)  reranker_ckpt="$2";  shift 2 ;;
    --alpha)          alpha="$2";          shift 2 ;;
    --valid-sample)   valid_sample="$2";   shift 2 ;;
    --all-queries)    scope="all";         shift   ;;
    --batch-size)     batch_size="$2";     shift 2 ;;
    --force)          force=true;          shift   ;;
    *) echo "Lỗi: Tham số không hợp lệ '$1'"; exit 1 ;;
  esac
done

case "$split" in
  valid|test) ;;
  *) echo "Lỗi: Split '${split}' không hợp lệ! Chọn: valid, test"; exit 1 ;;
esac

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_TAG=$(basename "${model}" | tr '[:upper:]' '[:lower:]')
[ -n "${retriever_tag}" ] && MODEL_TAG="${MODEL_TAG}-${retriever_tag}"
RETRIEVER_DIR="./output/${dataset}/${MODEL_TAG}"

eval "$(python rerank_pipeline.py resolve --dataset "${dataset}" --retriever-dir "${RETRIEVER_DIR}" \
          --query-variant "${query_variant}")"
reranker_model="${reranker_model:-${model}}"
RERANKER_DIR="${RETRIEVER_DIR}-reranker${QUERY_VARIANT:+-${QUERY_VARIANT}}"
[ "${reranker_model}" != "${model}" ] && RERANKER_DIR="${RERANKER_DIR}-$(basename "${reranker_model}" | tr '[:upper:]' '[:lower:]')"
case "${reranker_model}" in
  *4B*|*3B*|*1B*) batch_size="${batch_size:-64}" ;;
  *)              batch_size="${batch_size:-256}" ;;
esac
CONFIG="${RERANKER_DIR}/reranker_config.json"

if [ ! -f "${CONFIG}" ]; then
  echo "Lỗi: Không tìm thấy ${CONFIG}"
  echo "Hãy chạy: ./train_reranker.sh ${dataset}$([ -n "${retriever_tag}" ] && echo " --tag ${retriever_tag}")$([ "${query_variant}" != "auto" ] && echo " --query-variant ${query_variant}")"
  exit 1
fi
cfg() { python -c "import json,sys; print(json.load(open('${CONFIG}'))[sys.argv[1]])" "$1"; }
RETRIEVER_LORA=$(cfg retriever_lora)
RETRIEVER_LABEL=$(cfg retriever_label)
CORPUS_EMB=$(cfg corpus_emb)
depth=$(cfg depth)

[ -z "${reranker_ckpt}" ] && reranker_ckpt="${RERANKER_DIR}"
if [ ! -f "${reranker_ckpt}/adapter_model.safetensors" ] && \
   [ ! -f "${reranker_ckpt}/adapter_model.bin" ]; then
  echo "Lỗi: Không tìm thấy reranker weights tại ${reranker_ckpt}"
  exit 1
fi
[ -f "${reranker_ckpt}/INVALID_pad_bug.txt" ] && { echo "Lỗi: ${reranker_ckpt} là reranker cũ train khi còn pad bug — train lại."; exit 1; }
RLABEL=$([ "${reranker_ckpt}" = "${RERANKER_DIR}" ] && echo final || basename "${reranker_ckpt}")

DATA_DIR="./dataset/dataset/tevatron/${dataset}"
EVAL_DATA_DIR="${DATA_DIR}${EVAL_VARIANT:+-${EVAL_VARIANT}}"
INFER_DIR="${RERANKER_DIR}/inference"
mkdir -p "${INFER_DIR}"

case "${model}" in
  *4B*|*3B*|*1B*) eval_batch=16 ;;
  *)              eval_batch=64 ;;
esac

echo "════════════════════════════════════════════════"
echo "  Rerank (rankLLaMA) + fuse"
echo "════════════════════════════════════════════════"
echo "  Dataset        : ${dataset} / ${split}"
echo "  Retriever      : ${RETRIEVER_LORA} (${RETRIEVER_LABEL}), top-${depth}"
echo "  Queries        : ${EVAL_DATA_DIR}/${split}.jsonl"
echo "  Reranker       : ${reranker_ckpt} (${RLABEL}, ${reranker_model})"
echo "  Rerank scope   : $([ "${scope}" = "all" ] && echo 'mọi query' || echo 'query có positive trong top-K (metric trùng với chạy đủ)')"
echo "  α              : ${alpha:-chọn trên ${valid_sample} query valid}"
echo "════════════════════════════════════════════════"
echo ""

# ── Step 1: Retriever top-K (cache theo split) ────────────────────────────────

retrieve() {  # $1 = split
  local s=$1
  local trec="${INFER_DIR}/${s}_retriever_rank.trec"
  if [ -f "${trec}" ]; then
    echo "[1/4] Retriever ${s} top-${depth} đã có — bỏ qua." >&2
  else
    echo "[1/4] Retriever: encode ${s} queries + search top-${depth}..." >&2
    CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
      --output_dir temp \
      --model_name_or_path "${model}" \
      --lora --lora_name_or_path "${RETRIEVER_LORA}" \
      --query_prefix "Query: " \
      --bf16 --pooling last --padding_side left \
      --append_eos_token --normalize \
      --per_device_eval_batch_size ${eval_batch} \
      --query_max_len ${QUERY_MAX_LEN} \
      --dataset_name json \
      --dataset_path "${EVAL_DATA_DIR}/${s}.jsonl" \
      --encode_is_query \
      --encode_output_path "${INFER_DIR}/${s}_queries.pkl" >&2
    python -m tevatron.retriever.driver.search \
      --query_reps "${INFER_DIR}/${s}_queries.pkl" \
      --passage_reps "${CORPUS_EMB}" \
      --depth ${depth} --batch_size 256 \
      --save_text --save_ranking_to "${INFER_DIR}/${s}_retriever_rank.txt" >&2
    python -m tevatron.utils.format.convert_result_to_trec \
      --input "${INFER_DIR}/${s}_retriever_rank.txt" --output "${trec}" --remove_query >&2
  fi
  echo "${trec}"
}

# ── Step 2+3: pairs (history-filtered) → reranker scores ──────────────────────

score() {  # $1 = split, $2 = trec, $3 = pairs scope (inpool|all|sample)
  local s=$1 trec=$2 sc=$3
  local pairs="${INFER_DIR}/${s}_pairs_${sc}.jsonl"
  local scores="${INFER_DIR}/${s}_${sc}_${RLABEL}_scores.txt"
  if [ ! -f "${pairs}" ] || [ "${force}" = "true" ]; then
    echo "[2/4] Pairs ${s} (loại item lịch sử)..." >&2
    local extra=""
    [ "${sc}" = "sample" ] && extra="--scope inpool --sample ${valid_sample}" || extra="--scope ${sc}"
    python rerank_pipeline.py pairs --dataset "${dataset}" --split "${s}" \
      --queries "${EVAL_DATA_DIR}/${s}.jsonl" --corpus "${DATA_DIR}/corpus.jsonl" \
      --trec "${trec}" --depth ${depth} --output "${pairs}" ${extra} >&2
  fi
  if [ ! -f "${scores}" ] || [ "${force}" = "true" ]; then
    echo "[3/4] Reranking ${s} ($(wc -l < "${pairs}") pairs)..." >&2
    CUDA_VISIBLE_DEVICES=0 python -m tevatron.reranker.driver.rerank \
      --model_name_or_path "${reranker_model}" \
      --lora_name_or_path  "${reranker_ckpt}" \
      --dataset_name json \
      --dataset_path "${pairs}" \
      --dataset_split train \
      --rerank_output_path "${scores}.tmp" \
      --rerank_max_len 256 \
      --append_eos_token \
      --per_device_eval_batch_size ${batch_size} \
      --output_dir temp \
      --bf16 >&2
    mv "${scores}.tmp" "${scores}"
  else
    echo "[3/4] Reranker scores ${s} đã có — bỏ qua." >&2
  fi
  echo "${scores}"
}

TREC=$(retrieve "${split}")
SCORES=$(score "${split}" "${TREC}" "${scope}")

fuse_args=()
if [ -n "${alpha}" ]; then
  fuse_args=(--alpha "${alpha}")
else
  VALID_TREC=$(retrieve valid)
  VALID_SCORES=$(score valid "${VALID_TREC}" sample)
  fuse_args=(--valid-queries "${EVAL_DATA_DIR}/valid.jsonl" --valid-trec "${VALID_TREC}" \
             --valid-scores "${VALID_SCORES}" --alpha-json "${INFER_DIR}/alpha_${RLABEL}.json")
fi

# ── Step 4: Fuse + evaluate ───────────────────────────────────────────────────

EVAL_OUT="${INFER_DIR}/eval_${split}_${RLABEL}.txt"
echo "[4/4] Fuse + evaluate..."
python rerank_pipeline.py fuse --dataset "${dataset}" --split "${split}" \
  --queries "${EVAL_DATA_DIR}/${split}.jsonl" --trec "${TREC}" --scores "${SCORES}" \
  --depth ${depth} "${fuse_args[@]}" \
  --out-trec "${INFER_DIR}/${split}_${RLABEL}_fused.trec" --out-eval "${EVAL_OUT}"

echo ""
echo "✓ Kết quả lưu tại: ${EVAL_OUT}"
