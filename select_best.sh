#!/bin/bash

if [ -z "$1" ]; then
  echo "Lỗi: Bạn chưa nhập tên dataset!"
  echo "Cách sử dụng: ./select_best.sh <dataset> [model] [metric]"
  echo "  dataset : beauty | sports | ml-1m | steam"
  echo "  model   : HuggingFace model ID (mặc định: Qwen/Qwen3-Embedding-0.6B)"
  echo "  metric  : ndcg_5 | ndcg_10 | ndcg_20 | hr_5 | hr_10 | hr_20 | mrr_5 | mrr_10 | mrr_20"
  echo "            (mặc định: ndcg_10)"
  echo "Ví dụ: ./select_best.sh beauty"
  echo "Ví dụ: ./select_best.sh beauty Qwen/Qwen3-Embedding-0.6B mrr_10"
  exit 1
fi

dataset=$1
model=${2:-"Qwen/Qwen3-Embedding-0.6B"}
metric=${3:-"ndcg_10"}

case "${model}" in
  *Llama-3.2-3B*|*Llama-3-8B*|*Llama-3.1-8B*|*Llama-3.2-1B*)
    eval_batch=16
    ;;
  *)
    eval_batch=64
    ;;
esac

case "$dataset" in
  beauty|sports|ml-1m|steam) ;;
  *)
    echo "Lỗi: Dataset '${dataset}' không hợp lệ!"
    exit 1
    ;;
esac

case "$metric" in
  ndcg_5|ndcg_10|ndcg_20|hr_5|hr_10|hr_20|mrr_5|mrr_10|mrr_20) ;;
  *)
    echo "Lỗi: Metric '${metric}' không hợp lệ!"
    echo "Chọn: ndcg_5, ndcg_10, ndcg_20, hr_5, hr_10, hr_20, mrr_5, mrr_10, mrr_20"
    exit 1
    ;;
esac

MODEL_TAG=$(basename "${model}" | tr '[:upper:]' '[:lower:]')
LORA_BASE="./output/${dataset}/${MODEL_TAG}"
DATA_DIR="dataset/dataset/tevatron/${dataset}"
EMB_DIR="${LORA_BASE}/embeddings"
RESULTS_DIR="${EMB_DIR}/results"
LOG_FILE="${RESULTS_DIR}/checkpoint_selection.log"

echo "Dataset : ${dataset}"
echo "Model   : ${model}"
echo "Metric  : ${metric}"
echo ""

# ── Tìm tất cả checkpoints ────────────────────────────────────────────────────
CHECKPOINTS=()
for d in $(ls -d ${LORA_BASE}/checkpoint-* 2>/dev/null | sort -V); do
  CHECKPOINTS+=("$d")
done
# Thêm final model nếu có adapter weights
if [ -f "${LORA_BASE}/adapter_model.safetensors" ] || \
   [ -f "${LORA_BASE}/adapter_model.bin" ]; then
  CHECKPOINTS+=("${LORA_BASE}")
fi

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
  echo "Lỗi: Không tìm thấy checkpoint nào trong ${LORA_BASE}"
  echo "Hãy chạy train.sh trước."
  exit 1
fi

echo "Tìm thấy ${#CHECKPOINTS[@]} checkpoint(s):"
for ckpt in "${CHECKPOINTS[@]}"; do
  echo "  - $(basename ${ckpt})"
done
echo ""

mkdir -p "${RESULTS_DIR}"
printf "%-30s  %s\n" "checkpoint" "${metric}" > "${LOG_FILE}"

# ── Helper: build qrels từ jsonl (chỉ cần làm 1 lần mỗi split) ───────────────
build_qrels() {
  local split=$1
  local qrels_raw="${RESULTS_DIR}/${split}_qrels.txt"
  local qrels_clean="${RESULTS_DIR}/${split}_qrels_clean.txt"

  if [ -f "${qrels_clean}" ]; then
    return
  fi

  python << EOF >&2
import json
with open('${DATA_DIR}/${split}.jsonl') as f_in, \
     open('${qrels_raw}', 'w') as f_out:
    for line in f_in:
        d = json.loads(line)
        for p in d.get('positive_passages', []):
            f_out.write(f"{d['query_id']}\t0\t{p['docid']}\t1\n")
EOF

  python << EOF >&2
seen = set()
with open('${qrels_raw}') as f_in, \
     open('${qrels_clean}', 'w') as f_out:
    for line in f_in:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        key = f"{parts[0]}_{parts[2]}"
        if key not in seen:
            seen.add(key)
            f_out.write(line)
EOF
}

# ── Helper: encode + search + evaluate 1 checkpoint trên 1 split ─────────────
# Tất cả log/progress đi ra stderr; chỉ metric value (1 số) đi ra stdout
# để $(eval_checkpoint ...) capture đúng giá trị.
eval_checkpoint() {
  local ckpt=$1
  local split=$2
  local ckpt_name=$(basename "${ckpt}")

  local corpus_emb="${EMB_DIR}/corpus/${ckpt_name}.pkl"
  local query_emb="${EMB_DIR}/queries/${split}_${ckpt_name}.pkl"
  local rank_txt="${RESULTS_DIR}/${split}_${ckpt_name}_rank.txt"
  local rank_trec="${RESULTS_DIR}/${split}_${ckpt_name}_rank.trec"
  local rank_clean="${RESULTS_DIR}/${split}_${ckpt_name}_rank_clean.trec"
  local qrels_clean="${RESULTS_DIR}/${split}_qrels_clean.txt"

  # 1. Encode corpus (bỏ qua nếu đã có)
  if [ ! -f "${corpus_emb}" ]; then
    echo "  [1/4] Encoding corpus..." >&2
    mkdir -p "${EMB_DIR}/corpus"
    CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
      --output_dir temp \
      --model_name_or_path ${model} \
      --lora \
      --lora_name_or_path ${ckpt} \
      --passage_prefix "Passage: " \
      --bf16 \
      --pooling last \
      --padding_side left \
      --append_eos_token \
      --normalize \
      --per_device_eval_batch_size ${eval_batch} \
      --passage_max_len 196 \
      --dataset_name json \
      --dataset_path ${DATA_DIR}/corpus.jsonl \
      --encode_output_path ${corpus_emb} >&2
  else
    echo "  [1/4] Corpus embedding đã có, bỏ qua." >&2
  fi

  # 2. Encode queries
  echo "  [2/4] Encoding ${split} queries..." >&2
  mkdir -p "${EMB_DIR}/queries"
  CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
    --output_dir temp \
    --model_name_or_path ${model} \
    --lora \
    --lora_name_or_path ${ckpt} \
    --query_prefix "Query: " \
    --bf16 \
    --pooling last \
    --padding_side left \
    --append_eos_token \
    --normalize \
    --per_device_eval_batch_size 64 \
    --query_max_len 128 \
    --dataset_name json \
    --dataset_path ${DATA_DIR}/${split}.jsonl \
    --encode_is_query \
    --encode_output_path ${query_emb} >&2

  # 3. Search
  echo "  [3/4] Searching..." >&2
  python -m tevatron.retriever.driver.search \
    --query_reps ${query_emb} \
    --passage_reps ${corpus_emb} \
    --depth 100 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to ${rank_txt} >&2

  # 4. Convert + deduplicate
  echo "  [4/4] Evaluating..." >&2
  python -m tevatron.utils.format.convert_result_to_trec \
    --input ${rank_txt} \
    --output ${rank_trec} \
    --remove_query >&2

  python << EOF >&2
seen = set()
with open('${rank_trec}') as f_in, \
     open('${rank_clean}', 'w') as f_out:
    for line in f_in:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        key = f"{parts[0]}_{parts[2]}"
        if key not in seen:
            seen.add(key)
            f_out.write(line)
EOF

  # stdout: chỉ 1 số duy nhất — metric value để caller capture
  python compute_metrics.py \
    --run   ${rank_clean} \
    --qrels ${qrels_clean} \
    --ks    5,10,20 2>/dev/null \
    | awk -v m="${metric}" '$1 == m && $2 == "all" {print $3}'
}

# ── Sweep tất cả checkpoints trên valid set ───────────────────────────────────
build_qrels "valid"

best_metric=-1
best_ckpt=""

for ckpt in "${CHECKPOINTS[@]}"; do
  ckpt_name=$(basename "${ckpt}")
  echo "══════════════════════════════════════════════════"
  echo "Checkpoint: ${ckpt_name}"
  echo "══════════════════════════════════════════════════"

  metric_value=$(eval_checkpoint "${ckpt}" "valid")

  if [ -z "${metric_value}" ]; then
    metric_value=0
  fi

  echo ""
  echo "  ${metric} (valid) = ${metric_value}"
  printf "%-30s  %s\n" "${ckpt_name}" "${metric_value}" >> "${LOG_FILE}"

  is_better=$(python -c "print(1 if ${metric_value} > ${best_metric} else 0)")
  if [ "${is_better}" = "1" ]; then
    best_metric=${metric_value}
    best_ckpt=${ckpt}
    echo "  ★ New best!"
  fi
  echo ""
done

best_ckpt_name=$(basename "${best_ckpt}")

echo "══════════════════════════════════════════════════"
echo "BEST CHECKPOINT : ${best_ckpt_name}"
echo "BEST ${metric} (valid) : ${best_metric}"
echo "══════════════════════════════════════════════════"
echo ""

# ── Evaluate best checkpoint trên test set ────────────────────────────────────
echo "Evaluating best checkpoint on TEST set..."
echo ""

build_qrels "test"

EVAL_OUT="${RESULTS_DIR}/eval_test_best.txt"

metric_value_test=$(eval_checkpoint "${best_ckpt}" "test")

echo ""
echo "══════════════════════════════════════════════════"
echo "TEST RESULTS — best checkpoint: ${best_ckpt_name}"
echo "══════════════════════════════════════════════════"

# In lại toàn bộ metrics (không chỉ metric chính)
rank_clean="${RESULTS_DIR}/test_${best_ckpt_name}_rank_clean.trec"
qrels_clean="${RESULTS_DIR}/test_qrels_clean.txt"

python compute_metrics.py \
  --run   ${rank_clean} \
  --qrels ${qrels_clean} \
  --ks    5,10,20 \
  --out   ${EVAL_OUT} \
  | tee ${EVAL_OUT}

echo "" >> "${EVAL_OUT}"
echo "Best checkpoint : ${best_ckpt_name}" >> "${EVAL_OUT}"
echo "Selection metric: ${metric} (valid) = ${best_metric}" >> "${EVAL_OUT}"

echo ""
echo "✓ Kết quả test       : ${EVAL_OUT}"
echo "✓ Log valid sweep    : ${LOG_FILE}"
