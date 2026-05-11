#!/bin/bash
set -e

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Lỗi: Thiếu tham số!"
  echo "Cách sử dụng: ./eval_one.sh <dataset> <checkpoint> [model] [split]"
  echo "  dataset    : beauty | sports | ml-1m | steam"
  echo "  checkpoint : tên checkpoint (vd: checkpoint-200) hoặc 'final'"
  echo "  model      : HuggingFace model ID (mặc định: Qwen/Qwen3-Embedding-0.6B)"
  echo "  split      : valid | test  (mặc định: test)"
  echo ""
  echo "Ví dụ:"
  echo "  ./eval_one.sh beauty checkpoint-600"
  echo "  ./eval_one.sh beauty final"
  echo "  ./eval_one.sh beauty checkpoint-600 Qwen/Qwen3-Embedding-0.6B valid"
  exit 1
fi

dataset=$1
checkpoint=$2
model=${3:-"Qwen/Qwen3-Embedding-0.6B"}
split=${4:-test}

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

case "$split" in
  valid|test) ;;
  *)
    echo "Lỗi: Split '${split}' không hợp lệ! Chọn: valid, test"
    exit 1
    ;;
esac

MODEL_TAG=$(basename "${model}" | tr '[:upper:]' '[:lower:]')
LORA_BASE="./output/${dataset}/${MODEL_TAG}"
DATA_DIR="dataset/dataset/tevatron/${dataset}"
EMB_DIR="${LORA_BASE}/embeddings"
RESULTS_DIR="${EMB_DIR}/results"

# Resolve checkpoint path
if [ "${checkpoint}" = "final" ]; then
  CKPT_DIR="${LORA_BASE}"
else
  CKPT_DIR="${LORA_BASE}/${checkpoint}"
fi

if [ ! -d "${CKPT_DIR}" ]; then
  echo "Lỗi: Không tìm thấy checkpoint tại ${CKPT_DIR}"
  echo ""
  echo "Các checkpoint hiện có:"
  ls -d ${LORA_BASE}/checkpoint-* 2>/dev/null | sort -V | while read d; do
    echo "  $(basename $d)"
  done
  [ -f "${LORA_BASE}/adapter_model.safetensors" ] && echo "  final"
  exit 1
fi

CKPT_NAME=$(basename "${CKPT_DIR}")
echo "Dataset    : ${dataset}"
echo "Checkpoint : ${CKPT_NAME}"
echo "Model      : ${model}"
echo "Split      : ${split}"
echo ""

mkdir -p "${EMB_DIR}/corpus" "${EMB_DIR}/queries" "${RESULTS_DIR}"

CORPUS_EMB="${EMB_DIR}/corpus/${CKPT_NAME}.pkl"
QUERY_EMB="${EMB_DIR}/queries/${split}_${CKPT_NAME}.pkl"
RANK_TXT="${RESULTS_DIR}/${split}_${CKPT_NAME}_rank.txt"
RANK_TREC="${RESULTS_DIR}/${split}_${CKPT_NAME}_rank.trec"
RANK_CLEAN="${RESULTS_DIR}/${split}_${CKPT_NAME}_rank_clean.trec"
QRELS_RAW="${RESULTS_DIR}/${split}_qrels.txt"
QRELS_CLEAN="${RESULTS_DIR}/${split}_qrels_clean.txt"
EVAL_OUT="${RESULTS_DIR}/eval_${split}_${CKPT_NAME}.txt"

# ── Bước 1: Encode corpus (bỏ qua nếu đã có) ─────────────────────────────────
if [ -f "${CORPUS_EMB}" ]; then
  echo "── [1/4] Corpus embedding đã có, bỏ qua."
else
  echo "── [1/4] Encoding corpus..."
  CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
    --output_dir temp \
    --model_name_or_path ${model} \
    --lora \
    --lora_name_or_path ${CKPT_DIR} \
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
    --encode_output_path ${CORPUS_EMB}
fi

# ── Bước 2: Encode queries ────────────────────────────────────────────────────
echo "── [2/4] Encoding ${split} queries..."
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --output_dir temp \
  --model_name_or_path ${model} \
  --lora \
  --lora_name_or_path ${CKPT_DIR} \
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
  --encode_output_path ${QUERY_EMB}

# ── Bước 3: Search ────────────────────────────────────────────────────────────
echo "── [3/4] Searching..."
python -m tevatron.retriever.driver.search \
  --query_reps ${QUERY_EMB} \
  --passage_reps ${CORPUS_EMB} \
  --depth 100 \
  --batch_size 128 \
  --save_text \
  --save_ranking_to ${RANK_TXT}

# ── Bước 4: Evaluate ──────────────────────────────────────────────────────────
echo "── [4/4] Evaluating..."

python -m tevatron.utils.format.convert_result_to_trec \
  --input ${RANK_TXT} \
  --output ${RANK_TREC} \
  --remove_query

# Build qrels nếu chưa có
if [ ! -f "${QRELS_CLEAN}" ]; then
  python << EOF
import json
with open('${DATA_DIR}/${split}.jsonl') as f_in, \
     open('${QRELS_RAW}', 'w') as f_out:
    for line in f_in:
        d = json.loads(line)
        for p in d.get('positive_passages', []):
            f_out.write(f"{d['query_id']}\t0\t{p['docid']}\t1\n")
EOF
  python << EOF
seen = set()
with open('${QRELS_RAW}') as f_in, open('${QRELS_CLEAN}', 'w') as f_out:
    for line in f_in:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        key = f"{parts[0]}_{parts[2]}"
        if key not in seen:
            seen.add(key)
            f_out.write(line)
EOF
fi

python << EOF
seen = set()
with open('${RANK_TREC}') as f_in, open('${RANK_CLEAN}', 'w') as f_out:
    for line in f_in:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        key = f"{parts[0]}_{parts[2]}"
        if key not in seen:
            seen.add(key)
            f_out.write(line)
EOF

echo ""
echo "══════════════════════════════════════════════════"
echo "RESULTS — ${dataset} / ${CKPT_NAME} / ${split}"
echo "══════════════════════════════════════════════════"

python compute_metrics.py \
  --run   ${RANK_CLEAN} \
  --qrels ${QRELS_CLEAN} \
  --ks    5,10,20 \
  --out   ${EVAL_OUT} \
  | tee ${EVAL_OUT}

echo ""
echo "✓ Kết quả đã lưu tại: ${EVAL_OUT}"
