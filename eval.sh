#!/bin/bash
#
# eval.sh — Đánh giá model repLLaMA sau khi training.
#
# Cách sử dụng: ./eval.sh <dataset> [checkpoint] [model] [split] [variant]
#
#   checkpoint (positional, tuỳ chọn):
#     best         (mặc định) sweep tất cả checkpoints trên valid, chọn tốt nhất
#                             theo ndcg_10, sau đó evaluate trên <split>
#     latest       evaluate final model sau khi train xong tất cả epochs
#     base         evaluate base model KHÔNG fine-tune (zero-shot baseline)
#                  → bỏ qua --variant, lưu vào output/<dataset>/<model_tag>-zeroshot/
#     checkpoint-N evaluate một checkpoint cụ thể (ví dụ: checkpoint-1000)
#
#   --model MODEL    : HuggingFace model ID (mặc định: Qwen/Qwen3-Embedding-0.6B)
#   --split SPLIT    : valid | test  (mặc định: test)
#   --variant TAG    : hậu tố model dir, ví dụ 'aug' — bị bỏ qua khi checkpoint=base
#
# Output:
#   best   mode → eval_<split>_best.txt    (dùng bởi show_results.py)
#   latest mode → eval_<split>_latest.txt  (dùng bởi show_results.py)
#   base   mode → eval_<split>.txt         trong thư mục <model_tag>-zeroshot/
#   ckpt   mode → eval_<split>_<ckpt>.txt
#
# Ví dụ:
#   ./eval.sh beauty
#   ./eval.sh beauty latest
#   ./eval.sh beauty base
#   ./eval.sh beauty checkpoint-1000
#   ./eval.sh beauty --variant aug
#   ./eval.sh beauty --split valid
#   ./eval.sh beauty checkpoint-1000 --model Qwen/Qwen3-Embedding-0.6B --split valid
#   ./eval.sh beauty --model Qwen/Qwen3-Embedding-4B --variant aug

set -e
# Chỉ giữ path chứa libcuda.so (driver), bỏ CUDA 11.8 toolkit để tránh xung đột
# với CUDA 12.6 runtime mà torch tự bundle (torch 2.7.1+cu126).
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"

# HuggingFace cache — chuyển sang Data1 để tránh đầy ổ cài HĐH.
# export HF_HOME="/media/administrator/Data1/hf_cache"

# ── Tham số & validation ──────────────────────────────────────────────────────

if [ -z "$1" ]; then
  echo "Lỗi: Bạn chưa nhập tên dataset!"
  echo "Cách sử dụng: ./eval.sh <dataset> [checkpoint] [--model MODEL] [--split SPLIT] [--variant TAG]"
  echo ""
  echo "  dataset          : beauty | sports | ml-1m | steam  (bắt buộc)"
  echo "  checkpoint       : best (mặc định) | latest | base | checkpoint-N"
  echo "  --model MODEL    : HuggingFace model ID (mặc định: Qwen/Qwen3-Embedding-0.6B)"
  echo "  --split SPLIT    : valid | test  (mặc định: test)"
  echo "  --variant TAG    : hậu tố model dir, ví dụ 'aug'"
  echo ""
  echo "Ví dụ:"
  echo "  ./eval.sh beauty"
  echo "  ./eval.sh beauty latest"
  echo "  ./eval.sh beauty base"
  echo "  ./eval.sh beauty checkpoint-1000"
  echo "  ./eval.sh beauty --variant aug"
  echo "  ./eval.sh beauty --split valid"
  echo "  ./eval.sh beauty checkpoint-1000 --model Qwen/Qwen3-Embedding-0.6B --split valid"
  echo "  ./eval.sh beauty --model Qwen/Qwen3-Embedding-4B --variant aug"
  exit 1
fi

dataset=$1
shift

# Checkpoint: positional arg tuỳ chọn — nhận biết nếu không bắt đầu bằng '--'
checkpoint="best"
if [[ $# -gt 0 ]] && [[ "$1" != --* ]]; then
  checkpoint="$1"
  shift
fi

# Defaults
model="Qwen/Qwen3-Embedding-0.6B"
split="test"
variant=""

# Parse named flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   model="$2";   shift 2 ;;
    --split)   split="$2";   shift 2 ;;
    --variant) variant="$2"; shift 2 ;;
    *) echo "Lỗi: Tham số không hợp lệ '$1'"; echo "Chạy ./eval.sh để xem hướng dẫn."; exit 1 ;;
  esac
done

case "$dataset" in
  beauty|sports|ml-1m|steam) ;;
  *) echo "Lỗi: Dataset '${dataset}' không hợp lệ!"; exit 1 ;;
esac
case "$split" in
  valid|test) ;;
  *) echo "Lỗi: Split '${split}' không hợp lệ! Chọn: valid, test"; exit 1 ;;
esac

# ── Paths ─────────────────────────────────────────────────────────────────────

case "${model}" in
  *Qwen3-Embedding-4B*|*Qwen2.5-3B*|*Llama-3.2-3B*|*Llama-3-8B*|*Llama-3.1-8B*|*Llama-3.2-1B*) eval_batch=16 ;;
  *) eval_batch=64 ;;
esac

SELECTION_METRIC="ndcg_10"   # metric dùng để chọn best checkpoint (best mode)

MODEL_TAG=$(basename "${model}" | tr '[:upper:]' '[:lower:]')

if [ "${checkpoint}" = "base" ]; then
  # base mode: không dùng LoRA, bỏ qua variant, lưu vào thư mục -zeroshot
  USE_LORA=false
  MODEL_TAG="${MODEL_TAG}-zeroshot"
else
  USE_LORA=true
  [ -n "${variant}" ] && MODEL_TAG="${MODEL_TAG}-${variant}"
fi

LORA_BASE="./output/${dataset}/${MODEL_TAG}"
DATA_DIR="dataset/dataset/tevatron/${dataset}"
EMB_DIR="${LORA_BASE}/embeddings"
RESULTS_DIR="${EMB_DIR}/results"

echo "════════════════════════════════════════════════"
echo "  Evaluate repLLaMA"
echo "════════════════════════════════════════════════"
echo "  Dataset    : ${dataset}"
echo "  Checkpoint : ${checkpoint}"
echo "  Model      : ${model}"
echo "  Split      : ${split}"
[ -n "${variant}" ] && echo "  Variant    : ${variant}"
echo "════════════════════════════════════════════════"
echo ""

if [ "${checkpoint}" != "base" ] && [ ! -d "${LORA_BASE}" ]; then
  echo "Lỗi: Không tìm thấy model tại ${LORA_BASE}"
  echo "Hãy chạy ./train.sh ${dataset} trước."
  exit 1
fi

mkdir -p "${EMB_DIR}/corpus" "${EMB_DIR}/queries" "${RESULTS_DIR}"

# ── Helper: build qrels từ jsonl (idempotent) ─────────────────────────────────

build_qrels() {
  local s=$1
  local raw="${RESULTS_DIR}/${s}_qrels.txt"
  local clean="${RESULTS_DIR}/${s}_qrels_clean.txt"
  [ -f "${clean}" ] && return

  python << EOF >&2
import json
with open('${DATA_DIR}/${s}.jsonl') as fi, open('${raw}', 'w') as fo:
    for line in fi:
        d = json.loads(line)
        for p in d.get('positive_passages', []):
            fo.write(f"{d['query_id']}\t0\t{p['docid']}\t1\n")
EOF

  python << EOF >&2
seen = set()
with open('${raw}') as fi, open('${clean}', 'w') as fo:
    for line in fi:
        parts = line.strip().split()
        if len(parts) < 3: continue
        key = f"{parts[0]}_{parts[2]}"
        if key not in seen:
            seen.add(key)
            fo.write(line)
print("  qrels OK: ${clean}")
EOF
}

# ── Helper: evaluate một checkpoint trên một split ────────────────────────────
# Args: <ckpt_dir> <split> <label>
#   ckpt_dir : path đến LoRA weights
#   split    : valid | test
#   label    : tên dùng để đặt tên file output (ví dụ: checkpoint-600, latest)
# stdout: giá trị SELECTION_METRIC (để best mode capture)
# files:  eval_<split>_<label>.txt

eval_ckpt() {
  local ckpt_dir=$1
  local s=$2
  local label=$3

  local corpus_emb="${EMB_DIR}/corpus/${label}.pkl"
  local query_emb="${EMB_DIR}/queries/${s}_${label}.pkl"
  local rank_txt="${RESULTS_DIR}/${s}_${label}_rank.txt"
  local rank_trec="${RESULTS_DIR}/${s}_${label}_rank.trec"
  local rank_clean="${RESULTS_DIR}/${s}_${label}_rank_clean.trec"
  local qrels_clean="${RESULTS_DIR}/${s}_qrels_clean.txt"
  local eval_out="${RESULTS_DIR}/eval_${s}_${label}.txt"

  # LoRA flags: có khi fine-tuned, không có khi base mode
  local lora_opts=""
  if [ "${USE_LORA}" = "true" ]; then
    lora_opts="--lora --lora_name_or_path ${ckpt_dir}"
  fi

  # 1. Encode corpus (cache per label)
  if [ ! -f "${corpus_emb}" ]; then
    echo "  [1/4] Encoding corpus (${label})..." >&2
    CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
      --output_dir temp \
      --model_name_or_path ${model} \
      ${lora_opts} \
      --passage_prefix "Passage: " \
      --bf16 --pooling last --padding_side left \
      --append_eos_token --normalize \
      --per_device_eval_batch_size ${eval_batch} \
      --passage_max_len 196 \
      --dataset_name json \
      --dataset_path ${DATA_DIR}/corpus.jsonl \
      --encode_output_path ${corpus_emb} >&2
  else
    echo "  [1/4] Corpus cached, bỏ qua. (${label})" >&2
  fi

  # 2. Encode queries
  echo "  [2/4] Encoding ${s} queries..." >&2
  CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
    --output_dir temp \
    --model_name_or_path ${model} \
    ${lora_opts} \
    --query_prefix "Query: " \
    --bf16 --pooling last --padding_side left \
    --append_eos_token --normalize \
    --per_device_eval_batch_size 64 \
    --query_max_len 128 \
    --dataset_name json \
    --dataset_path ${DATA_DIR}/${s}.jsonl \
    --encode_is_query \
    --encode_output_path ${query_emb} >&2

  # 3. Search
  echo "  [3/4] Searching..." >&2
  python -m tevatron.retriever.driver.search \
    --query_reps ${query_emb} \
    --passage_reps ${corpus_emb} \
    --depth 100 --batch_size 128 \
    --save_text --save_ranking_to ${rank_txt} >&2

  # 4. Convert TREC + deduplicate + compute metrics
  echo "  [4/4] Evaluating..." >&2
  python -m tevatron.utils.format.convert_result_to_trec \
    --input ${rank_txt} --output ${rank_trec} --remove_query >&2

  python << EOF >&2
seen = set()
with open('${rank_trec}') as fi, open('${rank_clean}', 'w') as fo:
    for line in fi:
        parts = line.strip().split()
        if len(parts) < 3: continue
        key = f"{parts[0]}_{parts[2]}"
        if key not in seen:
            seen.add(key)
            fo.write(line)
EOF

  # Ghi kết quả đầy đủ ra file (cho show_results.py)
  python compute_metrics.py \
    --run ${rank_clean} --qrels ${qrels_clean} \
    --ks 5,10,20 --out ${eval_out} >&2

  # stdout: chỉ giá trị metric (để best mode capture qua $(...))
  python compute_metrics.py \
    --run ${rank_clean} --qrels ${qrels_clean} \
    --ks 5,10,20 2>/dev/null \
    | awk -v m="${SELECTION_METRIC}" '$1==m && $2=="all" {print $3}'
}

# ── Helper: in kết quả đầy đủ ─────────────────────────────────────────────────

print_results() {
  local label=$1
  local s=$2
  local eval_out="${RESULTS_DIR}/eval_${s}_${label}.txt"
  echo ""
  echo "════════════════════════════════════════════════"
  echo "  Results — ${dataset} / ${label} / ${s}"
  echo "════════════════════════════════════════════════"
  cat "${eval_out}"
  echo ""
  echo "✓ Kết quả lưu tại: ${eval_out}"
}

# ══════════════════════════════════════════════════
# MODE: BEST — sweep tất cả checkpoints
# ══════════════════════════════════════════════════

if [ "${checkpoint}" = "best" ]; then

  CHECKPOINTS=()
  for d in $(ls -d ${LORA_BASE}/checkpoint-* 2>/dev/null | sort -V); do
    CHECKPOINTS+=("$d")
  done
  if [ -f "${LORA_BASE}/adapter_model.safetensors" ] || \
     [ -f "${LORA_BASE}/adapter_model.bin" ]; then
    CHECKPOINTS+=("${LORA_BASE}")
  fi

  if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "Lỗi: Không tìm thấy checkpoint nào trong ${LORA_BASE}"
    echo "Hãy chạy ./train.sh ${dataset} trước."
    exit 1
  fi

  echo "Tìm thấy ${#CHECKPOINTS[@]} checkpoint(s):"
  for ckpt in "${CHECKPOINTS[@]}"; do echo "  - $(basename ${ckpt})"; done
  echo ""

  LOG_FILE="${RESULTS_DIR}/checkpoint_selection.log"
  printf "%-30s  %s\n" "checkpoint" "${SELECTION_METRIC}" > "${LOG_FILE}"
  build_qrels "valid"

  best_metric=-1
  best_ckpt=""
  best_label=""

  for ckpt in "${CHECKPOINTS[@]}"; do
    label=$(basename "${ckpt}")
    echo "──────────────────────────────────────────────"
    echo "Checkpoint: ${label}"

    val=$(eval_ckpt "${ckpt}" "valid" "${label}")
    [ -z "${val}" ] && val=0

    echo "  ${SELECTION_METRIC} (valid) = ${val}"
    printf "%-30s  %s\n" "${label}" "${val}" >> "${LOG_FILE}"

    is_better=$(python -c "print(1 if ${val} > ${best_metric} else 0)")
    if [ "${is_better}" = "1" ]; then
      best_metric=${val}
      best_ckpt=${ckpt}
      best_label=${label}
      echo "  ★ New best!"
    fi
    echo ""
  done

  echo "════════════════════════════════════════════════"
  echo "BEST: ${best_label}  (${SELECTION_METRIC} valid = ${best_metric})"
  echo "════════════════════════════════════════════════"
  echo ""
  echo "Evaluating best checkpoint on ${split} set..."
  echo ""

  build_qrels "${split}"
  eval_ckpt "${best_ckpt}" "${split}" "${best_label}" > /dev/null

  # Copy → eval_<split>_best.txt (chuẩn cho show_results.py)
  best_out="${RESULTS_DIR}/eval_${split}_best.txt"
  cp "${RESULTS_DIR}/eval_${split}_${best_label}.txt" "${best_out}"
  echo "" >> "${best_out}"
  echo "Best checkpoint : ${best_label}" >> "${best_out}"
  echo "Selection metric: ${SELECTION_METRIC} (valid) = ${best_metric}" >> "${best_out}"

  print_results "${best_label}" "${split}"
  echo "✓ Log checkpoint selection : ${LOG_FILE}"
  echo "✓ Kết quả best lưu tại     : ${best_out}"

# ══════════════════════════════════════════════════
# MODE: BASE — base model không fine-tune (zero-shot)
# ══════════════════════════════════════════════════

elif [ "${checkpoint}" = "base" ]; then

  build_qrels "${split}"
  eval_ckpt "" "${split}" "base" > /dev/null

  # Copy → eval_<split>.txt để show_results.py nhận ra (chuẩn zeroshot)
  cp "${RESULTS_DIR}/eval_${split}_base.txt" "${RESULTS_DIR}/eval_${split}.txt"

  print_results "base" "${split}"

# ══════════════════════════════════════════════════
# MODE: LATEST hoặc CHECKPOINT CỤ THỂ
# ══════════════════════════════════════════════════

else
  if [ "${checkpoint}" = "latest" ]; then
    CKPT_DIR="${LORA_BASE}"
    LABEL="latest"
    if [ ! -f "${LORA_BASE}/adapter_model.safetensors" ] && \
       [ ! -f "${LORA_BASE}/adapter_model.bin" ]; then
      echo "Lỗi: Không tìm thấy adapter weights tại ${LORA_BASE}"
      echo "Hãy chạy ./train.sh ${dataset} trước."
      exit 1
    fi
  else
    CKPT_DIR="${LORA_BASE}/${checkpoint}"
    LABEL="${checkpoint}"
    if [ ! -d "${CKPT_DIR}" ]; then
      echo "Lỗi: Không tìm thấy checkpoint tại ${CKPT_DIR}"
      echo ""
      echo "Các checkpoint hiện có:"
      ls -d ${LORA_BASE}/checkpoint-* 2>/dev/null | sort -V | \
        while read d; do echo "  $(basename $d)"; done
      [ -f "${LORA_BASE}/adapter_model.safetensors" ] && echo "  latest"
      exit 1
    fi
  fi

  build_qrels "${split}"
  eval_ckpt "${CKPT_DIR}" "${split}" "${LABEL}" > /dev/null
  print_results "${LABEL}" "${split}"

fi
