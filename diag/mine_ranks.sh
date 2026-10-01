#!/bin/bash
# Chẩn đoán negative cho reranker — encode corpus + train/valid/test queries bằng
# retriever, FAISS top-DEPTH, xuất trec cho diag/analyze_negatives.py.
#
# Dùng: bash diag/mine_ranks.sh <dataset> <retriever_tag> [depth] [data_variant] [splits]
#   ví dụ: bash diag/mine_ranks.sh beauty aug-5 200
#          bash diag/mine_ranks.sh beauty aug-5 100 aug train   # query augmented → aug_train_rank.trec

set -e
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"

dataset=$1
rtag=$2
depth=${3:-200}
variant=${4:-}              # data variant (vd: aug) → query lấy từ tevatron/<dataset>-<variant>/, file có tiền tố <variant>_
splits=${5:-"train valid test"}
model="Qwen/Qwen3-Embedding-0.6B"

RETRIEVER_DIR="./output/${dataset}/qwen3-embedding-0.6b-${rtag}"
DATA_DIR="./dataset/dataset/tevatron/${dataset}"
QUERY_DIR="${DATA_DIR}${variant:+-${variant}}"
PFX="${variant:+${variant}_}"
OUT="${RETRIEVER_DIR}/diag"
mkdir -p "${OUT}"

encode() {  # $1=input jsonl  $2=output pkl  $3=query|passage
  [ -f "$2" ] && { echo "  reuse $(basename $2)"; return; }
  local extra
  if [ "$3" = "query" ]; then
    extra=(--query_prefix "Query: " --query_max_len 128 --encode_is_query)
  else
    extra=(--passage_prefix "Passage: " --passage_max_len 196)
  fi
  CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
    --output_dir temp \
    --model_name_or_path "${model}" \
    --lora --lora_name_or_path "${RETRIEVER_DIR}" \
    --bf16 --pooling last --padding_side left \
    --append_eos_token --normalize \
    --per_device_eval_batch_size 128 \
    "${extra[@]}" \
    --dataset_name json --dataset_path "$1" \
    --encode_output_path "$2"
}

echo "[1] corpus"
encode "${DATA_DIR}/corpus.jsonl" "${OUT}/corpus.pkl" passage

for split in ${splits}; do
  echo "[2] ${PFX}${split} queries"
  encode "${QUERY_DIR}/${split}.jsonl" "${OUT}/${PFX}${split}_queries.pkl" query
  if [ ! -f "${OUT}/${PFX}${split}_rank.trec" ]; then
    python -m tevatron.retriever.driver.search \
      --query_reps "${OUT}/${PFX}${split}_queries.pkl" \
      --passage_reps "${OUT}/corpus.pkl" \
      --depth ${depth} --batch_size 256 \
      --save_text --save_ranking_to "${OUT}/${PFX}${split}_rank.txt"
    python -m tevatron.utils.format.convert_result_to_trec \
      --input "${OUT}/${PFX}${split}_rank.txt" --output "${OUT}/${PFX}${split}_rank.trec" --remove_query
  fi
done
echo "DONE"
