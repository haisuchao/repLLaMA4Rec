#!/bin/bash

if [ -z "$1" ]; then
  echo "Lỗi: Bạn chưa nhập tên dataset!"
  echo "Cách sử dụng: ./encode_corpus.sh <dataset> [model]"
  echo "  dataset : beauty | sports | ml-1m | steam"
  echo "  model   : HuggingFace model ID (mặc định: Qwen/Qwen3-Embedding-0.6B)"
  echo "Ví dụ: ./encode_corpus.sh beauty"
  echo "Ví dụ: ./encode_corpus.sh beauty meta-llama/Llama-2-7b-hf"
  exit 1
fi

dataset=$1
model=${2:-"Qwen/Qwen3-Embedding-0.6B"}

case "${model}" in
  *Llama-3.2-3B*|*Llama-3-8B*|*Llama-3.1-8B*|*Llama-3.2-1B*)
    eval_batch=16
    ;;
  *)
    eval_batch=64
    ;;
esac

case "$dataset" in
  beauty|sports|ml-1m|steam)
    echo "Dataset : ${dataset}"
    echo "Model   : ${model}"
    ;;
  *)
    echo "Lỗi: Dataset '${dataset}' không hợp lệ!"
    exit 1
    ;;
esac

BASE_MODEL="${model}"
MODEL_TAG=$(basename "${model}" | tr '[:upper:]' '[:lower:]')
LORA_DIR="./output/${dataset}/${MODEL_TAG}"
CORPUS_PATH="dataset/dataset/tevatron/${dataset}/corpus.jsonl"
EMB_OUT="./output/${dataset}/${MODEL_TAG}/embeddings/corpus/corpus.pkl"

if [ ! -f "${CORPUS_PATH}" ]; then
  echo "Lỗi: Không tìm thấy corpus tại ${CORPUS_PATH}"
  exit 1
fi

if [ ! -d "${LORA_DIR}" ]; then
  echo "Lỗi: Không tìm thấy LoRA checkpoint tại ${LORA_DIR}"
  echo "Hãy chạy train.sh trước."
  exit 1
fi

mkdir -p "./output/${dataset}/${MODEL_TAG}/embeddings/corpus"

CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --output_dir temp \
  --model_name_or_path ${BASE_MODEL} \
  --lora \
  --lora_name_or_path ${LORA_DIR} \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --bf16 \
  --pooling last \
  --padding_side left \
  --append_eos_token \
  --normalize \
  --per_device_eval_batch_size ${eval_batch} \
  --passage_max_len 196 \
  --dataset_name json \
  --dataset_path ${CORPUS_PATH} \
  --encode_output_path ${EMB_OUT}

echo ""
echo "✓ Corpus embeddings đã lưu tại: ${EMB_OUT}"
