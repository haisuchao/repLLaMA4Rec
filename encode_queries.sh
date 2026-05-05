#!/bin/bash

if [ -z "$1" ]; then
  echo "Lỗi: Bạn chưa nhập tên dataset!"
  echo "Cách sử dụng: ./encode_queries.sh <dataset> [model] [split]"
  echo "  dataset : beauty | sports | ml-1m | steam"
  echo "  model   : HuggingFace model ID (mặc định: Qwen/Qwen3-Embedding-0.6B)"
  echo "  split   : train | valid | test  (mặc định: test)"
  echo "Ví dụ: ./encode_queries.sh beauty"
  echo "Ví dụ: ./encode_queries.sh beauty meta-llama/Llama-2-7b-hf valid"
  exit 1
fi

dataset=$1
model=${2:-"Qwen/Qwen3-Embedding-0.6B"}
split=${3:-test}

case "$dataset" in
  beauty|sports|ml-1m|steam)
    echo "Dataset : ${dataset}"
    echo "Model   : ${model}"
    echo "Split   : ${split}"
    ;;
  *)
    echo "Lỗi: Dataset '${dataset}' không hợp lệ!"
    exit 1
    ;;
esac

case "$split" in
  train|valid|test) ;;
  *)
    echo "Lỗi: Split '${split}' không hợp lệ! Chọn: train, valid, test"
    exit 1
    ;;
esac

BASE_MODEL="${model}"
MODEL_TAG=$(basename "${model}" | tr '[:upper:]' '[:lower:]')
LORA_DIR="./output/${dataset}/${MODEL_TAG}"
QUERY_PATH="dataset/dataset/tevatron/${dataset}/${split}.jsonl"
EMB_OUT="./output/${dataset}/${MODEL_TAG}/embeddings/queries/${split}.pkl"

if [ ! -f "${QUERY_PATH}" ]; then
  echo "Lỗi: Không tìm thấy query file tại ${QUERY_PATH}"
  exit 1
fi

if [ ! -d "${LORA_DIR}" ]; then
  echo "Lỗi: Không tìm thấy LoRA checkpoint tại ${LORA_DIR}"
  echo "Hãy chạy train.sh trước."
  exit 1
fi

mkdir -p "./output/${dataset}/${MODEL_TAG}/embeddings/queries"

CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --output_dir temp \
  --model_name_or_path ${BASE_MODEL} \
  --lora \
  --lora_name_or_path ${LORA_DIR} \
  --query_prefix "Query: " \
  --bf16 \
  --pooling last \
  --padding_side left \
  --append_eos_token \
  --normalize \
  --per_device_eval_batch_size 64 \
  --query_max_len 128 \
  --dataset_name json \
  --dataset_path ${QUERY_PATH} \
  --encode_is_query \
  --encode_output_path ${EMB_OUT}

echo ""
echo "✓ Query embeddings đã lưu tại: ${EMB_OUT}"
