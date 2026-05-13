#!/bin/bash

# Chỉ giữ path chứa libcuda.so (driver), bỏ CUDA 11.8 toolkit để tránh xung đột
# với CUDA 12.6 runtime mà torch tự bundle (torch 2.7.1+cu126).
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"

# HuggingFace cache — chuyển sang Data1 để tránh đầy ổ cài HĐH.
# export HF_HOME="/media/administrator/Data1/hf_cache"

if [ -z "$1" ]; then
  echo "Lỗi: Bạn chưa nhập tên dataset!"
  echo "Cách sử dụng: ./train.sh <dataset> [model] [train_group_size] [variant]"
  echo "  dataset          : beauty | sports | ml-1m | steam"
  echo "  model            : HuggingFace model ID (mặc định: Qwen/Qwen3-Embedding-0.6B)"
  echo "  train_group_size : số passages/query = 1 positive + N negatives (mặc định: 8)"
  echo "  variant          : hậu tố cho data dir và model output dir (mặc định: rỗng)"
  echo "                     ví dụ: 'aug' → dùng data từ beauty-aug/, lưu vào qwen3-...-aug/"
  echo "Ví dụ: ./train.sh beauty"
  echo "Ví dụ: ./train.sh beauty Qwen/Qwen3-Embedding-0.6B 16"
  echo "Ví dụ: ./train.sh beauty Qwen/Qwen3-Embedding-0.6B 8 aug"
  exit 1
fi

dataset=$1
model=${2:-"Qwen/Qwen3-Embedding-0.6B"}
train_group_size=${3:-8}
variant=${4:-""}

# Điều chỉnh batch size theo kích thước model để tránh OOM
case "${model}" in
  *Qwen3-Embedding-4B*|*Qwen2.5-3B*|*Llama-3.2-3B*|*Llama-3-8B*|*Llama-3.1-8B*|*Llama-3.2-1B*)
    per_device_batch=1
    grad_accum=32
    eval_batch=16
    ;;
  *)
    per_device_batch=4
    grad_accum=8
    eval_batch=64
    ;;
esac

case "$dataset" in
  beauty|sports|ml-1m|steam)
    echo "Dataset          : ${dataset}"
    echo "Model            : ${model}"
    echo "Train group size : ${train_group_size} (1 positive + $((train_group_size - 1)) negatives)"
    ;;
  *)
    echo "Lỗi: Dataset '${dataset}' không hợp lệ!"
    echo "Vui lòng chỉ nhập một trong các dataset sau: beauty, sports, ml-1m, steam."
    exit 1
    ;;
esac

BASE_MODEL="${model}"
MODEL_TAG=$(basename "${model}" | tr '[:upper:]' '[:lower:]')
[ -n "${variant}" ] && MODEL_TAG="${MODEL_TAG}-${variant}"
LORA_DIR="./output/${dataset}/${MODEL_TAG}"
DATA_DIR="dataset/dataset/tevatron/${dataset}${variant:+-${variant}}"
TRAIN_PATH="${DATA_DIR}/train.jsonl"
CORPUS_PATH="${DATA_DIR}/corpus.jsonl"

if [ ! -f "${TRAIN_PATH}" ]; then
  echo "Lỗi: Không tìm thấy train data tại ${TRAIN_PATH}"
  exit 1
fi

if [ ! -f "${CORPUS_PATH}" ]; then
  echo "Lỗi: Không tìm thấy corpus tại ${CORPUS_PATH}"
  exit 1
fi

deepspeed --include localhost:0 --master_port 60000 \
  --module tevatron.retriever.driver.train \
  --deepspeed ds_config.json \
  --output_dir ${LORA_DIR} \
  --model_name_or_path ${BASE_MODEL} \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --dataset_path ${TRAIN_PATH} \
  --corpus_path ${CORPUS_PATH} \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --bf16 \
  --pooling last \
  --padding_side left \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size ${per_device_batch} \
  --gradient_accumulation_steps ${grad_accum} \
  --gradient_checkpointing \
  --train_group_size ${train_group_size} \
  --learning_rate 1e-4 \
  --query_max_len 128 \
  --passage_max_len 196 \
  --num_train_epochs 3 \
  --logging_steps 100 \
  --save_steps 2000

echo ""
echo "✓ Model đã lưu tại: ${LORA_DIR}"
