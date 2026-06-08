#!/bin/bash

# Chỉ giữ path chứa libcuda.so (driver), bỏ CUDA 11.8 toolkit để tránh xung đột
# với CUDA 12.6 runtime mà torch tự bundle (torch 2.7.1+cu126).
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"

# HuggingFace cache — chuyển sang Data1 để tránh đầy ổ cài HĐH.
# export HF_HOME="/media/administrator/Data1/hf_cache"

if [ -z "$1" ]; then
  echo "Lỗi: Bạn chưa nhập tên dataset!"
  echo "Cách sử dụng: ./train.sh <dataset> [--model MODEL] [--group-size N] [--epochs N] [--query-max-len N] [--data-variant TAG] [--tag TAG]"
  echo ""
  echo "  dataset             : beauty | sports | ml-1m | steam  (bắt buộc)"
  echo "  --model MODEL       : HuggingFace model ID (mặc định: Qwen/Qwen3-Embedding-0.6B)"
  echo "  --group-size N      : số passages/query = 1 positive + N-1 negatives (mặc định: 8)"
  echo "  --epochs N          : số training epochs (mặc định: 3)"
  echo "  --query-max-len N   : số token tối đa của query (mặc định: 128; tăng khi dùng context_size lớn)"
  echo "  --data-variant TAG  : đọc data từ tevatron/<dataset>-<TAG>/ (ví dụ: cs5, w3)"
  echo "  --tag TAG           : hậu tố output dir để phân biệt experiment (mặc định: lấy từ --data-variant)"
  echo ""
  echo "Ví dụ:"
  echo "  ./train.sh beauty"
  echo "  ./train.sh beauty --group-size 32 --tag gs32"
  echo "  ./train.sh beauty --epochs 5 --tag ep5"
  echo "  ./train.sh beauty --data-variant cs5 --query-max-len 160"
  echo "  ./train.sh beauty --data-variant cs10 --query-max-len 256"
  echo "  ./train.sh beauty --model Qwen/Qwen3-Embedding-4B --tag 4b"
  echo "  ./train.sh beauty --data-variant v2 --tag v2 --v2-format --query-max-len 256"
  echo "  ./train.sh beauty --data-variant v2-aug --tag v2-aug --v2-format --query-max-len 256"
  exit 1
fi

dataset=$1
shift

# Defaults
model="Qwen/Qwen3-Embedding-0.6B"
train_group_size=8
num_epochs=3
query_max_len=128
data_variant=""
tag=""
v2_format=false

# Parse named flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          model="$2";            shift 2 ;;
    --group-size)     train_group_size="$2"; shift 2 ;;
    --epochs)         num_epochs="$2";       shift 2 ;;
    --query-max-len)  query_max_len="$2";    shift 2 ;;
    --data-variant)   data_variant="$2";     shift 2 ;;
    --tag)            tag="$2";              shift 2 ;;
    --v2-format)      v2_format=true;        shift ;;
    *) echo "Lỗi: Tham số không hợp lệ '$1'"; echo "Chạy ./train.sh để xem hướng dẫn."; exit 1 ;;
  esac
done

# Output tag: --tag có ưu tiên, nếu không có thì fallback về --data-variant
tag="${tag:-${data_variant}}"

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
    echo "Epochs           : ${num_epochs}"
    echo "Query max len    : ${query_max_len}"
    [ -n "${data_variant}" ] && echo "Data variant     : ${data_variant}"
    [ -n "${tag}" ]          && echo "Output tag       : ${tag}"
    ;;
  *)
    echo "Lỗi: Dataset '${dataset}' không hợp lệ!"
    echo "Vui lòng chỉ nhập một trong các dataset sau: beauty, sports, ml-1m, steam."
    exit 1
    ;;
esac

BASE_MODEL="${model}"
MODEL_TAG=$(basename "${model}" | tr '[:upper:]' '[:lower:]')
[ -n "${tag}" ] && MODEL_TAG="${MODEL_TAG}-${tag}"
LORA_DIR="./output/${dataset}/${MODEL_TAG}"
DATA_DIR="dataset/dataset/tevatron/${dataset}${data_variant:+-${data_variant}}"
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

# Training hyperparameters — định nghĩa một lần, dùng chung cho deepspeed và train_config.json
# (num_epochs, query_max_len đã được parse từ CLI flags ở trên)
learning_rate="1e-4"

# v2 format: instruction-based query → không cần "Query: "/"Passage: " prefix
if [ "${v2_format}" = "true" ]; then
  query_prefix=""
  passage_prefix=""
  [ -n "${tag}" ] && echo "  Format v2    : on (empty query/passage prefix)"
else
  query_prefix="Query: "
  passage_prefix="Passage: "
fi
save_steps=1000
passage_max_len=196

# Lưu config training để show_results.py tự sinh nhãn và mô tả experiment
mkdir -p "${LORA_DIR}"
cat > "${LORA_DIR}/train_config.json" << JSON
{
  "dataset": "${dataset}",
  "model": "${model}",
  "data_variant": "${data_variant}",
  "tag": "${tag}",
  "train_group_size": ${train_group_size},
  "per_device_batch": ${per_device_batch},
  "gradient_accumulation": ${grad_accum},
  "learning_rate": "${learning_rate}",
  "epochs": ${num_epochs},
  "save_steps": ${save_steps},
  "query_max_len": ${query_max_len},
  "passage_max_len": ${passage_max_len},
  "timestamp": "$(date -Iseconds)"
}
JSON
echo "  Config saved → ${LORA_DIR}/train_config.json"
echo ""

deepspeed --include localhost:0 --master_port 60000 \
  --module tevatron.retriever.driver.train \
  --deepspeed ds_config.json \
  --output_dir ${LORA_DIR} \
  --model_name_or_path ${BASE_MODEL} \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --dataset_path ${TRAIN_PATH} \
  --corpus_path ${CORPUS_PATH} \
  --query_prefix "${query_prefix}" \
  --passage_prefix "${passage_prefix}" \
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
  --learning_rate ${learning_rate} \
  --query_max_len ${query_max_len} \
  --passage_max_len ${passage_max_len} \
  --num_train_epochs ${num_epochs} \
  --logging_steps 100 \
  --save_steps ${save_steps}

echo ""
echo "✓ Model đã lưu tại: ${LORA_DIR}"
