#!/bin/bash

if [ -z "$1" ]; then
  echo "Lỗi: Bạn chưa nhập tên dataset!"
  echo "Cách sử dụng: ./search.sh <dataset> [model] [split]"
  echo "  dataset : beauty | sports | ml-1m | steam"
  echo "  model   : HuggingFace model ID (mặc định: Qwen/Qwen3-Embedding-0.6B)"
  echo "  split   : train | valid | test  (mặc định: test)"
  echo "Ví dụ: ./search.sh beauty"
  echo "Ví dụ: ./search.sh beauty meta-llama/Llama-2-7b-hf valid"
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

MODEL_TAG=$(basename "${model}" | tr '[:upper:]' '[:lower:]')
CORPUS_EMB="./output/${dataset}/${MODEL_TAG}/embeddings/corpus/corpus.pkl"
QUERY_EMB="./output/${dataset}/${MODEL_TAG}/embeddings/queries/${split}.pkl"
RESULTS_DIR="./output/${dataset}/${MODEL_TAG}/embeddings/results"
RANK_OUT="${RESULTS_DIR}/${split}_rank.txt"

if [ ! -f "${CORPUS_EMB}" ]; then
  echo "Lỗi: Không tìm thấy corpus embeddings tại ${CORPUS_EMB}"
  echo "Hãy chạy encode_corpus.sh ${dataset} ${model} trước."
  exit 1
fi

if [ ! -f "${QUERY_EMB}" ]; then
  echo "Lỗi: Không tìm thấy query embeddings tại ${QUERY_EMB}"
  echo "Hãy chạy encode_queries.sh ${dataset} ${model} ${split} trước."
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

python -m tevatron.retriever.driver.search \
  --query_reps ${QUERY_EMB} \
  --passage_reps ${CORPUS_EMB} \
  --depth 100 \
  --batch_size 128 \
  --save_text \
  --save_ranking_to ${RANK_OUT}

echo ""
echo "✓ Kết quả search đã lưu tại: ${RANK_OUT}"
