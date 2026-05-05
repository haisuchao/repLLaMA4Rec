#!/bin/bash

if [ -z "$1" ]; then
  echo "Lỗi: Bạn chưa nhập tên dataset!"
  echo "Cách sử dụng: ./evaluate.sh <dataset> [model] [split]"
  echo "  dataset : beauty | sports | ml-1m | steam"
  echo "  model   : HuggingFace model ID (mặc định: Qwen/Qwen3-Embedding-0.6B)"
  echo "  split   : valid | test  (mặc định: test)"
  echo "Ví dụ: ./evaluate.sh beauty"
  echo "Ví dụ: ./evaluate.sh beauty meta-llama/Llama-2-7b-hf valid"
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
  valid|test) ;;
  *)
    echo "Lỗi: Split '${split}' không hợp lệ! Chọn: valid, test"
    exit 1
    ;;
esac

MODEL_TAG=$(basename "${model}" | tr '[:upper:]' '[:lower:]')
RESULTS_DIR="./output/${dataset}/${MODEL_TAG}/embeddings/results"
QUERY_PATH="dataset/dataset/tevatron/${dataset}/${split}.jsonl"
RANK_TXT="${RESULTS_DIR}/${split}_rank.txt"
RANK_TREC="${RESULTS_DIR}/${split}_rank.trec"
QRELS="${RESULTS_DIR}/${split}_qrels.txt"
QRELS_CLEAN="${RESULTS_DIR}/${split}_qrels_clean.txt"
RANK_CLEAN="${RESULTS_DIR}/${split}_rank_clean.trec"
EVAL_OUT="${RESULTS_DIR}/eval_${split}.txt"

if [ ! -f "${RANK_TXT}" ]; then
  echo "Lỗi: Không tìm thấy rank results tại ${RANK_TXT}"
  echo "Hãy chạy search.sh ${dataset} ${model} ${split} trước."
  exit 1
fi

# Bước 1: Chuyển sang định dạng TREC
echo "── Bước 1: Chuyển sang TREC format..."
python -m tevatron.utils.format.convert_result_to_trec \
  --input ${RANK_TXT} \
  --output ${RANK_TREC} \
  --remove_query

# Bước 2: Tạo qrels từ jsonl
echo "── Bước 2: Tạo qrels từ ${QUERY_PATH}..."
python << EOF
import json

with open('${QUERY_PATH}') as f_in, \
     open('${QRELS}', 'w') as f_out:
    for line in f_in:
        d = json.loads(line)
        qid = d['query_id']
        for p in d.get('positive_passages', []):
            f_out.write(f"{qid}\t0\t{p['docid']}\t1\n")

print("  qrels.txt OK: ${QRELS}")
EOF

# Bước 3: Loại bỏ duplicate (query_id, doc_id) trong cả qrels lẫn rank
echo "── Bước 3: Loại bỏ duplicate..."
python << EOF
def remove_duplicates(input_file, output_file):
    seen = set()
    kept, skipped = 0, 0
    with open(input_file) as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            key = f"{parts[0]}_{parts[2]}"
            if key not in seen:
                seen.add(key)
                f_out.write(line)
                kept += 1
            else:
                skipped += 1
    print(f"  {input_file}: kept={kept}, skipped={skipped}")

remove_duplicates('${QRELS}', '${QRELS_CLEAN}')
remove_duplicates('${RANK_TREC}', '${RANK_CLEAN}')
EOF

# Bước 4: Tính NDCG@K, HR@K, MRR@K
echo "── Bước 4: Tính metrics..."
python compute_metrics.py \
  --run   ${RANK_CLEAN} \
  --qrels ${QRELS_CLEAN} \
  --ks    5,10,20 \
  --out   ${EVAL_OUT} \
  | tee ${EVAL_OUT}

echo ""
echo "✓ Kết quả đánh giá đã lưu tại: ${EVAL_OUT}"
