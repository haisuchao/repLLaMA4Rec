#!/bin/bash
#
# eval_zeroshot.sh — Đánh giá base model KHÔNG fine-tune (zero-shot).
#
# Pipeline đầy đủ: encode corpus → encode queries → FAISS search → evaluate.
# Corpus được cache: nếu đã encode rồi thì bỏ qua, chỉ encode queries mới.
#
# Cách sử dụng: ./eval_zeroshot.sh <dataset> [model] [split]
#   dataset : beauty | sports | ml-1m | steam
#   model   : HuggingFace model ID (mặc định: Qwen/Qwen3-Embedding-0.6B)
#   split   : valid | test  (mặc định: test)
#
# Output: output/<dataset>/<model_tag>-zeroshot/embeddings/
#   Tách biệt hoàn toàn với thư mục của fine-tuned model.
#
# Ví dụ:
#   ./eval_zeroshot.sh beauty
#   ./eval_zeroshot.sh beauty Qwen/Qwen3-Embedding-0.6B test
#   ./eval_zeroshot.sh beauty Qwen/Qwen3-Embedding-0.6B valid

set -e

# Driver mới (575+) đặt libcuda.so tại đây nhưng không có trong LD_LIBRARY_PATH mặc định
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# ── Kiểm tra tham số ─────────────────────────────────────────────────────────

if [ -z "$1" ]; then
  echo "Lỗi: Bạn chưa nhập tên dataset!"
  echo "Cách sử dụng: ./eval_zeroshot.sh <dataset> [model] [split]"
  echo "  dataset : beauty | sports | ml-1m | steam"
  echo "  model   : HuggingFace model ID (mặc định: Qwen/Qwen3-Embedding-0.6B)"
  echo "  split   : valid | test  (mặc định: test)"
  echo "Ví dụ: ./eval_zeroshot.sh beauty"
  exit 1
fi

dataset=$1
model=${2:-"Qwen/Qwen3-Embedding-0.6B"}
split=${3:-test}

case "$dataset" in
  beauty|sports|ml-1m|steam) ;;
  *)
    echo "Lỗi: Dataset '${dataset}' không hợp lệ!"
    echo "Chọn trong: beauty, sports, ml-1m, steam"
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

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_TAG=$(basename "${model}" | tr '[:upper:]' '[:lower:]')-zeroshot
BASE_DIR="./output/${dataset}/${MODEL_TAG}"
CORPUS_PATH="dataset/dataset/tevatron/${dataset}/corpus.jsonl"
QUERY_PATH="dataset/dataset/tevatron/${dataset}/${split}.jsonl"
CORPUS_EMB="${BASE_DIR}/embeddings/corpus/corpus.pkl"
QUERY_EMB="${BASE_DIR}/embeddings/queries/${split}.pkl"
RESULTS_DIR="${BASE_DIR}/embeddings/results"
RANK_OUT="${RESULTS_DIR}/${split}_rank.txt"
RANK_TREC="${RESULTS_DIR}/${split}_rank.trec"
QRELS="${RESULTS_DIR}/${split}_qrels.txt"
QRELS_CLEAN="${RESULTS_DIR}/${split}_qrels_clean.txt"
RANK_CLEAN="${RESULTS_DIR}/${split}_rank_clean.trec"
EVAL_OUT="${RESULTS_DIR}/eval_${split}.txt"

echo "════════════════════════════════════════════════"
echo "  Zero-shot Evaluation"
echo "════════════════════════════════════════════════"
echo "  Dataset : ${dataset}"
echo "  Model   : ${model} (không LoRA)"
echo "  Split   : ${split}"
echo "  Output  : ${BASE_DIR}"
echo "════════════════════════════════════════════════"
echo ""

# ── Kiểm tra dữ liệu đầu vào ─────────────────────────────────────────────────

if [ ! -f "${CORPUS_PATH}" ]; then
  echo "Lỗi: Không tìm thấy corpus tại ${CORPUS_PATH}"
  echo "Hãy chạy: cd dataset && python run_all.py ${dataset}"
  exit 1
fi

if [ ! -f "${QUERY_PATH}" ]; then
  echo "Lỗi: Không tìm thấy query file tại ${QUERY_PATH}"
  exit 1
fi

mkdir -p "${BASE_DIR}/embeddings/corpus"
mkdir -p "${BASE_DIR}/embeddings/queries"
mkdir -p "${RESULTS_DIR}"

# ── Kiểm tra CUDA ─────────────────────────────────────────────────────────────

python -c "
import torch, sys
if not torch.cuda.is_available():
    print('Lỗi: torch.cuda.is_available() = False')
    print('Hãy chắc chắn đã chạy: source tevatron-env/bin/activate')
    sys.exit(1)
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# ── Bước 1: Encode corpus (cache — bỏ qua nếu đã tồn tại) ───────────────────

if [ -f "${CORPUS_EMB}" ]; then
  echo "── Bước 1: Corpus embeddings đã tồn tại, bỏ qua."
  echo "   ${CORPUS_EMB}"
else
  echo "── Bước 1: Encode corpus..."
  CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
    --output_dir temp \
    --model_name_or_path ${model} \
    --query_prefix "Query: " \
    --passage_prefix "Passage: " \
    --bf16 \
    --pooling last \
    --padding_side left \
    --append_eos_token \
    --normalize \
    --per_device_eval_batch_size 64 \
    --passage_max_len 196 \
    --dataset_name json \
    --dataset_path ${CORPUS_PATH} \
    --encode_output_path ${CORPUS_EMB}
  echo "   ✓ Corpus embeddings: ${CORPUS_EMB}"
fi
echo ""

# ── Bước 2: Encode queries ───────────────────────────────────────────────────

echo "── Bước 2: Encode queries (${split})..."
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --output_dir temp \
  --model_name_or_path ${model} \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
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
  --encode_output_path ${QUERY_EMB}
echo "   ✓ Query embeddings: ${QUERY_EMB}"
echo ""

# ── Bước 3: FAISS search ─────────────────────────────────────────────────────

echo "── Bước 3: FAISS search..."
python -m tevatron.retriever.driver.search \
  --query_reps ${QUERY_EMB} \
  --passage_reps ${CORPUS_EMB} \
  --depth 100 \
  --batch_size 128 \
  --save_text \
  --save_ranking_to ${RANK_OUT}
echo "   ✓ Rank results: ${RANK_OUT}"
echo ""

# ── Bước 4: Chuyển sang TREC format ─────────────────────────────────────────

echo "── Bước 4: Chuyển sang TREC format..."
python -m tevatron.utils.format.convert_result_to_trec \
  --input ${RANK_OUT} \
  --output ${RANK_TREC} \
  --remove_query

# ── Bước 5: Tạo qrels ────────────────────────────────────────────────────────

echo "── Bước 5: Tạo qrels..."
python << EOF
import json
with open('${QUERY_PATH}') as f_in, open('${QRELS}', 'w') as f_out:
    for line in f_in:
        d = json.loads(line)
        qid = d['query_id']
        for p in d.get('positive_passages', []):
            f_out.write(f"{qid}\t0\t{p['docid']}\t1\n")
print("   qrels OK")
EOF

# ── Bước 6: Loại bỏ duplicate ────────────────────────────────────────────────

echo "── Bước 6: Loại bỏ duplicate..."
python << EOF
def remove_duplicates(src, dst):
    seen, kept, skipped = set(), 0, 0
    with open(src) as f_in, open(dst, 'w') as f_out:
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
    print(f"   {src.split('/')[-1]}: kept={kept}, skipped={skipped}")

remove_duplicates('${QRELS}', '${QRELS_CLEAN}')
remove_duplicates('${RANK_TREC}', '${RANK_CLEAN}')
EOF

# ── Bước 7: Tính metrics ─────────────────────────────────────────────────────

echo "── Bước 7: Tính metrics..."
echo ""
python compute_metrics.py \
  --run   ${RANK_CLEAN} \
  --qrels ${QRELS_CLEAN} \
  --ks    5,10,20 \
  --out   ${EVAL_OUT}

cat ${EVAL_OUT}
echo ""
echo "════════════════════════════════════════════════"
echo "✓ Hoàn tất! Kết quả lưu tại: ${EVAL_OUT}"
echo "════════════════════════════════════════════════"
