#!/bin/bash
#
# rerank.sh — Rerank top-K candidates của retriever bằng cross-encoder đã train.
#
# Cách sử dụng:
#   ./rerank.sh <dataset> [--model MODEL] [--tag RTAG] [--split SPLIT]
#               [--retriever-trec FILE] [--reranker-ckpt DIR]
#
#   <dataset>           : beauty | sports | ml-1m
#   --model MODEL       : base model (mặc định: Qwen/Qwen3-Embedding-0.6B)
#   --tag RTAG          : tag của retriever (phải khớp với train_reranker.sh --tag)
#   --split SPLIT       : valid | test (mặc định: test)
#   --retriever-trec F  : override đường dẫn retriever trec file
#   --reranker-ckpt DIR : override reranker checkpoint dir (mặc định: final model)
#
# Điều kiện tiên quyết:
#   - eval.sh đã chạy để có retriever trec file
#   - train_reranker.sh đã chạy để có reranker weights
#
# Ví dụ:
#   ./rerank.sh beauty
#   ./rerank.sh beauty --split valid
#   ./rerank.sh beauty --tag aug-5
#   ./rerank.sh beauty --reranker-ckpt output/beauty/qwen3-embedding-0.6b-reranker/checkpoint-500

set -e
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"

# ── Parse args ────────────────────────────────────────────────────────────────

if [ -z "$1" ]; then
  echo "Lỗi: Cần nhập dataset!"
  echo "Dùng: ./rerank.sh <dataset> [--model MODEL] [--tag RTAG] [--split SPLIT]"
  exit 1
fi

dataset=$1; shift

model="Qwen/Qwen3-Embedding-0.6B"
retriever_tag=""
split="test"
retriever_trec=""
reranker_ckpt=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          model="$2";          shift 2 ;;
    --tag)            retriever_tag="$2";  shift 2 ;;
    --split)          split="$2";          shift 2 ;;
    --retriever-trec) retriever_trec="$2"; shift 2 ;;
    --reranker-ckpt)  reranker_ckpt="$2";  shift 2 ;;
    *) echo "Lỗi: Tham số không hợp lệ '$1'"; exit 1 ;;
  esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_TAG=$(basename "${model}" | tr '[:upper:]' '[:lower:]')
[ -n "${retriever_tag}" ] && MODEL_TAG="${MODEL_TAG}-${retriever_tag}"

RETRIEVER_DIR="./output/${dataset}/${MODEL_TAG}"
RETRIEVER_RESULTS="${RETRIEVER_DIR}/embeddings/results"
RERANKER_DIR="${RETRIEVER_DIR}-reranker"
INFER_DIR="${RERANKER_DIR}/inference"
DATA_DIR="./dataset/dataset/tevatron/${dataset}"

mkdir -p "${INFER_DIR}"

# ── Tìm retriever trec file ───────────────────────────────────────────────────

if [ -z "${retriever_trec}" ]; then
  # Ưu tiên: đọc best checkpoint label từ eval_<split>_best.txt
  BEST_TXT="${RETRIEVER_RESULTS}/eval_${split}_best.txt"
  if [ -f "${BEST_TXT}" ]; then
    best_label=$(grep "^Best checkpoint" "${BEST_TXT}" | awk '{print $NF}')
    candidate="${RETRIEVER_RESULTS}/${split}_${best_label}_rank_clean.trec"
    [ -f "${candidate}" ] && retriever_trec="${candidate}"
  fi

  # Fallback: lấy file _rank_clean.trec bất kỳ cho split này
  if [ -z "${retriever_trec}" ]; then
    retriever_trec=$(ls "${RETRIEVER_RESULTS}/${split}_"*"_rank_clean.trec" 2>/dev/null | head -1)
  fi

  if [ -z "${retriever_trec}" ]; then
    echo "Lỗi: Không tìm thấy retriever trec file cho split=${split} tại ${RETRIEVER_RESULTS}"
    echo "Hãy chạy: ./eval.sh ${dataset} --split ${split}"
    exit 1
  fi
fi

# ── Tìm reranker checkpoint ───────────────────────────────────────────────────

if [ -z "${reranker_ckpt}" ]; then
  reranker_ckpt="${RERANKER_DIR}"
fi
if [ ! -f "${reranker_ckpt}/adapter_model.safetensors" ] && \
   [ ! -f "${reranker_ckpt}/adapter_model.bin" ]; then
  echo "Lỗi: Không tìm thấy reranker weights tại ${reranker_ckpt}"
  echo "Hãy chạy: ./train_reranker.sh ${dataset}$([ -n "${retriever_tag}" ] && echo " --tag ${retriever_tag}")"
  exit 1
fi

# Đếm depth từ trec file
n_cands=$(awk '{print $1}' "${retriever_trec}" | sort | uniq -c | awk '{print $1}' | sort -rn | head -1)

echo "════════════════════════════════════════════════"
echo "  Rerank"
echo "════════════════════════════════════════════════"
echo "  Dataset        : ${dataset}"
echo "  Split          : ${split}"
echo "  Retriever trec : $(basename "${retriever_trec}")"
echo "  Reranker       : ${reranker_ckpt}"
echo "  Depth          : ${n_cands:-100} candidates/query"
echo "════════════════════════════════════════════════"
echo ""

PAIRS_JSONL="${INFER_DIR}/${split}_pairs.jsonl"
RERANKED_TXT="${INFER_DIR}/${split}_reranked.txt"
RERANKED_TREC="${INFER_DIR}/${split}_reranked.trec"
EVAL_OUT="${INFER_DIR}/eval_${split}_reranked.txt"

# ── Step 1: Prepare inference pairs ──────────────────────────────────────────

if [ ! -f "${PAIRS_JSONL}" ]; then
  echo "[1/3] Chuẩn bị inference pairs..."
  python prepare_rerank_data.py \
    --mode  infer \
    --queries "${DATA_DIR}/${split}.jsonl" \
    --corpus  "${DATA_DIR}/corpus.jsonl" \
    --trec    "${retriever_trec}" \
    --output  "${PAIRS_JSONL}"
else
  echo "[1/3] Inference pairs đã có — bỏ qua."
  echo "      (Xóa ${PAIRS_JSONL} nếu muốn tạo lại)"
fi

# ── Step 2: Rerank ────────────────────────────────────────────────────────────

echo "[2/3] Reranking..."
CUDA_VISIBLE_DEVICES=0 python -m tevatron.reranker.driver.rerank \
  --model_name_or_path "${model}" \
  --lora_name_or_path  "${reranker_ckpt}" \
  --dataset_name json \
  --dataset_path "${PAIRS_JSONL}" \
  --dataset_split train \
  --rerank_output_path "${RERANKED_TXT}" \
  --rerank_max_len 256 \
  --append_eos_token \
  --per_device_eval_batch_size 32 \
  --output_dir temp \
  --bf16

# ── Step 3: Convert → TREC + Evaluate ────────────────────────────────────────

echo "[3/3] Evaluating..."

# Convert qid\tdocid\tscore → TREC format với rank
RERANKED_TXT="${RERANKED_TXT}" RERANKED_TREC="${RERANKED_TREC}" python3 - << 'PYEOF'
import os
from collections import defaultdict

inp  = os.environ['RERANKED_TXT']
outp = os.environ['RERANKED_TREC']

results = defaultdict(list)
with open(inp) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue
        qid, docid, score = parts
        results[qid].append((docid, float(score)))

n_total = 0
with open(outp, 'w') as f:
    for qid, docs in results.items():
        docs.sort(key=lambda x: x[1], reverse=True)
        for rank, (docid, score) in enumerate(docs, 1):
            f.write(f"{qid} Q0 {docid} {rank} {score:.6f} reranker\n")
            n_total += 1

print(f"  {n_total:,} entries → {outp}")
PYEOF

# Qrels: reuse từ eval.sh nếu có, không thì tạo mới
QRELS="${RETRIEVER_RESULTS}/${split}_qrels_clean.txt"
if [ ! -f "${QRELS}" ]; then
  QRELS="${INFER_DIR}/${split}_qrels_clean.txt"
  if [ ! -f "${QRELS}" ]; then
    QRELS="${QRELS}" SPLIT="${split}" DATA_DIR="${DATA_DIR}" python3 - << 'PYEOF'
import json, os
split    = os.environ['SPLIT']
data_dir = os.environ['DATA_DIR']
qrels    = os.environ['QRELS']
raw      = qrels.replace('_clean', '_raw')

with open(f'{data_dir}/{split}.jsonl') as fi, open(raw, 'w') as fo:
    for line in fi:
        d = json.loads(line)
        for p in d.get('positive_passages', []):
            fo.write(f"{d['query_id']}\t0\t{p['docid']}\t1\n")

seen = set()
with open(raw) as fi, open(qrels, 'w') as fo:
    for line in fi:
        parts = line.strip().split()
        if len(parts) < 3: continue
        k = parts[0] + '_' + parts[2]
        if k not in seen:
            seen.add(k); fo.write(line)
print(f"  qrels: {len(seen):,} entries")
PYEOF
  fi
fi

python compute_metrics.py \
  --run   "${RERANKED_TREC}" \
  --qrels "${QRELS}" \
  --ks    5,10,20 \
  --out   "${EVAL_OUT}"

echo ""
echo "════════════════════════════════════════════════"
echo "  Results — ${dataset} / reranker / ${split}"
echo "════════════════════════════════════════════════"
cat "${EVAL_OUT}"
echo ""
echo "✓ Kết quả lưu tại: ${EVAL_OUT}"

# ── So sánh Retriever vs Reranker ─────────────────────────────────────────────

RETRIEVER_EVAL="${RETRIEVER_RESULTS}/eval_${split}_best.txt"
if [ -f "${RETRIEVER_EVAL}" ]; then
  echo ""
  echo "── So sánh: Retriever vs Reranker ──────────────"
  RETRIEVER_EVAL="${RETRIEVER_EVAL}" RERANKER_EVAL="${EVAL_OUT}" python3 - << 'PYEOF'
import os

def read_metrics(path):
    m = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[1] == 'all':
                try:
                    m[parts[0]] = float(parts[2])
                except ValueError:
                    pass
    return m

r  = read_metrics(os.environ['RETRIEVER_EVAL'])
rr = read_metrics(os.environ['RERANKER_EVAL'])

print(f"{'Metric':<12}  {'Retriever':>10}  {'Reranker':>10}  {'Δ':>8}")
print("─" * 46)
for k in ['ndcg_5', 'hr_5', 'ndcg_10', 'hr_10', 'ndcg_20', 'hr_20', 'mrr_10']:
    rv  = r.get(k, 0.0)
    rrv = rr.get(k, 0.0)
    d   = rrv - rv
    arrow = ' ↑' if d > 0.0001 else (' ↓' if d < -0.0001 else '')
    print(f"{k:<12}  {rv:>10.4f}  {rrv:>10.4f}  {d:>+8.4f}{arrow}")
PYEOF
fi
