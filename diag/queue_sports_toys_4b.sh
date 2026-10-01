#!/bin/bash
# Hàng đợi 2026-09-29: pipeline reranker cho Sports + Toys (0.6B), rồi Beauty với reranker 4B.
# Mỗi dataset độc lập: dataset lỗi không chặn dataset sau. Log: log/queue_<job>.log
cd "$(dirname "$0")/.."
source tevatron-env/bin/activate

run() {  # $1 = tên job, còn lại = lệnh
  local name=$1; shift
  echo "[${name}] start $(date '+%F %H:%M')" >> log/queue.log
  if "$@" > "log/queue_${name}.log" 2>&1; then
    echo "[${name}] OK $(date '+%F %H:%M')" >> log/queue.log
  else
    echo "[${name}] FAILED (exit $?) $(date '+%F %H:%M')" >> log/queue.log
    return 1
  fi
}

run sports_train  bash train_reranker.sh sports --tag cs5-aug-gs32 && \
run sports_rerank bash rerank.sh sports --tag cs5-aug-gs32

run toys_train    bash train_reranker.sh toys --tag cs5-gs32-aug-0.6b && \
run toys_rerank   bash rerank.sh toys --tag cs5-gs32-aug-0.6b

run beauty4b_train  bash train_reranker.sh beauty --tag aug-5 --reranker-model Qwen/Qwen3-Embedding-4B && \
run beauty4b_rerank bash rerank.sh beauty --tag aug-5 --reranker-model Qwen/Qwen3-Embedding-4B

echo "QUEUE DONE $(date '+%F %H:%M')" >> log/queue.log
