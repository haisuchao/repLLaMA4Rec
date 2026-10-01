#!/bin/bash
# Thí nghiệm kiểm chứng negative sampling cho rankLLaMA reranker (xem diag/README.md).
# Idempotent: bỏ qua model đã train / scores đã có.
#
# Dùng: bash diag/run_rerank_exp.sh [VARIANT ...]     (mặc định: OLD R2 R1 R0s R0)
#   OLD = reranker cũ (output/beauty/qwen3-embedding-0.6b-aug-5-reranker), chỉ inference

set -e
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"

model="Qwen/Qwen3-Embedding-0.6B"
D="./output/beauty/qwen3-embedding-0.6b-aug-5/diag/rerank"
PAIRS="${D}/test_pairs_inpool.jsonl"
VALID_PAIRS="${D}/valid_pairs_sample.jsonl"
variants=("$@")
[ ${#variants[@]} -eq 0 ] && variants=(OLD R2 R1 R0s R0)

for v in "${variants[@]}"; do
  if [ "${v}" = "OLD" ]; then
    ckpt="./output/beauty/qwen3-embedding-0.6b-aug-5-reranker"
  else
    ckpt="${D}/models/${v}"
    # Hậu tố -gNN = group size NN trên cùng dữ liệu (vd. R1-g32 → R1_train.jsonl, group 32).
    # Giữ 32 chuỗi/forward (cùng VRAM) và effective batch 32 query/step.
    data="${v}"; gs=8
    if [[ "${v}" =~ ^(.+)-g([0-9]+)$ ]]; then data="${BASH_REMATCH[1]}"; gs="${BASH_REMATCH[2]}"; fi
    per_batch=$(( 32 / gs > 0 ? 32 / gs : 1 )); accum=$(( 32 / per_batch ))
    if [ ! -f "${ckpt}/adapter_model.safetensors" ]; then
      echo "[${v}] training $(date +%H:%M) (data=${data}, group=${gs}, batch ${per_batch}×accum ${accum})"
      CUDA_VISIBLE_DEVICES=0 python -m tevatron.reranker.driver.train \
        --model_name_or_path "${model}" \
        --lora --lora_r 16 --lora_alpha 64 --lora_dropout 0.05 \
        --lora_target_modules "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj" \
        --dataset_name json \
        --dataset_path "${D}/${data}_train.jsonl" \
        --dataset_split train \
        --train_group_size ${gs} \
        --rerank_max_len 256 \
        --append_eos_token \
        --output_dir "${ckpt}" \
        --bf16 \
        --num_train_epochs 3 \
        --per_device_train_batch_size ${per_batch} \
        --gradient_accumulation_steps ${accum} \
        --learning_rate 1e-4 \
        --warmup_steps 100 \
        --save_steps 500 \
        --logging_steps 50 \
        > "${D}/${v}_train.log" 2>&1
    fi
  fi

  # test: toàn bộ query có positive trong pool; valid: mẫu 2,000 query (chọn α khi fuse)
  for split in test valid; do
    [ "${split}" = "test" ] && pairs="${PAIRS}" || pairs="${VALID_PAIRS}"
    scores="${D}/${v}_${split}_scores.txt"
    if [ ! -f "${scores}" ]; then
      echo "[${v}] inference ${split} $(date +%H:%M)"
      CUDA_VISIBLE_DEVICES=0 python -m tevatron.reranker.driver.rerank \
        --model_name_or_path "${model}" \
        --lora_name_or_path "${ckpt}" \
        --dataset_name json \
        --dataset_path "${pairs}" \
        --dataset_split train \
        --rerank_output_path "${scores}.tmp" \
        --rerank_max_len 256 \
        --append_eos_token \
        --per_device_eval_batch_size 256 \
        --output_dir temp \
        --bf16 \
        > "${D}/${v}_infer_${split}.log" 2>&1
      mv "${scores}.tmp" "${scores}"
    fi
  done
  echo "[${v}] done $(date +%H:%M)"
done

runs=()
for f in "${D}"/*_test_scores.txt; do
  runs+=("$(basename "${f}" _test_scores.txt)=${f}")
done
python diag/eval_rerank.py beauty aug-5 "${runs[@]}" --out "${D}/results.md"
echo "ALL DONE"
