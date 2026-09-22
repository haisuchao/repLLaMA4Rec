for ds in beauty; do
  for cs in 5; do
    if [ "$cs" = "3" ]; then
      variant_flag=""                    # context_size=3 = mặc định → data ở thư mục trơn, không hậu tố
    else
      variant_flag="--data-variant cs${cs}"
    fi
    tag="cs${cs}-gs32"

    ./train.sh $ds --model Qwen/Qwen3-Embedding-0.6B $variant_flag --tag $tag --group-size 32
    ./eval.sh  $ds --model Qwen/Qwen3-Embedding-0.6B --tag $tag $variant_flag
    python eval_filter.py $ds --model Qwen/Qwen3-Embedding-0.6B --tag $tag --filter-mode full
  done
done