"""
export_recbole.py
=================
Xuất dữ liệu sang định dạng RecBole cho SASRec:

  dataset/recbole/<dataset_name>/
    ├── <dataset_name>.inter          – toàn bộ interactions (full history)
    ├── <dataset_name>.item           – metadata của item
    └── sasrec_<dataset_name>.yaml    – config SASRec tự sinh

Triết lý khác biệt với Tevatron:
  RecBole/SASRec dùng TOÀN BỘ history của user.
  SASRec tự học attention để biết item nào quan trọng.
  Tevatron/repLLaMA chỉ dùng CONTEXT_SIZE=3 item gần nhất
  vì bị giới hạn bởi context length của LLM.

Cách RecBole tự chia train/valid/test từ .inter:
  eval_args.split = {LS: valid_and_test}
    → item cuối    = test target
    → item áp chót = valid target
    → phần còn lại = train
  Đây chính xác là leave-one-out của SASRec paper.
"""

import os

from tqdm import tqdm
from preprocess import preprocess, build_item_text, CONTEXT_SIZE

# ── Config ───────────────────────────────────────────────────────────────────

OUTPUT_BASE          = "dataset/recbole"
MAX_ITEM_LIST_LENGTH = 200   # cap để tránh OOM với self-attention O(n²)


# ── Main export ───────────────────────────────────────────────────────────────

def export_recbole(dataset_name: str):
    splits, all_items, item_meta, all_items_set, sequences = \
        preprocess(dataset_name)

    # all_items_set giờ là set, chuyển về list để dùng
    all_items_list = list(all_items_set) if isinstance(all_items_set, set) \
        else list(all_items)

    out_dir = os.path.join(OUTPUT_BASE, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── .inter: TOÀN BỘ chuỗi tương tác theo thứ tự thời gian ───────────────
    # RecBole + SASRec tự học attention trên full history.
    # KHÔNG giới hạn k item như Tevatron (CONTEXT_SIZE=3).
    inter_path = os.path.join(out_dir, f"{dataset_name}.inter")
    total_interactions = 0
    with open(inter_path, "w", encoding="utf-8") as f:
        f.write("user_id:token\titem_id:token\ttimestamp:float\n")
        for user_id, seq in tqdm(sequences.items(), desc="  .inter", unit="user"):
            for t, item_id in enumerate(seq):
                f.write(f"{user_id}\t{item_id}\t{float(t)}\n")
                total_interactions += 1
    print(
        f"  [RecBole] {dataset_name}.inter  : "
        f"{len(sequences):>6} users, "
        f"{total_interactions:>8} interactions → {inter_path}"
    )

    # ── .item: metadata ──────────────────────────────────────────────────────
    item_path = os.path.join(out_dir, f"{dataset_name}.item")
    with open(item_path, "w", encoding="utf-8") as f:
        f.write("item_id:token\titem_title:token_seq\t"
                "item_brand:token\titem_category:token_seq\n")
        for item_id in tqdm(all_items_list, desc="  .item ", unit="item"):
            meta  = item_meta.get(item_id, {})
            title = (meta.get("title", "")      or "").replace("\t", " ").replace("\n", " ")
            brand = (meta.get("brand", "")      or "").replace("\t", " ").replace("\n", " ")
            cats  = (meta.get("categories", "") or "").replace("\t", " ").replace("\n", " ")
            f.write(f"{item_id}\t{title}\t{brand}\t{cats}\n")
    print(
        f"  [RecBole] {dataset_name}.item   : "
        f"{len(all_items_list):>6} items              → {item_path}"
    )

    # ── sasrec_<dataset>.yaml: config tự sinh ────────────────────────────────
    seq_lens    = [len(s) for s in sequences.values()]
    max_seq_len = max(seq_lens)
    avg_seq_len = sum(seq_lens) / len(seq_lens)
    sparsity    = 1 - total_interactions / (len(sequences) * len(all_items_list))
    # Cap MAX_ITEM_LIST_LENGTH để tránh OOM với self-attention O(n²)
    max_item_list_length = min(max_seq_len, MAX_ITEM_LIST_LENGTH)

    config_path = os.path.join(out_dir, f"sasrec_{dataset_name}.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"""\
# ============================================================
# Auto-generated SASRec config for: {dataset_name}
# ============================================================
# Dataset stats:
#   Users        : {len(sequences)}
#   Items        : {len(all_items)}
#   Interactions : {total_interactions}
#   Sparsity     : {sparsity:.4f}
#   Seq len      : avg={avg_seq_len:.1f}, max={max_seq_len}
#
# Key difference vs Tevatron:
#   RecBole uses up to MAX_ITEM_LIST_LENGTH={max_item_list_length} most recent items
#   Tevatron uses only {CONTEXT_SIZE} most recent items as query
# ============================================================

data_path: dataset/recbole
dataset: {dataset_name}

# ── Dataset fields ───────────────────────────────────────────
# Phải load timestamp để RecBole sort theo thứ tự thời gian (order: TO)
TIME_FIELD: timestamp
load_col:
  inter: [user_id, item_id, timestamp]

# ── Model ────────────────────────────────────────────────────
model: SASRec
hidden_size: 64
inner_size: 256
n_layers: 2
n_heads: 2
hidden_dropout_prob: 0.5
attn_dropout_prob: 0.5
hidden_act: gelu
layer_norm_eps: 1.0e-12
initializer_range: 0.02
loss_type: CE
train_neg_sample_args: ~  # CE loss tự xử lý negatives nội bộ

# Capped tại {MAX_ITEM_LIST_LENGTH} để tránh OOM với self-attention O(n²)
# Dataset max seq len = {max_seq_len}
MAX_ITEM_LIST_LENGTH: {max_item_list_length}

# ── Training ─────────────────────────────────────────────────
epochs: 200
train_batch_size: 256
eval_batch_size: 512
learner: adam
learning_rate: 0.001
weight_decay: 0.0
stopping_step: 10          # early stopping

# ── Evaluation ───────────────────────────────────────────────
# Leave-one-out split giống SASRec paper và cách chia của Tevatron:
#   item cuối    → test target
#   item áp chót → valid target
#   phần còn lại → train
eval_args:
  split: {{LS: valid_and_test}}
  group_by: user
  order: TO                  # Time Order — bắt buộc với sequential models
  mode: full                 # Rank trên toàn bộ {len(all_items)} items

metrics: [Recall, NDCG, MRR, Hit]
topk: [10, 20]
valid_metric: NDCG@10
""")
    print(
        f"  [RecBole] sasrec_{dataset_name}.yaml: "
        f"MAX_ITEM_LIST_LENGTH={max_seq_len} → {config_path}"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Full history   : YES (avg {avg_seq_len:.1f} items/user)")
    print(f"  Tevatron uses  : {CONTEXT_SIZE} items/query (context window)")
    print(f"  Output dir     : {out_dir}/")


if __name__ == "__main__":
    import sys

    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["beauty"]
    for ds in datasets:
        print(f"\nExporting RecBole → {ds}")
        export_recbole(ds)
    print("\nDone.")