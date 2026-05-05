"""
export_tevatron.py
==================
Xuất dữ liệu sang định dạng Tevatron v2:

  dataset/tevatron/<dataset_name>/
    ├── corpus.jsonl   – tất cả items với text đầy đủ
    ├── train.jsonl    – queries dùng CONTEXT_SIZE item gần nhất
    ├── valid.jsonl
    └── test.jsonl

Định dạng mỗi dòng trong train/valid/test.jsonl:
  {
    "query_id": "<user_id>_<split>",
    "query": "Query: <item_{n-3}> </s> <item_{n-2}> </s> <item_{n-1}> </s>",
    "positive_passages": [{"docid": "...", "title": "", "text": "Passage: ..."}],
    "negative_passages": [{"docid": "...", "title": "", "text": "Passage: ..."}, ...]
  }

Lưu ý: Chỉ dùng CONTEXT_SIZE=3 item gần nhất làm query vì giới hạn
        context length của LLM. RecBole/SASRec dùng toàn bộ history.
"""

import json
import os
import random

from tqdm import tqdm
from preprocess import preprocess, build_item_text, CONTEXT_SIZE

# ── Config ───────────────────────────────────────────────────────────────────

random.seed(42)
NUM_NEGATIVES = 50      # số negative passages mỗi query
OUTPUT_BASE   = "dataset/tevatron"

# ── Helper ───────────────────────────────────────────────────────────────────

def make_passage(item_id: str, item_meta: dict) -> dict:
    """Tạo passage object theo chuẩn Tevatron v2."""
    text = build_item_text(item_id, item_meta)
    return {
        "docid": item_id,
        "title": "",
        "text":  f"Passage: {text}",
    }


# ── Main export ───────────────────────────────────────────────────────────────

def export_tevatron(dataset_name: str):
    splits, all_items, item_meta, all_items_set, sequences = \
        preprocess(dataset_name)

    out_dir = os.path.join(OUTPUT_BASE, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # all_items_set dùng để sample negative on-the-fly (không lưu toàn bộ vào RAM)
    all_items_list = list(all_items_set) if isinstance(all_items_set, set) \
        else list(all_items)

    # ── corpus.jsonl ─────────────────────────────────────────────────────────
    corpus_path = os.path.join(out_dir, "corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for item_id in tqdm(all_items_list, desc="  corpus", unit="item"):
            text = build_item_text(item_id, item_meta)
            f.write(json.dumps(
                {"docid": item_id, "text": f"Passage: {text}"},
                ensure_ascii=False,
            ) + "\n")
    print(f"  [Tevatron] corpus.jsonl  : {len(all_items_list):>6} items   → {corpus_path}")

    # ── train / valid / test.jsonl ────────────────────────────────────────────
    for split_name, split_data in splits.items():
        out_path = os.path.join(out_dir, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for user_id, data in tqdm(
                split_data.items(),
                desc=f"  {split_name:5s}",
                unit="query",
            ):

                # Query: context items gần nhất, nối bằng </s>
                # Cold-start: context có thể rỗng → query chỉ có prefix
                context_items = data["tevatron_context"]
                if context_items:
                    context_texts = [
                        build_item_text(iid, item_meta)
                        for iid in context_items
                    ]
                    query_text = (
                        "Query: "
                        + ", ".join(context_texts)
                        + " </s>"
                    )
                else:
                    # Cold-start hoàn toàn: không có history
                    query_text = "Query: </s>"

                # Positive
                positive_passages = [make_passage(data["positive"], item_meta)]

                # Negative: chỉ sinh cho train — valid/test không cần vì
                # evaluation dùng FAISS search trên toàn bộ corpus
                if split_name == "train":
                    user_items = set(sequences[user_id])
                    candidate_negs = [i for i in all_items_list if i not in user_items]
                    sampled = random.sample(candidate_negs, min(NUM_NEGATIVES, len(candidate_negs)))
                    negative_passages = [make_passage(nid, item_meta) for nid in sampled]
                else:
                    negative_passages = []

                f.write(json.dumps(
                    {
                        "query_id":           f"{user_id}_{split_name}",
                        "query":              query_text,
                        "positive_passages":  positive_passages,
                        "negative_passages":  negative_passages,
                    },
                    ensure_ascii=False,
                ) + "\n")

        print(
            f"  [Tevatron] {split_name}.jsonl   : "
            f"{len(split_data):>6} queries  → {out_path}"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    cold_start_train = sum(
        1 for d in splits["train"].values()
        if len(d["tevatron_context"]) == 0
    )
    print(f"\n  Context window : up to {CONTEXT_SIZE} items (shorter for cold-start)")
    print(f"  Cold-start     : {cold_start_train} train queries with empty context")
    print(f"  Negatives/query: {NUM_NEGATIVES} (sampled on-the-fly, no RAM overhead)")
    print(f"  Output dir     : {out_dir}/")


if __name__ == "__main__":
    import sys

    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["beauty"]
    for ds in datasets:
        print(f"\nExporting Tevatron → {ds}")
        export_tevatron(ds)
    print("\nDone.")