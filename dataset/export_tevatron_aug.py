"""
export_tevatron_aug.py
======================
Xuất training data với data augmentation: sliding window trên toàn bộ history.

So với export_tevatron.py (1 sample/user), script này tạo N-3 training
samples/user bằng cách trượt cửa sổ kích thước WINDOW_SIZE từ đầu đến cuối
sequence, dừng lại trước valid/test positives.

Ví dụ (window_size=3, sequence=[i1..i9], valid=i8, test=i9):
  Train : ({i1},i2), ({i1,i2},i3), ({i1,i2,i3},i4),
          ({i2,i3,i4},i5), ({i3,i4,i5},i6), ({i4,i5,i6},i7)
  Valid : ({i5,i6,i7}, i8)  ← không đổi so với export_tevatron.py
  Test  : ({i6,i7,i8}, i9)  ← không đổi so với export_tevatron.py

Negatives: loại trừ toàn bộ sequence của user (gồm cả i8, i9).

Cách dùng:
  python export_tevatron_aug.py <dataset> [--window_size N]
                                          [--max_aug_per_user N]
                                          [--tag TAG]

  --window_size      : kích thước cửa sổ context (mặc định: CONTEXT_SIZE=3)
  --max_aug_per_user : giới hạn số samples/user, lấy K vị trí gần nhất
                       (mặc định: unlimited — khuyến nghị đặt với ML-1M)
  --tag              : hậu tố thư mục output (mặc định: aug hoặc aug-w{N})

Ví dụ:
  python export_tevatron_aug.py beauty
  python export_tevatron_aug.py beauty --window_size 5 --tag aug-w5
  python export_tevatron_aug.py ml-1m --max_aug_per_user 20
"""

import json
import os
import random
import argparse

from tqdm import tqdm
from preprocess import preprocess, build_item_text, CONTEXT_SIZE

random.seed(42)
NUM_NEGATIVES = 50
OUTPUT_BASE   = "dataset/tevatron"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_passage(item_id: str, item_meta: dict) -> dict:
    text = build_item_text(item_id, item_meta)
    return {"docid": item_id, "title": "", "text": f"Passage: {text}"}


def build_query(context: list, item_meta: dict) -> str:
    if not context:
        return "Query: </s>"
    texts = [build_item_text(iid, item_meta) for iid in context]
    return "Query: " + ", ".join(texts) + " </s>"


# ── Export ────────────────────────────────────────────────────────────────────

def export_aug(dataset_name: str, window_size: int,
               max_aug_per_user: int | None, tag: str):

    splits, all_items, item_meta, all_items_set, sequences = \
        preprocess(dataset_name)

    all_items_list = list(all_items_set) if isinstance(all_items_set, set) \
        else list(all_items)

    out_dir = os.path.join(OUTPUT_BASE, f"{dataset_name}-{tag}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"Augmented Tevatron export → {out_dir}/")
    print(f"  window_size      : {window_size}")
    print(f"  max_aug_per_user : {max_aug_per_user or 'unlimited'}")
    print(f"{'─' * 60}\n")

    # ── corpus.jsonl ──────────────────────────────────────────────────────────

    corpus_path = os.path.join(out_dir, "corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for item_id in tqdm(all_items_list, desc="  corpus", unit="item"):
            text = build_item_text(item_id, item_meta)
            f.write(json.dumps(
                {"docid": item_id, "text": f"Passage: {text}"},
                ensure_ascii=False,
            ) + "\n")
    print(f"  corpus.jsonl   : {len(all_items_list):>6} items")

    # ── train.jsonl (augmented — sliding window) ──────────────────────────────

    total_samples = 0
    cold_start_samples = 0
    train_path = os.path.join(out_dir, "train.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for user_id, seq in tqdm(sequences.items(), desc="  train (aug)", unit="user"):
            N = len(seq)

            # Positions 1..N-3 (0-indexed): predict seq[j], exclude seq[N-2]=valid, seq[N-1]=test
            train_positions = list(range(1, N - 2))

            # Giới hạn số samples/user: lấy K vị trí gần nhất (most recent)
            if max_aug_per_user and len(train_positions) > max_aug_per_user:
                train_positions = train_positions[-max_aug_per_user:]

            # Loại trừ toàn bộ sequence khỏi negative pool (gồm cả valid/test)
            user_items = set(seq)
            candidate_negs = [i for i in all_items_list if i not in user_items]

            for j in train_positions:
                positive = seq[j]
                context  = seq[max(0, j - window_size):j]

                if not context:
                    cold_start_samples += 1

                sampled_negs = random.sample(
                    candidate_negs,
                    min(NUM_NEGATIVES, len(candidate_negs)),
                )

                f.write(json.dumps(
                    {
                        "query_id":          f"{user_id}_train_{j}",
                        "query":             build_query(context, item_meta),
                        "positive_passages": [make_passage(positive, item_meta)],
                        "negative_passages": [make_passage(nid, item_meta)
                                              for nid in sampled_negs],
                    },
                    ensure_ascii=False,
                ) + "\n")
                total_samples += 1

    orig_count = len(sequences)
    print(f"  train.jsonl    : {total_samples:>6} samples  "
          f"({total_samples / orig_count:.1f}× original {orig_count})")
    print(f"    cold-start (empty context) : {cold_start_samples} samples")

    # ── valid.jsonl / test.jsonl (không đổi — 1 query/user) ─────────────────

    for split_name in ["valid", "test"]:
        split_data = splits[split_name]
        out_path = os.path.join(out_dir, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for user_id, data in tqdm(
                split_data.items(), desc=f"  {split_name:5s}", unit="query"
            ):
                f.write(json.dumps(
                    {
                        "query_id":          f"{user_id}_{split_name}",
                        "query":             build_query(
                            data["tevatron_context"], item_meta
                        ),
                        "positive_passages": [make_passage(
                            data["positive"], item_meta
                        )],
                        "negative_passages": [],
                    },
                    ensure_ascii=False,
                ) + "\n")
        print(f"  {split_name}.jsonl     : {len(split_data):>6} queries "
              f"(unchanged)")

    print(f"\n✓ Done → {out_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export augmented Tevatron training data (sliding window)"
    )
    parser.add_argument(
        "dataset",
        help="Dataset: beauty | sports | ml-1m | steam",
    )
    parser.add_argument(
        "--window_size", type=int, default=CONTEXT_SIZE,
        help=f"Context window size (default: CONTEXT_SIZE={CONTEXT_SIZE})",
    )
    parser.add_argument(
        "--max_aug_per_user", type=int, default=None,
        help="Max training samples per user, taking K most recent positions "
             "(default: unlimited; recommend --max_aug_per_user 20 for ml-1m)",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Output dir suffix (default: 'aug' if window=default, else 'aug-w{N}')",
    )
    args = parser.parse_args()

    if args.tag is None:
        args.tag = "aug" if args.window_size == CONTEXT_SIZE \
                   else f"aug-w{args.window_size}"

    export_aug(args.dataset, args.window_size, args.max_aug_per_user, args.tag)
