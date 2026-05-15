"""
export_tevatron.py (unified)
============================
Xuất dữ liệu Tevatron với các tùy chọn kết hợp tự do:

  Query augmentation (--window_size):
    Không truyền    → 1 sample/user (dùng precomputed context từ preprocess)
    --window_size N → sliding window kích thước N, sinh N-3 samples/user

  Negative sampling (--neg_strategy  --num_hard_neg  --num_random_neg):

    random (mặc định)
      Toàn bộ pool là random negatives.
      Số lượng = num_hard_neg + num_random_neg (mặc định 10+40 = 50/query).

    bm25
      Toàn bộ pool là BM25 hard negatives (top scored từ BM25 index).
      Số lượng = num_hard_neg + num_random_neg (mặc định 10+40 = 50/query).

    mixed
      Pool gồm hai phần: num_hard_neg BM25 hard + num_random_neg random.
      Mặc định: 10 hard + 40 random = 50/query.
      Tevatron sẽ sample train_group_size-1 negatives từ pool này mỗi step.

Output: dataset/tevatron/<dataset>[-<tag>]/
  corpus.jsonl, train.jsonl, valid.jsonl, test.jsonl

Tag được tự động sinh từ tham số:
  window_size=None, neg=random  → ""         → beauty/
  window_size=3,    neg=random  → "w3"       → beauty-w3/
  window_size=None, neg=mixed   → "mixed"    → beauty-mixed/
  window_size=3,    neg=mixed   → "w3-mixed" → beauty-w3-mixed/
  (Override với --tag)

Ví dụ:

  # Standard — 1 sample/user, 50 random negatives (mặc định)
  python export_tevatron.py beauty

  # Augmented — sliding window=3, 50 random negatives
  python export_tevatron.py beauty --window_size 3

  # BM25 hard — 1 sample/user, 50 BM25 hard negatives
  python export_tevatron.py beauty --neg_strategy bm25

  # Mixed — 1 sample/user, 10 BM25 hard + 40 random negatives (mặc định ratio)
  python export_tevatron.py beauty --neg_strategy mixed

  # Mixed — tùy chỉnh ratio: 20 hard + 30 random = 50/query
  python export_tevatron.py beauty --neg_strategy mixed --num_hard_neg 20 --num_random_neg 30

  # Augmented + Mixed — sliding window=3, 10 BM25 hard + 40 random negatives
  python export_tevatron.py beauty --window_size 3 --neg_strategy mixed

  # ML-1M: giới hạn 20 samples/user, augmented + mixed negatives
  python export_tevatron.py ml-1m --window_size 3 --max_aug_per_user 20 --neg_strategy mixed
"""

import argparse
import json
import math
import os
import random
from collections import Counter

from tqdm import tqdm
from preprocess import preprocess, build_item_text, CONTEXT_SIZE

random.seed(42)
OUTPUT_BASE = "dataset/tevatron"


# ── BM25 ──────────────────────────────────────────────────────────────────────

class BM25:
    """
    BM25Okapi với inverted index — không cần external dependency.

    Inverted index giúp chỉ duyệt docs chứa query term, hiệu quả hơn
    brute-force O(N×|query|) với corpus nhỏ (~10k–50k items).
    """

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1    = k1
        self.b     = b
        self.N     = len(docs)
        self.avgdl = sum(len(d) for d in docs) / self.N if self.N else 1.0

        # inverted index: term → [(doc_id, tf), ...]
        self._index: dict[str, list[tuple[int, int]]] = {}
        self._dl: list[int] = []

        for doc_id, doc in enumerate(docs):
            self._dl.append(len(doc))
            for term, freq in Counter(doc).items():
                self._index.setdefault(term, []).append((doc_id, freq))

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)  — luôn dương
        self._idf: dict[str, float] = {
            term: math.log(
                (self.N - len(postings) + 0.5) / (len(postings) + 0.5) + 1
            )
            for term, postings in self._index.items()
        }

    def top_k(
        self,
        query: list[str],
        k: int,
        exclude: set[int],
    ) -> list[int]:
        """
        Trả về top-k doc indices (không có trong exclude), xếp theo BM25 score giảm dần.
        """
        scores: dict[int, float] = {}
        for term in query:
            if term not in self._index:
                continue
            idf = self._idf[term]
            for doc_id, f in self._index[term]:
                if doc_id in exclude:
                    continue
                dl = self._dl[doc_id]
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (
                    f * (self.k1 + 1)
                    / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                )
        return sorted(scores, key=lambda i: -scores[i])[:k]


# ── Tevatron helpers ──────────────────────────────────────────────────────────

def make_passage(item_id: str, item_meta: dict) -> dict:
    text = build_item_text(item_id, item_meta)
    return {"docid": item_id, "title": "", "text": f"Passage: {text}"}


def build_query(context: list, item_meta: dict) -> str:
    if not context:
        return "Query: </s>"
    texts = [build_item_text(iid, item_meta) for iid in context]
    return "Query: " + ", ".join(texts) + " </s>"


def tokenize(text: str) -> list[str]:
    return text.lower().split()


# ── Negative sampling ─────────────────────────────────────────────────────────

def sample_negatives(
    context: list[str],
    exclude_ids: set[str],
    all_items_list: list[str],
    item_to_idx: dict[str, int],
    item_meta: dict,
    strategy: str,
    num_hard: int,
    num_random: int,
    bm25: BM25 | None = None,
) -> list[str]:
    """
    Trả về list item IDs làm negatives.

    strategy=random : num_hard+num_random negatives ngẫu nhiên
    strategy=bm25   : top-(num_hard+num_random) từ BM25
    strategy=mixed  : num_hard BM25 hard + num_random random

    Cold-start (context rỗng): luôn fallback về random vì BM25 query rỗng
    sẽ cho score 0 cho tất cả docs — không có ý nghĩa gì.
    """
    total = num_hard + num_random
    non_excluded = [i for i in all_items_list if i not in exclude_ids]

    if strategy == "random" or not context:
        return random.sample(non_excluded, min(total, len(non_excluded)))

    # BM25-based: query = titles của context items
    query_tokens = tokenize(
        ", ".join(build_item_text(iid, item_meta) for iid in context)
    )
    exclude_idx = {item_to_idx[i] for i in exclude_ids if i in item_to_idx}
    hard_indices = bm25.top_k(query_tokens, num_hard + num_random, exclude_idx)
    hard_ids = [all_items_list[i] for i in hard_indices[:num_hard]]

    if strategy == "bm25":
        extra_ids = [all_items_list[i] for i in hard_indices[num_hard:]]
        return hard_ids + extra_ids

    # mixed
    hard_set = set(hard_ids)
    random_pool = [i for i in non_excluded if i not in hard_set]
    random_neg = random.sample(random_pool, min(num_random, len(random_pool)))
    return hard_ids + random_neg


# ── Auto tag ──────────────────────────────────────────────────────────────────

def auto_tag(window_size: int | None, neg_strategy: str) -> str:
    parts = []
    if window_size is not None:
        parts.append(f"w{window_size}")
    if neg_strategy != "random":
        parts.append(neg_strategy)
    return "-".join(parts)


# ── Main export ───────────────────────────────────────────────────────────────

def export_tevatron(
    dataset_name: str,
    window_size: int | None = None,
    max_aug_per_user: int | None = None,
    neg_strategy: str = "random",
    num_hard: int = 10,
    num_random: int = 40,
    tag: str | None = None,
):
    """
    Entry point cho cả import lẫn CLI.
    tag=None → auto_tag(window_size, neg_strategy).
    tag=""   → xuất thẳng vào <dataset>/ (không có hậu tố).
    """
    if tag is None:
        tag = auto_tag(window_size, neg_strategy)

    splits, all_items, item_meta, all_items_set, sequences = preprocess(dataset_name)
    all_items_list = list(all_items_set if isinstance(all_items_set, set) else all_items)
    item_to_idx    = {item_id: idx for idx, item_id in enumerate(all_items_list)}

    out_dir = os.path.join(OUTPUT_BASE, f"{dataset_name}-{tag}" if tag else dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Print config ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Tevatron export → {out_dir}/")
    if window_size is None:
        print(f"  query aug  : off (1 sample/user)")
    else:
        print(f"  query aug  : sliding window={window_size}"
              + (f", max {max_aug_per_user}/user" if max_aug_per_user else ""))
    if neg_strategy == "random":
        print(f"  negatives  : random × {num_hard + num_random}")
    else:
        print(f"  negatives  : {neg_strategy} ({num_hard} hard + {num_random} random)")
    print(f"{'─' * 60}\n")

    # ── BM25 index (chỉ build khi cần) ───────────────────────────────────────
    bm25 = None
    if neg_strategy != "random":
        print("  Building BM25 index...", end=" ", flush=True)
        corpus_tokens = [
            tokenize(build_item_text(item_id, item_meta))
            for item_id in all_items_list
        ]
        bm25 = BM25(corpus_tokens)
        print(f"done ({len(all_items_list)} items indexed)\n")

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

    # ── train.jsonl ───────────────────────────────────────────────────────────
    train_path  = os.path.join(out_dir, "train.jsonl")
    total_train = 0
    cold_start  = 0

    with open(train_path, "w", encoding="utf-8") as f:
        for user_id, seq in tqdm(sequences.items(), desc="  train", unit="user"):
            N          = len(seq)
            user_items = set(seq)

            # Tạo list (context, positive, query_id_suffix) cho user này
            if window_size is None:
                # Standard: 1 sample/user — dùng context đã tính sẵn (xử lý đúng cold-start)
                data      = splits["train"][user_id]
                positions = [(data["tevatron_context"], data["positive"], "train")]
            else:
                # Augmented: sliding window qua tất cả train positions
                train_pos = list(range(1, N - 2))
                if max_aug_per_user and len(train_pos) > max_aug_per_user:
                    train_pos = train_pos[-max_aug_per_user:]
                positions = [
                    (seq[max(0, j - window_size) : j], seq[j], f"train_{j}")
                    for j in train_pos
                ]

            for context, positive, qid_suffix in positions:
                if not context:
                    cold_start += 1

                neg_ids = sample_negatives(
                    context        = context,
                    exclude_ids    = user_items,
                    all_items_list = all_items_list,
                    item_to_idx    = item_to_idx,
                    item_meta      = item_meta,
                    strategy       = neg_strategy,
                    num_hard       = num_hard,
                    num_random     = num_random,
                    bm25           = bm25,
                )

                f.write(json.dumps({
                    "query_id":          f"{user_id}_{qid_suffix}",
                    "query":             build_query(context, item_meta),
                    "positive_passages": [make_passage(positive, item_meta)],
                    "negative_passages": [make_passage(nid, item_meta) for nid in neg_ids],
                }, ensure_ascii=False) + "\n")
                total_train += 1

    if window_size is not None:
        orig = len(sequences)
        print(f"  train.jsonl    : {total_train:>6} samples  "
              f"({total_train / orig:.1f}× original {orig})")
    else:
        print(f"  train.jsonl    : {total_train:>6} queries")
    print(f"    cold-start (empty context): {cold_start} samples")

    # ── valid.jsonl / test.jsonl ──────────────────────────────────────────────
    for split_name in ["valid", "test"]:
        split_data = splits[split_name]
        out_path   = os.path.join(out_dir, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for user_id, data in tqdm(
                split_data.items(), desc=f"  {split_name:5s}", unit="query"
            ):
                f.write(json.dumps({
                    "query_id":          f"{user_id}_{split_name}",
                    "query":             build_query(data["tevatron_context"], item_meta),
                    "positive_passages": [make_passage(data["positive"], item_meta)],
                    "negative_passages": [],
                }, ensure_ascii=False) + "\n")
        print(f"  {split_name}.jsonl     : {len(split_data):>6} queries (unchanged)")

    print(f"\n✓ Done → {out_dir}/\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export Tevatron data — query augmentation + negative strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", help="beauty | sports | ml-1m | steam")

    aug = parser.add_argument_group("Query augmentation")
    aug.add_argument(
        "--window_size", type=int, default=None,
        metavar="N",
        help="Sliding window size (default: off — 1 sample/user)",
    )
    aug.add_argument(
        "--max_aug_per_user", type=int, default=None,
        metavar="N",
        help="Max samples/user for augmentation — recommend 20 for ml-1m",
    )

    neg = parser.add_argument_group("Negative sampling")
    neg.add_argument(
        "--neg_strategy", choices=["random", "bm25", "mixed"], default="random",
        help="Negative sampling strategy",
    )
    neg.add_argument(
        "--num_hard_neg", type=int, default=10,
        metavar="N",
        help="BM25 hard negatives per query (bm25/mixed)",
    )
    neg.add_argument(
        "--num_random_neg", type=int, default=40,
        metavar="N",
        help="Random negatives per query (random/mixed). "
             "Total pool = num_hard_neg + num_random_neg",
    )

    parser.add_argument(
        "--tag", default=None,
        help="Override auto-generated output tag",
    )

    args = parser.parse_args()

    export_tevatron(
        dataset_name   = args.dataset,
        window_size    = args.window_size,
        max_aug_per_user = args.max_aug_per_user,
        neg_strategy   = args.neg_strategy,
        num_hard       = args.num_hard_neg,
        num_random     = args.num_random_neg,
        tag            = args.tag,
    )


if __name__ == "__main__":
    main()
