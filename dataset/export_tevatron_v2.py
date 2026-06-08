"""
export_tevatron_v2.py
=====================
Xuất dữ liệu Tevatron theo format v2 (instruction-based, structured metadata).

Khác biệt so với export_tevatron.py (v1):
  - build_item_text()  : "Title: X. Category: A > B > C. Brand: Z." (v1: title only)
  - build_query()      : instruction format, newline-separated items (v1: "Query: t1, t2 </s>")
  - make_passage()     : không có prefix "Passage: " (v1: có)
  - corpus.jsonl       : không có prefix "Passage: " (v1: có)
  - auto_tag()         : luôn prepend "v2" để phân biệt output directory
  - Cần dùng --query_prefix "" --passage_prefix "" trong train.sh/eval.sh

Output: dataset/tevatron/<dataset>-v2[-<options>]/
  corpus.jsonl, train.jsonl, valid.jsonl, test.jsonl

Ví dụ:
  python export_tevatron_v2.py beauty                     # → beauty-v2/
  python export_tevatron_v2.py beauty --augment           # → beauty-v2-aug/
  python export_tevatron_v2.py beauty --context_size 5    # → beauty-v2-cs5/
  python export_tevatron_v2.py beauty --augment --context_size 5  # → beauty-v2-cs5-aug/
  python export_tevatron_v2.py ml-1m --augment            # → ml-1m-v2-aug/
  python export_tevatron_v2.py beauty --tag my-exp        # → beauty-my-exp/ (override)
"""

import argparse
import json
import math
import os
import random
from collections import Counter

from tqdm import tqdm
from preprocess import preprocess, CONTEXT_SIZE  # NOT importing build_item_text

random.seed(42)
OUTPUT_BASE = "dataset/tevatron"


# ── Instructions ──────────────────────────────────────────────────────────────

AMAZON_INSTRUCTION = (
    "Below is a customer's purchase history on Amazon, listed in chronological order "
    "(earliest to latest). Each item is represented by the following format: "
    "Title: <item title>. Category: <item category 1> > <item category 2> > .... "
    "Brand: <item brand>. (Brand field is omitted when unavailable.) "
    "Based on this history, predict only one item the customer is most likely to "
    "purchase next, described in the same format."
)

ML1M_INSTRUCTION = (
    "Below is a customer's movie watch history, listed in chronological order "
    "(earliest to latest). Each item is represented by the following format: "
    "Title: <movie title>. Genre: <genre 1>, <genre 2>, .... "
    "Based on this history, predict only one movie the customer is most likely to "
    "watch next, described in the same format."
)


# ── Item text builder (v2) ────────────────────────────────────────────────────

def build_item_text(item_id: str, item_meta: dict) -> str:
    """
    v2 format: "Title: X. Category: A > B > C. Brand: Z."
    ML-1M:     "Title: X. Genre: Action, Adventure."
    Brand field is omitted when unavailable.
    """
    meta = item_meta.get(item_id, {})

    title = meta.get("title") or ""
    if isinstance(title, list):
        title = " ".join(str(t) for t in title)
    title = str(title).strip()
    if not title:
        return item_id

    parts = [f"Title: {title}."]

    # Genre (ML-1M) takes priority over Category (Amazon/Steam)
    if "genres" in meta:
        genres = str(meta.get("genres", "") or "").strip()
        if genres:
            # "Action|Adventure|Sci-Fi" → "Action, Adventure, Sci-Fi"
            parts.append(f"Genre: {genres.replace('|', ', ')}.")
    else:
        cats = str(meta.get("categories", "") or "").strip()
        if cats:
            parts.append(f"Category: {cats}.")

    brand = str(meta.get("brand", "") or "").strip()
    if brand:
        parts.append(f"Brand: {brand}.")

    return " ".join(parts)


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


# ── Tevatron helpers (v2) ─────────────────────────────────────────────────────

def make_passage(item_id: str, item_meta: dict) -> dict:
    # v2: no "Passage: " prefix — text is already "Title: X. Category: Y. [Brand: Z.]"
    return {"docid": item_id, "title": "", "text": build_item_text(item_id, item_meta)}


def build_query(context: list, item_meta: dict, dataset_name: str = "beauty") -> str:
    """
    v2 instruction format. No " </s>" — Tevatron appends <|im_end|> via --append_eos_token.
    """
    inst   = ML1M_INSTRUCTION if dataset_name == "ml-1m" else AMAZON_INSTRUCTION
    header = "Watch history:" if dataset_name == "ml-1m" else "Purchase history:"
    if not context:
        return f"{inst}\n{header}\n(no history)"
    items_text = "\n".join(build_item_text(iid, item_meta) for iid in context)
    return f"{inst}\n{header}\n{items_text}"


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

    # BM25-based: query = v2 text của context items (có category/brand → richer signal)
    query_tokens = tokenize(
        " ".join(build_item_text(iid, item_meta) for iid in context)
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

def auto_tag(context_size: int, augment: bool, neg_strategy: str) -> str:
    parts = ["v2"]  # always prefix with "v2" to distinguish from v1 output
    if context_size != CONTEXT_SIZE:
        parts.append(f"cs{context_size}")
    if augment:
        parts.append("aug")
    if neg_strategy != "random":
        parts.append(neg_strategy)
    return "-".join(parts)


# ── Main export ───────────────────────────────────────────────────────────────

def export_tevatron(
    dataset_name: str,
    context_size: int = CONTEXT_SIZE,
    augment: bool = False,
    max_aug_per_user: int | None = None,
    neg_strategy: str = "random",
    num_hard: int = 10,
    num_random: int = 40,
    tag: str | None = None,
):
    """
    Entry point cho cả import lẫn CLI.
    tag=None → auto_tag(context_size, augment, neg_strategy)  e.g. "v2", "v2-aug", "v2-cs5"
    tag=""   → xuất thẳng vào <dataset>/ (không có hậu tố, hiếm dùng)
    """
    if tag is None:
        tag = auto_tag(context_size, augment, neg_strategy)

    splits, all_items, item_meta, all_items_set, sequences = preprocess(dataset_name)
    all_items_list = list(all_items_set if isinstance(all_items_set, set) else all_items)
    item_to_idx    = {item_id: idx for idx, item_id in enumerate(all_items_list)}

    out_dir = os.path.join(OUTPUT_BASE, f"{dataset_name}-{tag}" if tag else dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Print config ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Tevatron v2 export → {out_dir}/")
    print(f"  format     : instruction-based, structured metadata")
    if not augment:
        cs_note = "" if context_size == CONTEXT_SIZE else f" (context_size={context_size})"
        print(f"  query aug  : off (1 sample/user){cs_note}")
    else:
        aug_note = f", max {max_aug_per_user}/user" if max_aug_per_user else ""
        print(f"  query aug  : on (sliding window, context={context_size}{aug_note})")
    if neg_strategy == "random":
        print(f"  negatives  : random × {num_hard + num_random}")
    else:
        print(f"  negatives  : {neg_strategy} ({num_hard} hard + {num_random} random)")
    print(f"{'─' * 60}\n")

    # ── BM25 index (chỉ build khi cần) ───────────────────────────────────────
    bm25 = None
    if neg_strategy != "random":
        print("  Building BM25 index (v2 text)...", end=" ", flush=True)
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
            # v2: no "Passage: " prefix
            f.write(json.dumps(
                {"docid": item_id, "text": build_item_text(item_id, item_meta)},
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

            if not augment:
                data = splits["train"][user_id]
                ctx  = data["tevatron_context"] if context_size == CONTEXT_SIZE \
                       else seq[: N - 3][-context_size:]
                positions = [(ctx, data["positive"], "train")]
            else:
                train_pos = list(range(1, N - 2))
                if max_aug_per_user and len(train_pos) > max_aug_per_user:
                    train_pos = train_pos[-max_aug_per_user:]
                positions = [
                    (seq[max(0, j - context_size) : j], seq[j], f"train_{j}")
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
                    "query":             build_query(context, item_meta, dataset_name),
                    "positive_passages": [make_passage(positive, item_meta)],
                    "negative_passages": [make_passage(nid, item_meta) for nid in neg_ids],
                }, ensure_ascii=False) + "\n")
                total_train += 1

    if augment:
        orig = len(sequences)
        print(f"  train.jsonl    : {total_train:>6} samples  "
              f"({total_train / orig:.1f}× original {orig})")
    else:
        print(f"  train.jsonl    : {total_train:>6} queries")
    print(f"    cold-start (empty context): {cold_start} samples")

    # ── valid.jsonl / test.jsonl ──────────────────────────────────────────────
    split_offsets = {"valid": 2, "test": 1}
    for split_name in ["valid", "test"]:
        split_data = splits[split_name]
        offset     = split_offsets[split_name]
        out_path   = os.path.join(out_dir, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for user_id, data in tqdm(
                split_data.items(), desc=f"  {split_name:5s}", unit="query"
            ):
                if context_size == CONTEXT_SIZE:
                    ctx = data["tevatron_context"]
                else:
                    seq = sequences[user_id]
                    ctx = seq[: len(seq) - offset][-context_size:]
                f.write(json.dumps({
                    "query_id":          f"{user_id}_{split_name}",
                    "query":             build_query(ctx, item_meta, dataset_name),
                    "positive_passages": [make_passage(data["positive"], item_meta)],
                    "negative_passages": [],
                }, ensure_ascii=False) + "\n")
        print(f"  {split_name}.jsonl     : {len(split_data):>6} queries (unchanged)")

    print(f"\n✓ Done → {out_dir}/")
    print(f"  Dùng --query_prefix \"\" --passage_prefix \"\" trong train.sh/eval.sh")
    print(f"  Hoặc truyền --v2-format khi gọi ./train.sh và ./eval.sh\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export Tevatron data — v2 format (instruction-based, structured metadata)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", help="beauty | sports | ml-1m | steam")

    aug = parser.add_argument_group("Query augmentation & context")
    aug.add_argument(
        "--context_size", type=int, default=CONTEXT_SIZE,
        metavar="N",
        help=f"Số item gần nhất dùng làm query (default: {CONTEXT_SIZE})",
    )
    aug.add_argument(
        "--augment", action="store_true",
        help="Bật sliding window augmentation — sinh N-3 samples/user thay vì 1",
    )
    aug.add_argument(
        "--max_aug_per_user", type=int, default=None,
        metavar="N",
        help="Max samples/user khi augment — recommend 20 for ml-1m",
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
        help="Random negatives per query (random/mixed).",
    )

    parser.add_argument(
        "--tag", default=None,
        help="Override auto-generated output tag (default: v2, v2-aug, v2-cs5, ...)",
    )

    args = parser.parse_args()

    export_tevatron(
        dataset_name     = args.dataset,
        context_size     = args.context_size,
        augment          = args.augment,
        max_aug_per_user = args.max_aug_per_user,
        neg_strategy     = args.neg_strategy,
        num_hard         = args.num_hard_neg,
        num_random       = args.num_random_neg,
        tag              = args.tag,
    )


if __name__ == "__main__":
    main()
