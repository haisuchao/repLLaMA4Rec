"""
filter_history.py
=================
Post-processing filter: xóa history items khỏi FAISS retrieval results.

Vấn đề: bi-encoder encode "item1, item2, item3" → query embedding gần với
embedding của item1/2/3 → FAISS trả về history items ở rank 1-2 trong 92.5%
queries → lãng phí 44.6% top-5 slots (đo trên aug-5 best checkpoint).

Script này filter những items đó ra và re-rank phần còn lại từ rank 1.
Không cần train lại — chỉ áp dụng sau FAISS search.

Usage:
  python filter_history.py <dataset> <trec_in> <trec_out>
                           [--split SPLIT] [--context-size N] [--filter-mode MODE]
                           [--no-stats]

Args:
  dataset            : beauty | sports | ml-1m | steam
  trec_in            : TREC run file đầu vào (đã deduplicated, paths relative to project root)
  trec_out           : TREC run file đầu ra (đã filtered)
  --split SPLIT      : valid | test (default: test)
  --context-size N   : số items trong query context (default: CONTEXT_SIZE=3 từ preprocess.py)
                       chỉ quan trọng với filter-mode=context; bỏ qua với filter-mode=full
  --filter-mode MODE : context — chỉ filter các items có trong query text (default)
                       full    — filter toàn bộ items user đã tương tác
  --no-stats         : bỏ qua tính toán waste-slot stats (chạy nhanh hơn)

Output:
  trec_out   : TREC run file đã lọc (history items bị xóa, ranks re-assigned từ 1)
  stdout     : stats về số items bị filter, waste slots trước và sau filter

Ví dụ:
  python filter_history.py beauty \\
    output/beauty/qwen3-embedding-0.6b-aug-5/embeddings/results/test_checkpoint-12000_rank_clean.trec \\
    output/beauty/qwen3-embedding-0.6b-aug-5/embeddings/results/test_checkpoint-12000_rank_filtered_ctx.trec

  python filter_history.py beauty \\
    output/beauty/qwen3-embedding-0.6b-aug-5/embeddings/results/test_checkpoint-12000_rank_clean.trec \\
    output/beauty/qwen3-embedding-0.6b-aug-5/embeddings/results/test_checkpoint-12000_rank_filtered_full.trec \\
    --filter-mode full
"""

import argparse
import os
import sys

# ── Bootstrap: import preprocess từ dataset/ ─────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_DATASET_DIR  = os.path.join(_PROJECT_ROOT, "dataset")
sys.path.insert(0, _DATASET_DIR)

# preprocess.py dùng relative paths ("raw/...") → cần chạy từ dataset/
_ORIG_CWD = os.getcwd()
os.chdir(_DATASET_DIR)

from preprocess import preprocess, CONTEXT_SIZE as DEFAULT_CONTEXT_SIZE

os.chdir(_ORIG_CWD)  # restore sau khi import


# ── Build filter sets ─────────────────────────────────────────────────────────

def build_filter_sets(
    splits: dict,
    sequences: dict,
    split: str,
    context_size: int,
    filter_mode: str,
) -> dict[str, set]:
    """
    Trả về {user_id: set(item_ids cần filter)}.

    filter_mode="context":
      Chỉ filter các items có trong query context (items gây text-overlap bias).
      Số lượng = context_size (hoặc ít hơn nếu cold-start).

    filter_mode="full":
      Filter toàn bộ items user đã tương tác trước positive.
      Tương đương evaluation protocol của SASRec.
    """
    split_data  = splits[split]
    filter_sets = {}

    for user_id, seq in sequences.items():
        N    = len(seq)
        data = split_data.get(user_id)
        if data is None:
            continue

        positive = data["positive"]

        if filter_mode == "context":
            if context_size == DEFAULT_CONTEXT_SIZE:
                # Dùng precomputed tevatron_context (có xử lý cold-start đúng)
                filter_ids = set(data["tevatron_context"])
            else:
                # Tính lại từ sequence với context_size tùy chỉnh
                if split == "test":
                    ctx = seq[: N - 1][-context_size:]
                else:  # valid
                    ctx = seq[: N - 2][-context_size:]
                filter_ids = set(ctx)
        else:  # full
            if split == "test":
                filter_ids = set(seq[: N - 1])
            else:  # valid
                filter_ids = set(seq[: N - 2])

        # positive không nằm trong history theo định nghĩa, nhưng guard phòng thủ
        filter_ids.discard(positive)
        filter_sets[user_id] = filter_ids

    return filter_sets


# ── Filter TREC file ──────────────────────────────────────────────────────────

def filter_trec(
    trec_in: str,
    trec_out: str,
    filter_sets: dict[str, set],
    split: str,
    compute_stats: bool = True,
) -> dict:
    """
    Đọc TREC run file, filter history items, ghi file mới với ranks mới.

    TREC format (space-separated):
      qid  Q0  docid  rank  score  run_name

    Returns stats dict.
    """
    # Đọc toàn bộ file và nhóm theo query
    runs: dict[str, list] = {}
    with open(trec_in) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            qid   = parts[0]
            docid = parts[2]
            rank  = int(parts[3])
            score = parts[4]
            rname = parts[5] if len(parts) > 5 else "dense"
            runs.setdefault(qid, []).append((rank, docid, score, rname))

    suffix         = f"_{split}"
    n_queries      = 0
    total_removed  = 0
    rank1_history  = 0   # số queries mà rank 1 là history item

    waste_top5_list:  list[int] = []
    waste_top10_list: list[int] = []

    with open(trec_out, "w") as f:
        for qid in sorted(runs):
            items = runs[qid]

            # Tách user_id từ qid: strip "_test" hoặc "_valid" suffix
            user_id = qid[: -len(suffix)] if qid.endswith(suffix) else qid

            filter_ids = filter_sets.get(user_id, set())

            # Sort theo rank gốc
            items.sort(key=lambda x: x[0])

            if compute_stats:
                waste5  = sum(1 for r, d, _, _ in items if r <= 5  and d in filter_ids)
                waste10 = sum(1 for r, d, _, _ in items if r <= 10 and d in filter_ids)
                waste_top5_list.append(waste5)
                waste_top10_list.append(waste10)
                if items and items[0][1] in filter_ids:
                    rank1_history += 1

            # Ghi ra file, bỏ history items, re-rank từ 1
            new_rank = 1
            for _, docid, score, rname in items:
                if docid in filter_ids:
                    total_removed += 1
                    continue
                f.write(f"{qid} Q0 {docid} {new_rank} {score} {rname}\n")
                new_rank += 1

            n_queries += 1

    stats: dict = {"n_queries": n_queries, "total_removed": total_removed}

    if compute_stats and n_queries > 0:
        stats["avg_removed"]       = total_removed / n_queries
        stats["rank1_history_pct"] = 100 * rank1_history / n_queries
        stats["avg_waste_top5"]    = sum(waste_top5_list)  / n_queries
        stats["avg_waste_top10"]   = sum(waste_top10_list) / n_queries

        dist: dict[int, int] = {}
        for w in waste_top5_list:
            dist[w] = dist.get(w, 0) + 1
        stats["dist_top5"] = dict(sorted(dist.items()))

    return stats


def print_stats(stats: dict, filter_mode: str, context_size: int) -> None:
    n = stats["n_queries"]
    print(f"\n{'─' * 52}")
    print(f"  Filter Stats  (mode={filter_mode}, context_size={context_size})")
    print(f"{'─' * 52}")
    print(f"  Queries processed         : {n:,}")
    print(f"  Items removed total       : {stats['total_removed']:,}")
    if "avg_removed" in stats:
        print(f"  Avg removed / query       : {stats['avg_removed']:.2f}")
        print()
        print(f"  Rank-1 was history item   : {stats['rank1_history_pct']:.1f}% queries")
        print(f"  Avg waste slots in top-5  : {stats['avg_waste_top5']:.2f} / 5 "
              f"({100*stats['avg_waste_top5']/5:.1f}%)")
        print(f"  Avg waste slots in top-10 : {stats['avg_waste_top10']:.2f} / 10 "
              f"({100*stats['avg_waste_top10']/10:.1f}%)")
        print()
        print(f"  #history items in top-5 distribution:")
        for k, v in stats["dist_top5"].items():
            pct = 100 * v / n
            bar = "█" * max(1, int(pct / 2))
            print(f"    {k} items : {v:>7,} queries  ({pct:5.1f}%)  {bar}")
    print(f"{'─' * 52}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Filter history items từ FAISS retrieval results (post-processing)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset",  help="beauty | sports | ml-1m | steam")
    parser.add_argument("trec_in",  help="Input TREC run file (path relative to project root)")
    parser.add_argument("trec_out", help="Output TREC run file")
    parser.add_argument(
        "--split", default="test", choices=["valid", "test"],
        help="Split tương ứng với TREC file",
    )
    parser.add_argument(
        "--context-size", type=int, default=DEFAULT_CONTEXT_SIZE,
        dest="context_size",
        help=f"Số items trong query context (dùng với --filter-mode context)",
    )
    parser.add_argument(
        "--filter-mode", default="context", choices=["context", "full"],
        dest="filter_mode",
        help="context: filter context items | full: filter toàn bộ history",
    )
    parser.add_argument(
        "--no-stats", action="store_true", dest="no_stats",
        help="Bỏ qua tính toán waste-slot stats (nhanh hơn)",
    )
    args = parser.parse_args()

    # Resolve paths từ project root
    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(_PROJECT_ROOT, p)

    trec_in  = resolve(args.trec_in)
    trec_out = resolve(args.trec_out)

    print(f"\nLoading sequences for '{args.dataset}'...")
    os.chdir(_DATASET_DIR)
    splits, _, _, _, sequences = preprocess(args.dataset)
    os.chdir(_ORIG_CWD)

    print(f"\nBuilding filter sets "
          f"(mode={args.filter_mode}, context_size={args.context_size})...")
    filter_sets = build_filter_sets(
        splits, sequences, args.split, args.context_size, args.filter_mode,
    )

    print(f"Filtering {os.path.basename(trec_in)} ...")
    stats = filter_trec(
        trec_in, trec_out, filter_sets, args.split,
        compute_stats=(not args.no_stats),
    )

    print_stats(stats, args.filter_mode, args.context_size)
    print(f"✓ Output: {trec_out}\n")


if __name__ == "__main__":
    main()
