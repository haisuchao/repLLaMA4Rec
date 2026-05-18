"""
compute_stats.py
================
Tính thống kê title words và query words cho từng dataset.
Dùng để xác định avg/P95 title words và đề xuất query_max_len phù hợp
khi tăng context_size trong train.sh.

Cách dùng:
  python compute_stats.py                     # tất cả dataset có sẵn
  python compute_stats.py beauty sports       # chỉ các dataset chỉ định
"""

import sys
import numpy as np
from preprocess import preprocess, build_item_text, CONTEXT_SIZE

TOKENS_PER_WORD = 1.3   # hệ số ước lượng tokens từ words (tiếng Anh, Qwen3/LLaMA)
CONTEXT_SIZES   = [3, 5, 10]


def title_stats(all_items_list: list, item_meta: dict) -> dict:
    """Thống kê số từ trong item titles của corpus."""
    counts = [
        len(build_item_text(iid, item_meta).split())
        for iid in all_items_list
    ]
    return {
        "n":   len(counts),
        "avg": round(float(np.mean(counts)), 1),
        "p50": int(np.percentile(counts, 50)),
        "p95": int(np.percentile(counts, 95)),
        "max": max(counts),
    }


def query_stats(sequences: dict, item_meta: dict, context_size: int) -> dict:
    """
    Thống kê số từ trong query với context_size cho trước.
    Dùng train split (predict item N-2): context = seq[:N-3][-context_size:].
    """
    counts = []
    for seq in sequences.values():
        N   = len(seq)
        ctx = seq[: N - 3][-context_size:]
        if ctx:
            text = "Query: " + ", ".join(
                build_item_text(iid, item_meta) for iid in ctx
            ) + " </s>"
        else:
            text = "Query: </s>"
        counts.append(len(text.split()))

    return {
        "avg_words": round(float(np.mean(counts)), 1),
        "p95_words": int(np.percentile(counts, 95)),
        "avg_tokens_est": round(float(np.mean(counts)) * TOKENS_PER_WORD),
        "p95_tokens_est": round(int(np.percentile(counts, 95)) * TOKENS_PER_WORD),
    }


def suggest_query_max_len(p95_tokens: float) -> int:
    """Làm tròn P95 token estimate lên bội số của 32, tối thiểu 128."""
    base = max(128, int(p95_tokens))
    return ((base + 31) // 32) * 32


def run(dataset_name: str):
    print(f"\n{'═' * 58}")
    print(f"  {dataset_name.upper()}")
    print(f"{'═' * 58}")

    _, all_items, item_meta, all_items_set, sequences = preprocess(dataset_name)
    all_items_list = list(all_items_set if isinstance(all_items_set, set) else all_items)

    # ── Title stats ───────────────────────────────────────────────────────────
    ts = title_stats(all_items_list, item_meta)
    print(f"\nItem title word counts  (corpus: {ts['n']:,} items):")
    print(f"  Avg   : {ts['avg']}")
    print(f"  P50   : {ts['p50']}")
    print(f"  P95   : {ts['p95']}")
    print(f"  Max   : {ts['max']}")

    # ── Query stats per context_size ──────────────────────────────────────────
    print(f"\nQuery word counts (train split, {len(sequences):,} users):")
    print(f"  {'context_size':>12}  {'avg_words':>9}  {'p95_words':>9}  "
          f"{'avg_tokens':>10}  {'p95_tokens':>10}  {'query_max_len':>13}")
    print(f"  {'─'*12}  {'─'*9}  {'─'*9}  {'─'*10}  {'─'*10}  {'─'*13}")

    for cs in CONTEXT_SIZES:
        qs  = query_stats(sequences, item_meta, cs)
        rec = suggest_query_max_len(qs["p95_tokens_est"])
        marker = " ← default" if cs == CONTEXT_SIZE else ""
        print(f"  {cs:>12}  {qs['avg_words']:>9}  {qs['p95_words']:>9}  "
              f"{qs['avg_tokens_est']:>10}  {qs['p95_tokens_est']:>10}  "
              f"{rec:>13}{marker}")

    print()


def main():
    from preprocess import DATASETS
    available = list(DATASETS.keys())

    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
        invalid  = [d for d in datasets if d not in available]
        if invalid:
            print(f"Dataset không hợp lệ: {invalid}. Supported: {available}")
            sys.exit(1)
    else:
        # Chỉ chạy các dataset có raw files
        import os
        datasets = []
        for ds, cfg in DATASETS.items():
            if cfg.get("type") == "amazon":
                if os.path.exists(cfg["review_file"]):
                    datasets.append(ds)
            elif cfg.get("type") == "movielens":
                if os.path.exists(os.path.join(cfg["data_dir"], "ratings.dat")):
                    datasets.append(ds)
            elif cfg.get("type") == "steam":
                if os.path.exists(cfg.get("review_file", "")):
                    datasets.append(ds)
        if not datasets:
            print("Không tìm thấy raw data. Truyền tên dataset cụ thể.")
            sys.exit(1)

    for ds in datasets:
        run(ds)


if __name__ == "__main__":
    main()
