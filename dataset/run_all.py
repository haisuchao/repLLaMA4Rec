"""
run_all.py
==========
Entry point chạy toàn bộ pipeline cho một hoặc nhiều dataset.
Xuất định dạng standard (1 sample/user, random negatives).

Để augmentation / BM25 negatives, dùng trực tiếp export_tevatron.py:
  python export_tevatron.py beauty --window_size 3
  python export_tevatron.py beauty --neg_strategy mixed
  python export_tevatron.py beauty --window_size 3 --neg_strategy mixed

Cách dùng:
  python run_all.py                        # chạy beauty (mặc định)
  python run_all.py beauty                 # chạy beauty
  python run_all.py beauty sports ml-1m    # chạy nhiều dataset
  python run_all.py --all                  # chạy tất cả dataset

Output:
  dataset/
    tevatron/
      <dataset>/
        corpus.jsonl
        train.jsonl
        valid.jsonl
        test.jsonl
    recbole/
      <dataset>/
        <dataset>.inter
        <dataset>.item
        sasrec_<dataset>.yaml
"""

import sys
import os
import time

from export_tevatron import export_tevatron
from export_recbole   import export_recbole

# ── Config ───────────────────────────────────────────────────────────────────

SUPPORTED_DATASETS = ["beauty", "sports", "ml-1m", "steam"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def print_header(text: str):
    print(f"\n{'#' * 60}")
    print(f"#  {text}")
    print(f"{'#' * 60}")


def check_raw_files(dataset_name: str) -> bool:
    """Kiểm tra raw files có tồn tại không trước khi xử lý."""
    from preprocess import DATASETS
    cfg = DATASETS.get(dataset_name, {})

    missing = []
    if cfg.get("type") == "amazon":
        for key in ["review_file", "meta_file"]:
            if not os.path.exists(cfg[key]):
                missing.append(cfg[key])
    elif cfg.get("type") == "movielens":
        for fname in ["ratings.dat", "movies.dat"]:
            path = os.path.join(cfg["data_dir"], fname)
            if not os.path.exists(path):
                missing.append(path)
    elif cfg.get("type") == "steam":
        for key in ["review_file", "meta_file"]:
            if not os.path.exists(cfg[key]):
                missing.append(cfg[key])

    if missing:
        print(f"\n  [SKIP] '{dataset_name}' — thiếu raw files:")
        for f in missing:
            print(f"    ✗ {f}")
        return False
    return True


def run_dataset(dataset_name: str):
    """Chạy toàn bộ pipeline (Tevatron + RecBole) cho 1 dataset."""
    print_header(f"DATASET: {dataset_name.upper()}")
    t0 = time.time()

    # Kiểm tra raw files
    if not check_raw_files(dataset_name):
        return False

    # Export Tevatron format
    print(f"\n── Tevatron export ──────────────────────────────────────")
    export_tevatron(dataset_name)

    # Export RecBole format
    print(f"\n── RecBole export ───────────────────────────────────────")
    export_recbole(dataset_name)

    elapsed = time.time() - t0
    print(f"\n✓ {dataset_name} hoàn tất ({elapsed:.1f}s)")
    print(f"  → dataset/tevatron/{dataset_name}/")
    print(f"  → dataset/recbole/{dataset_name}/")
    return True


def print_summary(results: dict):
    """In tổng kết sau khi chạy xong."""
    print(f"\n{'=' * 60}")
    print("  TỔNG KẾT")
    print(f"{'=' * 60}")
    for ds, ok in results.items():
        status = "✓ OK  " if ok else "✗ SKIP"
        print(f"  {status}  {ds}")

    # In cấu trúc thư mục output
    print(f"\n{'─' * 60}")
    print("  OUTPUT:")
    for ds, ok in results.items():
        if ok:
            print(f"\n  dataset/tevatron/{ds}/")
            for f in ["corpus.jsonl", "train.jsonl", "valid.jsonl", "test.jsonl"]:
                path = f"dataset/tevatron/{ds}/{f}"
                size = (
                    f"{os.path.getsize(path) / 1024:.1f} KB"
                    if os.path.exists(path) else "N/A"
                )
                print(f"    {f:<20} {size:>10}")
            print(f"\n  dataset/recbole/{ds}/")
            for f in [f"{ds}.inter", f"{ds}.item", f"sasrec_{ds}.yaml"]:
                path = f"dataset/recbole/{ds}/{f}"
                size = (
                    f"{os.path.getsize(path) / 1024:.1f} KB"
                    if os.path.exists(path) else "N/A"
                )
                print(f"    {f:<30} {size:>10}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args:
        datasets = ["beauty"]
    elif "--all" in args:
        datasets = SUPPORTED_DATASETS
    else:
        datasets = args

    # Validate
    invalid = [d for d in datasets if d not in SUPPORTED_DATASETS]
    if invalid:
        print(f"Dataset không hợp lệ: {invalid}")
        print(f"Supported: {SUPPORTED_DATASETS}")
        sys.exit(1)

    results = {}
    total_t0 = time.time()

    for ds in datasets:
        results[ds] = run_dataset(ds)

    total_elapsed = time.time() - total_t0
    print_summary(results)
    print(f"Tổng thời gian: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
