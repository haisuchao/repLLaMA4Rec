"""
eval_filter.py
==============
Áp dụng history filter lên TREC results đã có từ eval.sh.

Không cần encode lại hay FAISS search lại — chỉ đọc rank_clean.trec hiện có,
filter history items, tính metrics, ghi kết quả ra file riêng.

Kết quả:
  eval_{split}_{label}_filtered_{mode}.txt   — filtered metrics
  {split}_{label}_rank_filtered_{mode}.trec  — filtered TREC run (để reranker dùng)

Usage:
  python eval_filter.py <dataset> [options]

Options:
  --model MODEL           HuggingFace model ID (default: Qwen/Qwen3-Embedding-0.6B)
  --tag TAG               Model tag để chọn thư mục output (ví dụ: aug-5)
  --split SPLIT           valid | test (default: test)
  --checkpoint CKPT       best | latest | checkpoint-N (default: best)
  --filter-mode MODE      context — filter items có trong query (default)
                          full    — filter toàn bộ history (như SASRec)
  --context-size N        Số items trong query context (default: 3, dùng với mode=context)

Ví dụ:
  # Filter best model với mode context (mặc định)
  python eval_filter.py beauty --tag aug-5

  # So sánh context vs full filter
  python eval_filter.py beauty --tag aug-5 --filter-mode full

  # Filter model cụ thể, split valid
  python eval_filter.py beauty --tag aug-5 --checkpoint checkpoint-8000 --split valid

  # Filter tất cả mode để so sánh
  python eval_filter.py beauty --tag aug-5 --filter-mode context
  python eval_filter.py beauty --tag aug-5 --filter-mode full
"""

import argparse
import math
import os
import sys
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_DATASET_DIR  = os.path.join(_PROJECT_ROOT, "dataset")
sys.path.insert(0, _DATASET_DIR)

# preprocess.py dùng relative paths → cần CWD = dataset/
_ORIG_CWD = os.getcwd()
os.chdir(_DATASET_DIR)
from preprocess import preprocess, CONTEXT_SIZE as DEFAULT_CONTEXT_SIZE
os.chdir(_ORIG_CWD)

# Import filter logic từ filter_history.py
sys.path.insert(0, _PROJECT_ROOT)
from filter_history import build_filter_sets, filter_trec, print_stats


# ── Metrics (inline — tránh phụ thuộc CLI của compute_metrics.py) ─────────────

def load_qrels(path: str) -> dict[str, set]:
    qrels = defaultdict(set)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
            if int(rel) > 0:
                qrels[qid].add(docid)
    return dict(qrels)


def load_run(path: str) -> dict[str, list]:
    runs = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, docid, rank = parts[0], parts[2], int(parts[3])
            runs[qid].append((rank, docid))
    for qid in runs:
        runs[qid].sort(key=lambda x: x[0])
    return dict(runs)


def compute_metrics(run: dict, qrels: dict, ks: list[int]) -> dict[str, float]:
    results = {f"{m}_{k}": 0.0 for k in ks for m in ("ndcg", "hr", "mrr")}
    n = len(qrels)
    if n == 0:
        return results
    for qid, relevant in qrels.items():
        ranked = run.get(qid, [])
        hit_rank = next((r for r, d in ranked if d in relevant), None)
        for k in ks:
            if hit_rank is not None and hit_rank <= k:
                results[f"hr_{k}"]   += 1.0
                results[f"mrr_{k}"]  += 1.0 / hit_rank
                results[f"ndcg_{k}"] += 1.0 / math.log2(hit_rank + 1)
    for key in results:
        results[key] /= n
    return results


def format_metrics(m: dict, ks: list[int]) -> str:
    lines = []
    for k in sorted(ks):
        for metric in ("ndcg", "hr", "mrr"):
            key = f"{metric}_{k}"
            lines.append(f"{key:<20s}\tall\t{m[key]:.4f}")
    return "\n".join(lines)


def print_comparison(unfiltered: dict, filtered: dict, ks: list[int], mode: str) -> None:
    """In bảng so sánh unfiltered vs filtered."""
    print(f"\n{'═' * 60}")
    print(f"  Comparison: Unfiltered vs Filtered (mode={mode})")
    print(f"{'═' * 60}")
    print(f"  {'Metric':<18}  {'Unfiltered':>12}  {'Filtered':>10}  {'Δ':>8}")
    print(f"  {'─' * 56}")
    for k in sorted(ks):
        for metric in ("ndcg", "hr", "mrr"):
            key  = f"{metric}_{k}"
            u    = unfiltered[key]
            filt = filtered[key]
            delta = filt - u
            sign  = "+" if delta >= 0 else ""
            print(f"  {key:<18}  {u:>12.4f}  {filt:>10.4f}  {sign}{delta:>7.4f}")
    print(f"{'═' * 60}\n")


# ── Path helpers ──────────────────────────────────────────────────────────────

def get_model_tag(model: str) -> str:
    return os.path.basename(model).lower()


def get_results_dir(dataset: str, model: str, tag: str) -> str:
    model_tag = get_model_tag(model)
    full_tag  = f"{model_tag}-{tag}" if tag else model_tag
    return os.path.join(_PROJECT_ROOT, "output", dataset, full_tag, "embeddings", "results")


def get_best_label(results_dir: str, split: str) -> str:
    """Đọc label của best checkpoint từ eval_{split}_best.txt."""
    best_file = os.path.join(results_dir, f"eval_{split}_best.txt")
    if not os.path.exists(best_file):
        raise FileNotFoundError(
            f"Không tìm thấy {best_file}.\n"
            f"Hãy chạy './eval.sh {os.path.basename(os.path.dirname(os.path.dirname(results_dir)))}' trước."
        )
    with open(best_file) as f:
        for line in f:
            if line.startswith("Best checkpoint :"):
                return line.split(":", 1)[1].strip()
    raise RuntimeError(f"Không tìm thấy 'Best checkpoint' trong {best_file}")


def get_label_for_checkpoint(checkpoint: str, dataset: str, model: str, tag: str) -> str:
    """Chuyển checkpoint argument → label (tên file TREC)."""
    if checkpoint == "latest":
        # label khi eval.sh chạy với checkpoint=latest
        return "latest"
    elif checkpoint == "best":
        results_dir = get_results_dir(dataset, model, tag)
        return get_best_label(results_dir, split="test")  # split hinted later
    else:
        return checkpoint  # "checkpoint-N"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Áp dụng history filter lên TREC results đã có từ eval.sh",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", help="beauty | sports | ml-1m | steam")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-Embedding-0.6B",
        help="HuggingFace model ID",
    )
    parser.add_argument("--tag",         default="",     help="Model tag (ví dụ: aug-5)")
    parser.add_argument("--split",       default="test", choices=["valid", "test"])
    parser.add_argument(
        "--checkpoint", default="best",
        help="best | latest | checkpoint-N",
    )
    parser.add_argument(
        "--filter-mode", default="context", choices=["context", "full"],
        dest="filter_mode",
        help="context: filter query items | full: filter toàn bộ history",
    )
    parser.add_argument(
        "--context-size", type=int, default=DEFAULT_CONTEXT_SIZE,
        dest="context_size",
        help="Số items trong query context (dùng với mode=context)",
    )
    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    results_dir = get_results_dir(args.dataset, args.model, args.tag)

    if not os.path.isdir(results_dir):
        print(f"Lỗi: Không tìm thấy results dir: {results_dir}")
        print(f"Hãy chạy eval.sh trước.")
        sys.exit(1)

    # Với checkpoint=best: cần biết split để đọc file đúng
    if args.checkpoint == "best":
        label = get_best_label(results_dir, args.split)
    elif args.checkpoint == "latest":
        label = "latest"
    else:
        label = args.checkpoint

    mode_suffix = "ctx" if args.filter_mode == "context" else "full"

    rank_clean    = os.path.join(results_dir, f"{args.split}_{label}_rank_clean.trec")
    rank_filtered = os.path.join(results_dir, f"{args.split}_{label}_rank_filtered_{mode_suffix}.trec")
    qrels_clean   = os.path.join(results_dir, f"{args.split}_qrels_clean.txt")
    eval_out      = os.path.join(results_dir, f"eval_{args.split}_{label}.txt")
    eval_filtered = os.path.join(results_dir, f"eval_{args.split}_{label}_filtered_{mode_suffix}.txt")

    # Khi best mode: tìm file bằng label vừa lấy
    if not os.path.exists(rank_clean):
        # Thử fallback: rank_clean tên theo "best" (không phải checkpoint cụ thể)
        fallback = os.path.join(results_dir, f"{args.split}_best_rank_clean.trec")
        if os.path.exists(fallback):
            rank_clean    = fallback
            rank_filtered = os.path.join(results_dir, f"{args.split}_best_rank_filtered_{mode_suffix}.trec")
            eval_filtered = os.path.join(results_dir, f"eval_{args.split}_best_filtered_{mode_suffix}.txt")
            label         = "best"
        else:
            # List available files để guide user
            available = [f for f in os.listdir(results_dir)
                         if f.endswith("_rank_clean.trec") and f.startswith(args.split)]
            print(f"Lỗi: Không tìm thấy {rank_clean}")
            print(f"\nFiles available trong {results_dir}:")
            for fn in sorted(available):
                print(f"  {fn}")
            sys.exit(1)

    if not os.path.exists(qrels_clean):
        print(f"Lỗi: Không tìm thấy qrels file: {qrels_clean}")
        print(f"Hãy chạy eval.sh --split {args.split} trước.")
        sys.exit(1)

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 56}")
    print(f"  eval_filter.py")
    print(f"{'═' * 56}")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Tag          : {args.tag or '(standard)'}")
    print(f"  Split        : {args.split}")
    print(f"  Checkpoint   : {label}")
    print(f"  Filter mode  : {args.filter_mode}")
    if args.filter_mode == "context":
        print(f"  Context size : {args.context_size}")
    print(f"  TREC input   : {os.path.basename(rank_clean)}")
    print(f"{'═' * 56}\n")

    # ── Load sequences ────────────────────────────────────────────────────────
    print(f"Loading sequences for '{args.dataset}'...")
    os.chdir(_DATASET_DIR)
    splits, _, _, _, sequences = preprocess(args.dataset)
    os.chdir(_ORIG_CWD)

    # ── Build filter sets ─────────────────────────────────────────────────────
    print(f"\nBuilding filter sets (mode={args.filter_mode})...")
    filter_sets = build_filter_sets(
        splits, sequences, args.split, args.context_size, args.filter_mode,
    )

    # ── Filter TREC ───────────────────────────────────────────────────────────
    print(f"Filtering {os.path.basename(rank_clean)} ...")
    stats = filter_trec(
        rank_clean, rank_filtered, filter_sets, args.split,
        compute_stats=True,
    )
    print_stats(stats, args.filter_mode, args.context_size)

    # ── Compute metrics ───────────────────────────────────────────────────────
    ks      = [5, 10, 20]
    qrels   = load_qrels(qrels_clean)

    # Unfiltered
    run_u   = load_run(rank_clean)
    m_u     = compute_metrics(run_u, qrels, ks)

    # Filtered
    run_f   = load_run(rank_filtered)
    m_f     = compute_metrics(run_f, qrels, ks)

    # ── Print & save ──────────────────────────────────────────────────────────
    print_comparison(m_u, m_f, ks, args.filter_mode)

    # Ghi filtered metrics ra file
    with open(eval_filtered, "w") as fp:
        fp.write(f"# Filter mode: {args.filter_mode}")
        if args.filter_mode == "context":
            fp.write(f"  context_size={args.context_size}")
        fp.write(f"\n# Checkpoint: {label}  Split: {args.split}\n\n")
        fp.write(format_metrics(m_f, ks))
        fp.write("\n")
    print(f"✓ Unfiltered metrics : {eval_out}")
    print(f"✓ Filtered metrics   : {eval_filtered}")
    print(f"✓ Filtered TREC      : {rank_filtered}\n")


if __name__ == "__main__":
    main()
