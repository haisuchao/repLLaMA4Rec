"""
show_results.py
===============
Tự động đọc kết quả từ output/ và hiển thị bảng so sánh các experiment.

Cách dùng:
  python show_results.py                        # In bảng ra terminal
  python show_results.py --update-experiments   # In + cập nhật experiments.md
  python show_results.py --update-readme        # In + cập nhật README.md (legacy)

Quy ước thư mục:
  output/<dataset>/<model_tag>/embeddings/results/eval_test.txt
  output/<dataset>/sasrec/eval_test.txt   (SASRec — tạo thủ công từ RecBole log)
"""

import re
import argparse
from pathlib import Path
from collections import defaultdict

# ── Config ───────────────────────────────────────────────────────────────────

METRICS_DISPLAY = ["ndcg_5", "hr_5", "ndcg_10", "hr_10", "ndcg_20", "hr_20", "mrr_10"]
METRIC_LABELS   = {
    "ndcg_5":  "NDCG@5",
    "hr_5":    "HR@5",
    "ndcg_10": "NDCG@10",
    "hr_10":   "HR@10",
    "ndcg_20": "NDCG@20",
    "hr_20":   "HR@20",
    "mrr_10":  "MRR@10",
}

DATASET_ORDER = ["beauty", "sports", "ml-1m", "steam"]

# Nhãn hiển thị — thêm variant mới vào đây
MODEL_LABELS = {
    "sasrec":                          "SASRec",
    "qwen3-embedding-0.6b-zeroshot":   "Qwen3-0.6B · zero-shot",
    "qwen3-embedding-0.6b":            "Qwen3-0.6B · FT",
    "qwen3-embedding-0.6b-aug":        "Qwen3-0.6B · Aug-8neg",
    "qwen3-embedding-0.6b-aug-3":      "Qwen3-0.6B · Aug3-8neg",
    "qwen3-embedding-0.6b-aug-w5":     "Qwen3-0.6B · Aug-w5-8neg",
    "qwen3-embedding-0.6b-reversed":   "Qwen3-0.6B · reversed",
    "qwen3-embedding-0.6b-random":     "Qwen3-0.6B · random",
    "qwen3-embedding-4b-zeroshot":     "Qwen3-4B · zero-shot",
    "qwen3-embedding-4b":              "Qwen3-4B · FT",
    "qwen2.5-3b-zeroshot":             "Qwen2.5-3B · zero-shot",
    "qwen2.5-3b":                      "Qwen2.5-3B · FT",
    "llama-3.2-1b-zeroshot":           "Llama-3.2-1B · zero-shot",
    "llama-3.2-1b":                    "Llama-3.2-1B · FT-8neg",
    "cnnmamba4recv2":                  "CMamba4Rec",
}

# Thứ tự ưu tiên hiển thị trong bảng
MODEL_ORDER = [
    "sasrec",
    "qwen3-embedding-0.6b-zeroshot",
    "qwen3-embedding-0.6b",
    "qwen3-embedding-0.6b-aug",
    "qwen3-embedding-0.6b-aug-3",
    "qwen3-embedding-0.6b-aug-w5",
    "qwen3-embedding-0.6b-reversed",
    "qwen3-embedding-0.6b-random",
    "qwen3-embedding-4b-zeroshot",
    "qwen3-embedding-4b",
    "qwen2.5-3b-zeroshot",
    "qwen2.5-3b",
    "llama-3.2-1b-zeroshot",
    "llama-3.2-1b",
]

RESULTS_MARKER_START = "<!-- RESULTS_START -->"
RESULTS_MARKER_END   = "<!-- RESULTS_END -->"

# Legacy alias
README_MARKER_START = RESULTS_MARKER_START
README_MARKER_END   = RESULTS_MARKER_END


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_eval_file(path: Path) -> dict:
    """
    Parse eval_*.txt (output của compute_metrics.py).
    Lấy lần xuất hiện CUỐI của mỗi metric để xử lý file bị append nhiều lần.
    Format mỗi dòng: <metric_name>  all  <value>
    """
    metrics = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[1] == "all":
                try:
                    metrics[parts[0]] = float(parts[2])
                except ValueError:
                    pass
    return metrics


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_results(output_dir: str = "./output") -> dict:
    """
    Duyệt output/ và thu thập tất cả eval_test.txt.
    Trả về: {(dataset, model_tag): {metric: value}}
    """
    results = {}
    root = Path(output_dir)
    if not root.exists():
        return results

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_tag = model_dir.name

            results_dir = model_dir / "embeddings" / "results"
            # Ưu tiên: best → latest → base (zero-shot) → manual → SASRec flat
            candidates = [
                results_dir / "eval_test_best.txt",
                results_dir / "eval_test_latest.txt",
                results_dir / "eval_test_base.txt",
                results_dir / "eval_test.txt",
                model_dir / "eval_test.txt",
            ]
            eval_file = next((f for f in candidates if f.exists()), None)

            if eval_file:
                metrics = parse_eval_file(eval_file)
                if any(m in metrics for m in METRICS_DISPLAY):
                    results[(dataset, model_tag)] = metrics

    return results


# ── Formatting ────────────────────────────────────────────────────────────────

def _model_sort_key(tag: str) -> tuple:
    try:
        return (MODEL_ORDER.index(tag), tag)
    except ValueError:
        return (len(MODEL_ORDER), tag)


def _get_label(tag: str) -> str:
    return MODEL_LABELS.get(tag, tag)


def format_markdown(results: dict) -> str:
    """Tạo markdown table từ results dict."""
    if not results:
        return "_Chưa có kết quả._\n"

    by_dataset = defaultdict(dict)
    for (dataset, model_tag), metrics in results.items():
        by_dataset[dataset][model_tag] = metrics

    present = [d for d in DATASET_ORDER if d in by_dataset]
    present += [d for d in sorted(by_dataset) if d not in present]

    lines = []
    for dataset in present:
        lines.append(f"### {dataset.upper()}\n")

        header = "| Model | " + " | ".join(METRIC_LABELS[m] for m in METRICS_DISPLAY) + " |"
        sep    = "| :--- | " + " | ".join([":---:"] * len(METRICS_DISPLAY)) + " |"
        lines += [header, sep]

        for tag in sorted(by_dataset[dataset], key=_model_sort_key):
            vals = [
                f"{by_dataset[dataset][tag][m]:.4f}" if m in by_dataset[dataset][tag] else "—"
                for m in METRICS_DISPLAY
            ]
            lines.append(f"| {_get_label(tag)} | " + " | ".join(vals) + " |")

        lines.append("")

    return "\n".join(lines)


def format_terminal(results: dict) -> str:
    """Tạo bảng terminal từ results dict."""
    if not results:
        return "Chưa có kết quả nào trong output/\n"

    by_dataset = defaultdict(dict)
    for (dataset, model_tag), metrics in results.items():
        by_dataset[dataset][model_tag] = metrics

    present = [d for d in DATASET_ORDER if d in by_dataset]
    present += [d for d in sorted(by_dataset) if d not in present]

    col = 32
    header_row = f"{'Model':<{col}}" + "".join(f"  {METRIC_LABELS[m]:>8}" for m in METRICS_DISPLAY)
    width = len(header_row)

    lines = []
    for dataset in present:
        lines.append(f"\n{'═' * width}")
        lines.append(f"  {dataset.upper()}")
        lines.append(f"{'─' * width}")
        lines.append(header_row)
        lines.append(f"{'─' * width}")

        for tag in sorted(by_dataset[dataset], key=_model_sort_key):
            label = _get_label(tag)[:col - 1]
            row = f"{label:<{col}}"
            for m in METRICS_DISPLAY:
                v = by_dataset[dataset][tag].get(m)
                row += f"  {v:>8.4f}" if v is not None else f"  {'—':>8}"
            lines.append(row)

    lines.append(f"{'═' * width}\n")
    return "\n".join(lines)


# ── README updater ────────────────────────────────────────────────────────────

def update_file(table_md: str, file_path: str):
    """Cập nhật phần kết quả giữa RESULTS_START/END markers trong file chỉ định."""
    path = Path(file_path)
    content = path.read_text()

    inner = f"\n{table_md}\n"
    new_block = f"{RESULTS_MARKER_START}{inner}{RESULTS_MARKER_END}"

    if RESULTS_MARKER_START in content and RESULTS_MARKER_END in content:
        pattern = re.escape(RESULTS_MARKER_START) + r".*?" + re.escape(RESULTS_MARKER_END)
        new_content = re.sub(pattern, new_block, content, flags=re.DOTALL)
    else:
        print(f"  Cảnh báo: Không tìm thấy markers trong {file_path}. Thêm vào cuối.")
        new_content = content.rstrip() + f"\n\n{new_block}\n"

    path.write_text(new_content)
    print(f"✓ {file_path} đã được cập nhật.")


def update_readme(table_md: str, readme_path: str = "README.md"):
    """Legacy wrapper — giữ backward compatibility."""
    update_file(table_md, readme_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hiển thị và cập nhật bảng kết quả thực nghiệm")
    parser.add_argument("--update-experiments", action="store_true",
                        help="Cập nhật bảng kết quả trong experiments.md (khuyến nghị)")
    parser.add_argument("--update-readme", action="store_true",
                        help="Cập nhật bảng kết quả trong README.md (legacy)")
    parser.add_argument("--output-dir", default="./output",
                        help="Thư mục chứa kết quả (mặc định: ./output)")
    parser.add_argument("--file", default=None,
                        help="Đường dẫn file cần cập nhật (override mặc định)")
    args = parser.parse_args()

    results = discover_results(args.output_dir)

    print(format_terminal(results))

    if args.update_experiments or args.update_readme:
        if args.file:
            target = args.file
        elif args.update_readme:
            target = "README.md"
        else:
            target = "experiments.md"
        update_file(format_markdown(results), target)
    else:
        print("(Dùng --update-experiments để cập nhật experiments.md tự động)")


if __name__ == "__main__":
    main()
