"""
show_results.py
===============
Tự động đọc kết quả từ output/ và hiển thị bảng so sánh các experiment.

Nhãn và thứ tự hiển thị được sinh tự động từ train_config.json (lưu bởi train.sh).
Không cần cập nhật MODEL_LABELS / MODEL_ORDER thủ công khi thêm experiment mới.

Cách dùng:
  python show_results.py                        # In bảng ra terminal
  python show_results.py --update-experiments   # In + cập nhật results table
                                                # và auto-exp descriptions trong experiments.md
  python show_results.py --update-readme        # In + cập nhật README.md (legacy)

Cấu trúc thư mục output:
  output/<dataset>/<model_tag>/
    train_config.json              ← sinh bởi train.sh, dùng cho auto-label
    embeddings/results/
      eval_test_best.txt           ← kết quả chính (ưu tiên 1)
      eval_test_latest.txt         ← (ưu tiên 2)
      eval_test_base.txt           ← (ưu tiên 3)
      eval_test.txt                ← (ưu tiên 4)
    eval_test.txt                  ← SASRec / manual (ưu tiên 5, flat)
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

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

DEFAULT_MODEL      = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_GROUP_SIZE = 8

RESULTS_MARKER_START  = "<!-- RESULTS_START -->"
RESULTS_MARKER_END    = "<!-- RESULTS_END -->"
AUTO_EXP_MARKER_START = "<!-- AUTO_EXP_START -->"
AUTO_EXP_MARKER_END   = "<!-- AUTO_EXP_END -->"

# Legacy alias
README_MARKER_START = RESULTS_MARKER_START
README_MARKER_END   = RESULTS_MARKER_END


# ── File readers ──────────────────────────────────────────────────────────────

def parse_eval_file(path: Path) -> dict:
    """
    Parse metric lines từ eval_*.txt.
    Lấy lần xuất hiện CUỐI của mỗi metric để xử lý file bị append nhiều lần.
    Format: <metric_name>  all  <value>
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


def parse_eval_meta(path: Path) -> dict:
    """Parse metadata từ eval_*.txt (best checkpoint, selection metric)."""
    meta = {}
    with open(path) as f:
        for line in f:
            if line.startswith("Best checkpoint"):
                meta["best_checkpoint"] = line.split(":", 1)[1].strip()
            elif line.startswith("Selection metric"):
                meta["selection_metric"] = line.split(":", 1)[1].strip()
    return meta


def load_train_config(model_dir: Path) -> dict | None:
    """Đọc train_config.json nếu tồn tại — được lưu bởi train.sh."""
    path = model_dir / "train_config.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return None


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_all(output_dir: str = "./output") -> dict:
    """
    Duyệt output/ và thu thập tất cả kết quả + config.

    Returns:
      {
        (dataset, model_tag): {
          "metrics":   {metric: float},
          "eval_meta": {key: str},      # best_checkpoint, selection_metric
          "config":    dict | None,     # train_config.json (None nếu không có)
          "mtime":     float,           # directory mtime — dùng để sort fallback
          "model_dir": Path,
        }
      }
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

            # Eval file theo thứ tự ưu tiên
            results_dir = model_dir / "embeddings" / "results"
            candidates  = [
                results_dir / "eval_test_best.txt",
                results_dir / "eval_test_latest.txt",
                results_dir / "eval_test_base.txt",
                results_dir / "eval_test.txt",
                model_dir   / "eval_test.txt",
            ]
            eval_file = next((f for f in candidates if f.exists()), None)

            metrics   = parse_eval_file(eval_file)  if eval_file else {}
            eval_meta = parse_eval_meta(eval_file)  if eval_file else {}
            config    = load_train_config(model_dir)
            mtime     = model_dir.stat().st_mtime

            # Chỉ include nếu có metrics hoặc có config
            # (config không có metrics = đang train, chưa eval — vẫn hiện trong auto-exp)
            if any(m in metrics for m in METRICS_DISPLAY) or config:
                results[(dataset, model_tag)] = {
                    "metrics":   metrics,
                    "eval_meta": eval_meta,
                    "config":    config,
                    "mtime":     mtime,
                    "model_dir": model_dir,
                }

    return results


# ── Label & sort ──────────────────────────────────────────────────────────────

def auto_label(model_tag: str, config: dict | None) -> str:
    """
    Sinh nhãn hiển thị từ model_tag và train_config.json.

    Ưu tiên:
      1. Có config → "<model_basename> · <tag>" hoặc "<model_basename>"
      2. Không có config, tên kết thúc bằng -zeroshot → parse pattern
      3. Fallback → tên thư mục nguyên gốc
    """
    if model_tag == "sasrec":
        return "SASRec"

    if config:
        model_name = config["model"].split("/")[-1]  # basename của HF model ID
        tag        = config.get("tag", "")
        return f"{model_name} · {tag}" if tag else model_name

    # Fallback: pattern matching (experiments cũ không có train_config.json)
    if model_tag.endswith("-zeroshot"):
        return model_tag[: -len("-zeroshot")] + " · zero-shot"

    return model_tag


def auto_sort_key(model_tag: str, info: dict) -> tuple:
    """
    Thứ tự hiển thị trong bảng:
      0. sasrec (luôn đầu — baseline dễ so sánh)
      1. experiments có config → sort theo (model_basename, tag) alphabet
      2. experiments không config → sort theo tên thư mục alphabet
    """
    if model_tag == "sasrec":
        return (0, "", "")
    config = info.get("config")
    if config:
        model_name = config["model"].split("/")[-1].lower()
        tag        = config.get("tag", "").lower()
        return (1, model_name, tag)
    return (2, model_tag.lower(), "")


def make_anchor_id(dataset: str, model_tag: str) -> str:
    """Sinh anchor ID ổn định cho liên kết bảng → mô tả experiment."""
    return f"exp-{dataset}-{model_tag}"


# ── Command builders ──────────────────────────────────────────────────────────

def build_train_command(config: dict) -> str:
    """Tái tạo lệnh train.sh từ train_config.json."""
    parts = [f"./train.sh {config['dataset']}"]
    if config.get("model", DEFAULT_MODEL) != DEFAULT_MODEL:
        parts.append(f"--model {config['model']}")
    dv  = config.get("data_variant", "")
    tag = config.get("tag", "")
    if dv:
        parts.append(f"--data-variant {dv}")
    if tag and tag != dv:
        parts.append(f"--tag {tag}")
    if config.get("train_group_size", DEFAULT_GROUP_SIZE) != DEFAULT_GROUP_SIZE:
        parts.append(f"--group-size {config['train_group_size']}")
    return " ".join(parts)


def build_eval_command(config: dict) -> str:
    """Tái tạo lệnh eval.sh từ train_config.json."""
    parts = [f"./eval.sh {config['dataset']}"]
    if config.get("model", DEFAULT_MODEL) != DEFAULT_MODEL:
        parts.append(f"--model {config['model']}")
    tag = config.get("tag", "")
    if tag:
        parts.append(f"--tag {tag}")
    return " ".join(parts)


# ── Formatters ────────────────────────────────────────────────────────────────

def format_markdown(all_results: dict) -> str:
    """Tạo markdown results table."""
    entries = {
        k: v for k, v in all_results.items()
        if any(m in v["metrics"] for m in METRICS_DISPLAY)
    }
    if not entries:
        return "_Chưa có kết quả._\n"

    by_dataset: dict[str, dict] = defaultdict(dict)
    for (dataset, model_tag), info in entries.items():
        by_dataset[dataset][model_tag] = info

    present = [d for d in DATASET_ORDER if d in by_dataset]
    present += [d for d in sorted(by_dataset) if d not in present]

    lines = []
    for dataset in present:
        lines.append(f"### {dataset.upper()}\n")
        header = "| Model | " + " | ".join(METRIC_LABELS[m] for m in METRICS_DISPLAY) + " |"
        sep    = "| :--- | " + " | ".join([":---:"] * len(METRICS_DISPLAY)) + " |"
        lines += [header, sep]

        tags_sorted = sorted(
            by_dataset[dataset],
            key=lambda t: auto_sort_key(t, by_dataset[dataset][t]),
        )
        for tag in tags_sorted:
            info   = by_dataset[dataset][tag]
            label  = auto_label(tag, info["config"])
            anchor = make_anchor_id(dataset, tag)
            vals   = [
                f"{info['metrics'][m]:.4f}" if m in info["metrics"] else "—"
                for m in METRICS_DISPLAY
            ]
            lines.append(f"| [{label}](#{anchor}) | " + " | ".join(vals) + " |")

        lines.append("")

    return "\n".join(lines)


def format_terminal(all_results: dict) -> str:
    """Tạo bảng terminal."""
    entries = {
        k: v for k, v in all_results.items()
        if any(m in v["metrics"] for m in METRICS_DISPLAY)
    }
    if not entries:
        return "Chưa có kết quả nào trong output/\n"

    by_dataset: dict[str, dict] = defaultdict(dict)
    for (dataset, model_tag), info in entries.items():
        by_dataset[dataset][model_tag] = info

    present = [d for d in DATASET_ORDER if d in by_dataset]
    present += [d for d in sorted(by_dataset) if d not in present]

    col        = 36
    header_row = f"{'Model':<{col}}" + "".join(f"  {METRIC_LABELS[m]:>8}" for m in METRICS_DISPLAY)
    width      = len(header_row)

    lines = []
    for dataset in present:
        lines.append(f"\n{'═' * width}")
        lines.append(f"  {dataset.upper()}")
        lines.append(f"{'─' * width}")
        lines.append(header_row)
        lines.append(f"{'─' * width}")

        tags_sorted = sorted(
            by_dataset[dataset],
            key=lambda t: auto_sort_key(t, by_dataset[dataset][t]),
        )
        for tag in tags_sorted:
            info  = by_dataset[dataset][tag]
            label = auto_label(tag, info["config"])[: col - 1]
            row   = f"{label:<{col}}"
            for m in METRICS_DISPLAY:
                v = info["metrics"].get(m)
                row += f"  {v:>8.4f}" if v is not None else f"  {'—':>8}"
            lines.append(row)

    lines.append(f"{'═' * width}\n")
    return "\n".join(lines)


def format_auto_experiments(all_results: dict) -> str:
    """
    Sinh mô tả markdown cho tất cả experiment có train_config.json.
    Nhóm theo dataset, sort theo timestamp (oldest first).
    Bao gồm cả experiment đang train (chưa có metrics).
    """
    by_dataset: dict[str, dict] = defaultdict(dict)
    for (dataset, model_tag), info in all_results.items():
        if info["config"]:
            by_dataset[dataset][model_tag] = info

    if not by_dataset:
        return "_Chưa có experiment nào có train_config.json. Chạy train.sh để tự sinh._\n"

    present = [d for d in DATASET_ORDER if d in by_dataset]
    present += [d for d in sorted(by_dataset) if d not in present]

    lines = []
    for dataset in present:
        lines.append(f"### {dataset.upper()}\n")

        tags_sorted = sorted(
            by_dataset[dataset],
            key=lambda t: auto_sort_key(t, by_dataset[dataset][t]),
        )

        for model_tag in tags_sorted:
            info   = by_dataset[dataset][model_tag]
            config = info["config"]
            meta   = info.get("eval_meta", {})
            label  = auto_label(model_tag, config)

            anchor = make_anchor_id(dataset, model_tag)
            lines.append(f'<a id="{anchor}"></a>')
            lines.append(f"#### {label}\n")

            # Config table
            lines.append("| Thuộc tính | Giá trị |")
            lines.append("|---|---|")
            lines.append(f"| Base model | `{config['model']}` |")
            lines.append(f"| Dataset | {dataset} |")

            dv = config.get("data_variant") or "—"
            lines.append(f"| Data variant | {dv} |")

            gs = config.get("train_group_size", DEFAULT_GROUP_SIZE)
            lines.append(f"| train_group_size | {gs} (1 positive + {gs - 1} negatives) |")

            b = config.get("per_device_batch", 4)
            g = config.get("gradient_accumulation", 8)
            lines.append(f"| per_device_batch | {b} |")
            lines.append(f"| gradient_accumulation | {g} (effective batch = {b * g}) |")
            lines.append(f"| Learning rate | {config.get('learning_rate', '1e-4')} |")
            lines.append(f"| Epochs | {config.get('epochs', 3)} |")
            lines.append(f"| Save steps | {config.get('save_steps', 2000)} |")
            lines.append(f"| Query max len | {config.get('query_max_len', 128)} |")
            lines.append(f"| Passage max len | {config.get('passage_max_len', 196)} |")

            if meta.get("best_checkpoint"):
                lines.append(f"| Best checkpoint | {meta['best_checkpoint']} |")
            if meta.get("selection_metric"):
                lines.append(f"| Selection metric | {meta['selection_metric']} |")

            lines.append(f"| Trained at | {config.get('timestamp', '—')} |")
            lines.append("")

            # Reconstructed commands
            lines.append("```bash")
            lines.append(build_train_command(config))
            lines.append(build_eval_command(config))
            lines.append("```")
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


# ── File updater ──────────────────────────────────────────────────────────────

def update_file(
    content: str,
    file_path: str,
    start_marker: str = RESULTS_MARKER_START,
    end_marker:   str = RESULTS_MARKER_END,
):
    """Cập nhật nội dung giữa start_marker và end_marker trong file."""
    path      = Path(file_path)
    text      = path.read_text()
    inner     = f"\n{content}\n"
    new_block = f"{start_marker}{inner}{end_marker}"

    if start_marker in text and end_marker in text:
        pattern     = re.escape(start_marker) + r".*?" + re.escape(end_marker)
        new_content = re.sub(pattern, new_block, text, flags=re.DOTALL)
    else:
        print(f"  Cảnh báo: Không tìm thấy markers '{start_marker}' trong {file_path}. Thêm vào cuối.")
        new_content = text.rstrip() + f"\n\n{new_block}\n"

    path.write_text(new_content)
    print(f"✓ {file_path} đã được cập nhật.")


def update_readme(table_md: str, readme_path: str = "README.md"):
    """Legacy wrapper — backward compatibility."""
    update_file(table_md, readme_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hiển thị và cập nhật bảng kết quả thực nghiệm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--update-experiments", action="store_true",
        help="Cập nhật results table VÀ auto-exp descriptions trong experiments.md",
    )
    parser.add_argument(
        "--update-readme", action="store_true",
        help="Cập nhật results table trong README.md (legacy)",
    )
    parser.add_argument(
        "--output-dir", default="./output",
        help="Thư mục chứa kết quả",
    )
    parser.add_argument(
        "--file", default=None,
        help="Đường dẫn file cần cập nhật (override mặc định)",
    )
    args = parser.parse_args()

    all_results = discover_all(args.output_dir)

    print(format_terminal(all_results))

    if args.update_experiments or args.update_readme:
        target = args.file or ("README.md" if args.update_readme else "experiments.md")

        # Cập nhật results table
        update_file(
            format_markdown(all_results),
            target,
            RESULTS_MARKER_START,
            RESULTS_MARKER_END,
        )

        # Cập nhật auto-exp descriptions (chỉ khi --update-experiments)
        if args.update_experiments:
            update_file(
                format_auto_experiments(all_results),
                target,
                AUTO_EXP_MARKER_START,
                AUTO_EXP_MARKER_END,
            )
    else:
        print("(Dùng --update-experiments để cập nhật experiments.md tự động)")


if __name__ == "__main__":
    main()
