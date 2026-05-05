"""
compute_metrics.py
==================
Tính các metric đánh giá cho dense retrieval / sequential recommendation.

Input:
  --run   : file TREC run (đã deduplicated)   format: qid Q0 docid rank score run
  --qrels : file qrels (đã deduplicated)      format: qid 0 docid rel
  --ks    : danh sách K, ngăn cách bằng dấu phẩy (mặc định: 5,10,20)
  --out   : file output (nếu không truyền thì chỉ in ra stdout)

Output (mỗi dòng):
  <metric_name>  all  <value>

Metrics được tính:
  ndcg_K  — Normalized Discounted Cumulative Gain tại K
  hr_K    — Hit Rate tại K (= 1 nếu relevant item có trong top-K)
  mrr_K   — Mean Reciprocal Rank tại K (= 0 nếu relevant item ngoài top-K)

Giả thiết: mỗi query có đúng 1 relevant item (leave-one-out protocol).
"""

import argparse
import math
from collections import defaultdict


def load_qrels(path: str) -> dict[str, set]:
    """Trả về {qid: set(relevant_docids)}."""
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
    """Trả về {qid: [(rank, docid), ...]} đã sắp xếp theo rank tăng dần."""
    runs = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, rank = parts[0], parts[1], parts[2], int(parts[3])
            runs[qid].append((rank, docid))
    for qid in runs:
        runs[qid].sort(key=lambda x: x[0])
    return dict(runs)


def compute_all(run: dict, qrels: dict, ks: list[int]) -> dict[str, float]:
    """Tính NDCG@K, HR@K, MRR@K cho tất cả K."""
    results = {f"ndcg_{k}": 0.0 for k in ks}
    results.update({f"hr_{k}": 0.0 for k in ks})
    results.update({f"mrr_{k}": 0.0 for k in ks})

    n_queries = len(qrels)
    if n_queries == 0:
        return results

    for qid, relevant in qrels.items():
        ranked = run.get(qid, [])

        # Tìm rank của relevant item đầu tiên
        hit_rank = None
        for rank, docid in ranked:
            if docid in relevant:
                hit_rank = rank
                break

        for k in ks:
            # HR@K
            if hit_rank is not None and hit_rank <= k:
                results[f"hr_{k}"] += 1.0

            # MRR@K
            if hit_rank is not None and hit_rank <= k:
                results[f"mrr_{k}"] += 1.0 / hit_rank

            # NDCG@K (với 1 relevant item: IDCG = 1/log2(2) = 1)
            if hit_rank is not None and hit_rank <= k:
                results[f"ndcg_{k}"] += 1.0 / math.log2(hit_rank + 1)

    # Trung bình
    for key in results:
        results[key] /= n_queries

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute NDCG@K, HR@K, MRR@K")
    parser.add_argument("--run",   required=True, help="TREC run file")
    parser.add_argument("--qrels", required=True, help="qrels file")
    parser.add_argument("--ks",    default="5,10,20",
                        help="Comma-separated list of K values (default: 5,10,20)")
    parser.add_argument("--out",   default=None, help="Output file (optional)")
    args = parser.parse_args()

    ks = [int(k) for k in args.ks.split(",")]

    qrels = load_qrels(args.qrels)
    run   = load_run(args.run)

    results = compute_all(run, qrels, ks)

    # Format output giống trec_eval: "<metric>\tall\t<value>"
    lines = []
    for k in sorted(ks):
        for metric in ("ndcg", "hr", "mrr"):
            key = f"{metric}_{k}"
            lines.append(f"{key:<20s}\tall\t{results[key]:.4f}")

    output = "\n".join(lines)
    print(output)

    if args.out:
        with open(args.out, "a") as f:
            f.write("\n" + output + "\n")


if __name__ == "__main__":
    main()
