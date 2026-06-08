"""
Tính Recall@K của retriever tại nhiều depth khác nhau.

Dùng pkl embeddings sẵn có (không encode lại corpus/queries).

Ví dụ:
  python compute_recall_depth.py beauty --tag aug-5 --checkpoint checkpoint-12000
  python compute_recall_depth.py beauty --tag aug-5 --checkpoint checkpoint-12000 \
      --depths 50,100,150,200,300,500
"""
import argparse
import pickle
import os
import numpy as np
import faiss


def load_pkl(path):
    with open(path, "rb") as f:
        reps, ids = pickle.load(f)
    if not isinstance(reps, np.ndarray):
        reps = np.array(reps)
    return reps.astype(np.float32), list(ids)


def load_qrels(path):
    """Trả về dict: qid → set(docid)"""
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts
            if int(rel) > 0:
                qrels.setdefault(qid, set()).add(docid)
    return qrels


def compute_recall(run, qrels, depth):
    """
    run: dict qid → list of docid (đã sorted by score, length = depth)
    Trả về Recall@depth = tỉ lệ queries có relevant item trong top-depth.
    """
    hits = 0
    total = 0
    for qid, relevant in qrels.items():
        if qid not in run:
            continue
        retrieved = set(run[qid][:depth])
        if retrieved & relevant:
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="beauty | sports | ml-1m")
    parser.add_argument("--tag", default="", help="Model tag, ví dụ: aug-5")
    parser.add_argument("--checkpoint", default="", help="checkpoint-12000 hoặc latest")
    parser.add_argument("--split", default="test", help="valid | test")
    parser.add_argument(
        "--depths",
        default="50,100,150,200,300,500",
        help="Comma-separated list of depths để thử",
    )
    parser.add_argument("--output-dir", default="", help="Override output dir")
    args = parser.parse_args()

    # ── Paths ────────────────────────────────────────────────────────────────────
    base_dir = os.path.join("output", args.dataset)
    model_tag = "qwen3-embedding-0.6b"
    if args.tag:
        model_tag = f"{model_tag}-{args.tag}"

    model_dir = os.path.join(base_dir, model_tag)
    emb_dir = os.path.join(model_dir, "embeddings")
    results_dir = os.path.join(emb_dir, "results")

    # Tìm corpus pkl
    ckpt = args.checkpoint
    corpus_pkl = os.path.join(emb_dir, "corpus", f"{ckpt}.pkl")
    if not os.path.exists(corpus_pkl):
        # fallback: final model pkl
        corpus_pkl = os.path.join(emb_dir, "corpus", f"{model_tag}.pkl")
    if not os.path.exists(corpus_pkl):
        raise FileNotFoundError(f"Không tìm thấy corpus pkl. Đã thử:\n  {corpus_pkl}")

    # Tìm query pkl
    query_pkl = os.path.join(emb_dir, "queries", f"{args.split}_{ckpt}.pkl")
    if not os.path.exists(query_pkl):
        query_pkl = os.path.join(emb_dir, "queries", f"{args.split}_{model_tag}.pkl")
    if not os.path.exists(query_pkl):
        raise FileNotFoundError(f"Không tìm thấy query pkl. Đã thử:\n  {query_pkl}")

    # Qrels
    qrels_path = os.path.join(results_dir, f"{args.split}_qrels_clean.txt")
    if not os.path.exists(qrels_path):
        raise FileNotFoundError(f"Không tìm thấy qrels: {qrels_path}\nChạy eval.sh trước.")

    depths = sorted(int(d) for d in args.depths.split(","))
    max_depth = max(depths)

    print(f"Dataset   : {args.dataset}")
    print(f"Model tag : {model_tag}")
    print(f"Checkpoint: {ckpt}")
    print(f"Split     : {args.split}")
    print(f"Corpus    : {corpus_pkl}")
    print(f"Queries   : {query_pkl}")
    print()

    # ── Load embeddings ────────────────────────────────────────────────────────
    print("Loading corpus embeddings...", end=" ", flush=True)
    p_reps, p_ids = load_pkl(corpus_pkl)
    print(f"{p_reps.shape[0]:,} items × {p_reps.shape[1]}d")

    print("Loading query embeddings... ", end=" ", flush=True)
    q_reps, q_ids = load_pkl(query_pkl)
    print(f"{q_reps.shape[0]:,} queries × {q_reps.shape[1]}d")

    # ── FAISS brute-force search ───────────────────────────────────────────────
    print(f"\nBuilding FAISS flat index ({p_reps.shape[0]:,} items)...", end=" ", flush=True)
    index = faiss.IndexFlatIP(p_reps.shape[1])  # inner product (embeddings đã normalize)
    index.add(p_reps)
    print("done")

    print(f"Searching at depth={max_depth}...", end=" ", flush=True)
    scores, indices = index.search(q_reps, max_depth)
    print("done\n")

    # Build run dict: qid → list of docid (sorted by score, length = max_depth)
    run = {}
    for i, qid in enumerate(q_ids):
        run[qid] = [p_ids[idx] for idx in indices[i] if idx >= 0]

    # ── Load qrels ─────────────────────────────────────────────────────────────
    qrels = load_qrels(qrels_path)
    n_queries = len(qrels)

    # ── Compute Recall@K for each depth ───────────────────────────────────────
    print(f"{'Depth':>8}  {'Recall@Depth':>14}  {'Δ vs depth=100':>16}  {'Queries Hit':>12}")
    print("─" * 58)

    recall_at = {}
    for d in depths:
        r = compute_recall(run, qrels, d)
        recall_at[d] = r

    recall_100 = recall_at.get(100)

    for d in depths:
        r = recall_at[d]
        hits = sum(
            1 for qid, rel in qrels.items()
            if qid in run and (set(run[qid][:d]) & rel)
        )
        if d == 100:
            delta = "—  (baseline)"
        elif recall_100 is not None:
            diff = r - recall_100
            delta = f"{diff:+.4f} ({diff/recall_100*100:+.1f}%)"
        else:
            delta = "n/a"
        print(f"{d:>8}  {r:>14.4f}  {delta:>22}  {hits:>6}/{n_queries}")

    print()
    print(f"Tổng số queries: {n_queries:,}")
    print(f"Corpus size    : {p_reps.shape[0]:,} items")
    print()

    if recall_100 is not None and depths[-1] > 100:
        recall_max = compute_recall(run, qrels, depths[-1])
        absolute_gain = recall_max - recall_100
        print(
            f"Recall@{depths[-1]} = {recall_max:.4f}  vs  Recall@100 = {recall_100:.4f}"
            f"  →  tăng {absolute_gain:+.4f} ({absolute_gain/recall_100*100:+.1f}%)"
        )
        if recall_200 := compute_recall(run, qrels, 200) if 200 in depths else None:
            gap = recall_200 - recall_100
            print(
                f"\nRecall@200 = {recall_200:.4f}  →  tăng {gap:+.4f} so với depth=100"
            )
            print(
                f"Điều này có nghĩa: với {int(gap * n_queries)} query thêm,"
                f" item đúng nằm trong top-200 nhưng không nằm trong top-100."
            )


if __name__ == "__main__":
    main()
