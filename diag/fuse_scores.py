"""
Fuse score retriever + reranker: s = α·norm(retriever) + (1−α)·norm(reranker), chuẩn hoá theo từng query.
α (và cách chuẩn hoá) chọn trên mẫu VALID theo NDCG@10 sau history filter, rồi áp dụng một lần lên TEST.

Luôn filter lịch sử TRƯỚC khi chuẩn hoá (khớp pipeline inference: filter → rerank → fuse).
Query không có score reranker (positive ngoài top-K) giữ thứ tự retriever — với chúng mọi hệ thống đều miss.

Dùng: python diag/fuse_scores.py beauty aug-5 R1 [R3 ...]
      (đọc diag/rerank/<V>_valid_scores.txt và <V>_test_scores.txt)
"""
import argparse
import os
from collections import defaultdict

import numpy as np

from analyze_negatives import load_sequences
from eval_rerank import metrics

ALPHAS = [round(a, 2) for a in np.arange(0, 1.0001, 0.05)]


def load_trec_scores(path, depth):
    out = defaultdict(list)
    with open(path) as f:
        for line in f:
            qid, _, d, rank, score, _ = line.split()
            out[qid].append((int(rank), d, float(score)))
    res = {}
    for q, lst in out.items():
        seen, docs = set(), {}
        for _, d, s in sorted(lst):
            if d not in seen and len(docs) < depth:
                seen.add(d)
                docs[d] = s
        res[q] = docs
    return res


def load_rer(path):
    out = defaultdict(dict)
    with open(path) as f:
        for line in f:
            q, d, s = line.rstrip('\n').split('\t')
            out[q][d] = float(s)
    return out


def norm(vals, kind):
    v = np.asarray(vals, dtype=np.float64)
    if kind == 'minmax':
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def fuse(ret, rer, hist, alpha, kind, filtered=True):
    """Trả về {qid: [docid ranked]}. kind ∈ {minmax, zscore, rrf}."""
    ranked = {}
    for q, rs in ret.items():
        docs = [d for d in rs if not (filtered and d in hist.get(q, ()))]
        if q not in rer or not docs:
            ranked[q] = docs
            continue
        r = rer[q]
        docs = [d for d in docs if d in r]
        if kind == 'rrf':
            # alpha = trọng số retriever; k = 60 như RRF gốc
            ret_rank = {d: i for i, d in enumerate(docs)}
            rer_rank = {d: i for i, d in enumerate(sorted(docs, key=lambda x: -r[x]))}
            s = [alpha / (60 + ret_rank[d]) + (1 - alpha) / (60 + rer_rank[d]) for d in docs]
        else:
            a = norm([rs[d] for d in docs], kind)
            b = norm([r[d] for d in docs], kind)
            s = alpha * a + (1 - alpha) * b
        order = np.argsort(-np.asarray(s), kind='stable')
        ranked[q] = [docs[i] for i in order]
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('rtag')
    ap.add_argument('variants', nargs='+')
    ap.add_argument('--model-tag', default='qwen3-embedding-0.6b')
    ap.add_argument('--depth', type=int, default=100)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    diag = f'{root}/output/{args.dataset}/{args.model_tag}-{args.rtag}/diag'
    seqs = load_sequences(f'{root}/dataset/dataset/recbole/{args.dataset}/{args.dataset}.inter')

    gold = {'valid': {}, 'test': {}}
    hist = {'valid': {}, 'test': {}}
    for u, s in seqs.items():
        gold['test'][f'{u}_test'] = s[-1]
        hist['test'][f'{u}_test'] = set(s[:-1]) - {s[-1]}
        gold['valid'][f'{u}_valid'] = s[-2]
        hist['valid'][f'{u}_valid'] = set(s[:-2]) - {s[-2]}

    ret = {sp: load_trec_scores(f'{diag}/{sp}_rank.trec', args.depth) for sp in ('valid', 'test')}
    lines = ['| System | α (chọn trên valid) | NDCG@10 filt (valid mẫu) | NDCG@5 | HR@5 | NDCG@10 | HR@10 | MRR@10 | NDCG@20 | HR@20 |',
             '|---|---|---|---|---|---|---|---|---|---|']

    def row(name, a, vbest, m):
        return (f'| {name} | {a} | {vbest} | {m["ndcg_5"]:.4f} | {m["hr_5"]:.4f} | {m["ndcg_10"]:.4f} | '
                f'{m["hr_10"]:.4f} | {m["mrr_10"]:.4f} | {m["ndcg_20"]:.4f} | {m["hr_20"]:.4f} |')

    # Baseline: retriever filtered (α = 1)
    m = metrics(fuse(ret['test'], {}, hist['test'], 1.0, 'minmax'), gold['test'])
    lines.append(row('retriever (filtered)', '—', '—', m))

    for v in args.variants:
        rer = {sp: load_rer(f'{diag}/rerank/{v}_{sp}_scores.txt') for sp in ('valid', 'test')}
        vq = set(rer['valid'])  # mẫu valid
        vgold = {q: gold['valid'][q] for q in vq}
        vret = {q: ret['valid'][q] for q in vq}
        m = metrics(fuse(ret['test'], rer['test'], hist['test'], 0.0, 'minmax'), gold['test'])
        lines.append(row(f'{v} only (filtered)', '0', '—', m))
        for kind in ('minmax', 'zscore', 'rrf'):
            curve = {a: metrics(fuse(vret, rer['valid'], hist['valid'], a, kind), vgold)['ndcg_10'] for a in ALPHAS}
            best_a = max(curve, key=lambda a: (curve[a], -abs(a - 0.5)))
            print(f'{v} {kind:6s} valid NDCG@10: ' +
                  ' '.join(f'{a}:{curve[a]:.4f}' for a in ALPHAS[::2]) + f'  → α={best_a}')
            m = metrics(fuse(ret['test'], rer['test'], hist['test'], best_a, kind), gold['test'])
            lines.append(row(f'{v} + retriever [{kind}]', best_a, f'{curve[best_a]:.4f} (α=1: {curve[1.0]:.4f})', m))

    table = '\n'.join(lines)
    print('\nTEST (22,363 query, history-filtered):\n' + table)
    with open(f'{diag}/rerank/fusion_results.md', 'w') as f:
        f.write(table + '\n')


if __name__ == '__main__':
    main()
