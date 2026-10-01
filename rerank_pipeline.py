#!/usr/bin/env python3
"""
rerank_pipeline.py — các bước Python của pipeline reranker (dùng bởi rerank.sh).

Pipeline (kiểm chứng trong diag/README.md):
  retriever top-K → history filter (mode full) → reranker score → fuse với retriever → metric

Subcommands:
  resolve: in biến shell EVAL_VARIANT / QUERY_VARIANT / QUERY_MAX_LEN suy từ train_config.json của retriever
  pairs : sinh flat (query, candidate) pairs cho reranker từ retriever trec, đã loại item lịch sử.
            --scope inpool : chỉ query có positive trong top-K (nhanh; metric trùng với chạy đủ vì các
                             query còn lại mọi hệ thống đều miss)
            --scope all    : mọi query
            --sample N     : lấy mẫu N query in-pool (dùng cho valid khi chọn α)
  fuse  : s = α·z(retriever) + (1−α)·z(reranker), chuẩn hoá trong từng query sau history filter.
            α chọn trên valid (--valid-trec/--valid-scores) theo NDCG@10, hoặc cố định bằng --alpha.
            In bảng so sánh retriever / reranker / fused, ghi trec + file metric.

History của user lấy từ dataset/dataset/recbole/<ds>/<ds>.inter (cùng chuỗi với Tevatron export).
Positive: valid = item N-1, test = item N (leave-one-out). History = mọi item trước positive.
"""
import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np

from compute_metrics import compute_all

ROOT = os.path.dirname(os.path.abspath(__file__))
KS = [5, 10, 20]
POS_INDEX = {'valid': -2, 'test': -1}


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_histories(dataset, split):
    """{qid: set(history items)} với qid = '<user>_<split>'."""
    seqs = defaultdict(list)
    with open(f'{ROOT}/dataset/dataset/recbole/{dataset}/{dataset}.inter') as f:
        next(f)
        for line in f:
            u, i, t = line.rstrip('\n').split('\t')
            seqs[u].append((float(t), i))
    out = {}
    for u, v in seqs.items():
        s = [i for _, i in sorted(v)]
        pi = len(s) + POS_INDEX[split]
        out[f'{u}_{split}'] = set(s[:pi]) - {s[pi]}
    return out


def load_queries(path):
    with open(path) as f:
        return {r['query_id']: r for r in map(json.loads, f)}


def load_trec(path, depth):
    """{qid: {docid: score}} theo thứ tự rank, dedup, cắt depth."""
    raw = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            raw[p[0]].append((int(p[3]), p[2], float(p[4])))
    out = {}
    for q, lst in raw.items():
        docs = {}
        for _, d, s in sorted(lst):
            if d not in docs and len(docs) < depth:
                docs[d] = s
        out[q] = docs
    return out


def load_scores(path):
    out = defaultdict(dict)
    with open(path) as f:
        for line in f:
            q, d, s = line.rstrip('\n').split('\t')
            out[q][d] = float(s)
    return out


def positives_of(queries):
    return {q: {p['docid'] for p in r.get('positive_passages', [])} for q, r in queries.items()}


# ── pairs ────────────────────────────────────────────────────────────────────

def cmd_pairs(a):
    queries = load_queries(a.queries)
    hist = load_histories(a.dataset, a.split)
    ranked = load_trec(a.trec, a.depth)
    with open(a.corpus) as f:
        corpus = {r['docid']: r['text'] for r in map(json.loads, f)}
    pos = positives_of(queries)

    qids = [q for q in queries if q in ranked]
    if a.scope == 'inpool':
        qids = [q for q in qids if pos[q] & set(ranked[q])]
    if a.sample:
        random.Random(a.seed).shuffle(qids)
        qids = sorted(qids[:a.sample])

    n_pairs = n_hist = 0
    with open(a.output, 'w') as f:
        for q in qids:
            for d in ranked[q]:
                if d in hist.get(q, ()):
                    n_hist += 1
                    continue
                if d not in corpus:
                    continue
                f.write(json.dumps({'query_id': q, 'query': queries[q]['query'], 'docid': d,
                                    'title': '', 'text': corpus[d]}) + '\n')
                n_pairs += 1
    print(f'  {a.split}: {len(qids):,}/{len(queries):,} queries (scope={a.scope}'
          f'{f", sample={a.sample}" if a.sample else ""}), {n_pairs:,} pairs, '
          f'{n_hist:,} history items đã loại → {a.output}')


# ── fuse ─────────────────────────────────────────────────────────────────────

def _norm(v, kind):
    v = np.asarray(v, dtype=np.float64)
    if kind == 'minmax':
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def fuse(ret, rer, hist, alpha, kind, filtered=True):
    """{qid: [docid]} — query không có score reranker giữ thứ tự retriever."""
    out = {}
    for q, rs in ret.items():
        docs = [d for d in rs if not (filtered and d in hist.get(q, ()))]
        r = rer.get(q)
        if r and docs and alpha < 1.0:
            docs = [d for d in docs if d in r]
            s = alpha * _norm([rs[d] for d in docs], kind) + (1 - alpha) * _norm([r[d] for d in docs], kind)
            docs = [docs[i] for i in np.argsort(-s, kind='stable')]
        out[q] = docs
    return out


def evaluate(ranked, qrels):
    run = {q: [(i + 1, d) for i, d in enumerate(docs)] for q, docs in ranked.items()}
    return compute_all(run, qrels, KS)


def cmd_fuse(a):
    alpha, curve = a.alpha, None
    if alpha is None:
        vq = load_queries(a.valid_queries)
        vscores = load_scores(a.valid_scores)
        vhist = load_histories(a.dataset, 'valid')
        vret = {q: d for q, d in load_trec(a.valid_trec, a.depth).items() if q in vscores}
        vqrels = {q: p for q, p in positives_of(vq).items() if q in vscores}
        grid = [round(x, 2) for x in np.arange(0, 1.0001, 0.05)]
        curve = {x: evaluate(fuse(vret, vscores, vhist, x, a.norm), vqrels)['ndcg_10'] for x in grid}
        alpha = max(curve, key=lambda x: (curve[x], -abs(x - 0.5)))
        print(f'  α theo valid ({len(vqrels):,} query in-pool, NDCG@10): ' +
              ' '.join(f'{x}:{curve[x]:.4f}' for x in grid[::2]) + f'  → α = {alpha}')

    queries = load_queries(a.queries)
    qrels = positives_of(queries)
    hist = load_histories(a.dataset, a.split)
    ret = load_trec(a.trec, a.depth)
    scores = load_scores(a.scores)

    systems = [
        ('retriever', 'raw', fuse(ret, {}, hist, 1.0, a.norm, filtered=False)),
        ('retriever', 'filtered', fuse(ret, {}, hist, 1.0, a.norm)),
        ('reranker', 'filtered', fuse(ret, scores, hist, 0.0, a.norm)),
        (f'fused α={alpha}', 'filtered', fuse(ret, scores, hist, alpha, a.norm)),
    ]
    results = [(n, m, evaluate(r, qrels)) for n, m, r in systems]

    cols = [f'{m}_{k}' for k in KS for m in ('ndcg', 'hr', 'mrr')]
    lines = [f'# {a.dataset} / {a.split} — {len(qrels):,} queries, norm={a.norm}, α={alpha}',
             '| System | Candidates | ' + ' | '.join(cols) + ' |',
             '|' + '---|' * (len(cols) + 2)]
    for n, m, r in results:
        lines.append(f'| {n} | {m} | ' + ' | '.join(f'{r[c]:.4f}' for c in cols) + ' |')
    base, best = results[1][2], results[3][2]
    lines.append('')
    lines.append('Δ fused vs retriever (filtered): ' + ', '.join(
        f'{c} {100 * (best[c] / base[c] - 1):+.1f}%' for c in ('ndcg_10', 'hr_10', 'mrr_10') if base[c]))
    table = '\n'.join(lines)
    print(table)

    with open(a.out_eval, 'w') as f:
        f.write(table + '\n')
    with open(a.out_trec, 'w') as f:
        for q, docs in systems[3][2].items():
            for i, d in enumerate(docs, 1):
                f.write(f'{q} Q0 {d} {i} {1.0 / i:.6f} fused\n')
    if a.alpha_json and curve is not None:
        with open(a.alpha_json, 'w') as f:
            json.dump({'alpha': alpha, 'norm': a.norm, 'valid_ndcg_10': curve}, f, indent=2)


# ── resolve ──────────────────────────────────────────────────────────────────

def cmd_resolve(a):
    """In biến shell (eval được) mô tả dữ liệu của retriever và dữ liệu train reranker.

    EVAL_VARIANT : data variant retriever đã train (train_config.json) — query valid/test phải cùng format
                   (vd. cs5-aug → context 5 item). Không có config → '' (bộ chuẩn, context 3).
    QUERY_VARIANT: nguồn query train reranker. auto = EVAL_VARIANT nếu đã là augmented, ngược lại
                   '<variant>-aug'/'aug' nếu thư mục tồn tại; none = EVAL_VARIANT; còn lại = giá trị truyền vào.
    """
    cfg_path = f'{a.retriever_dir}/train_config.json'
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    eval_variant = cfg.get('data_variant', '') or ''
    qlen = int(cfg.get('query_max_len', 128))
    if eval_variant.startswith('v2'):
        raise SystemExit(f'Lỗi: retriever dùng format v2 ({eval_variant}) — pipeline reranker chỉ hỗ trợ v1')

    base = f'{ROOT}/dataset/dataset/tevatron/{a.dataset}'
    exists = lambda v: os.path.isdir(f'{base}-{v}' if v else base)
    if a.query_variant == 'auto':
        if 'aug' in eval_variant.split('-'):
            qv = eval_variant
        else:
            cand = f'{eval_variant}-aug' if eval_variant else 'aug'
            qv = cand if exists(cand) else eval_variant
    elif a.query_variant == 'none':
        qv = eval_variant
    else:
        qv = a.query_variant
    for v in {eval_variant, qv}:
        if not exists(v):
            raise SystemExit(f'Lỗi: không có {base}{"-" + v if v else ""}/ — chạy dataset/export_tevatron.py trước')
    print(f'EVAL_VARIANT="{eval_variant}"')
    print(f'QUERY_VARIANT="{qv}"')
    print(f'QUERY_MAX_LEN={qlen}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('resolve')
    r.add_argument('--dataset', required=True)
    r.add_argument('--retriever-dir', required=True)
    r.add_argument('--query-variant', default='auto')
    r.set_defaults(func=cmd_resolve)

    p = sub.add_parser('pairs')
    p.add_argument('--dataset', required=True)
    p.add_argument('--split', required=True, choices=['valid', 'test'])
    p.add_argument('--queries', required=True)
    p.add_argument('--corpus', required=True)
    p.add_argument('--trec', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--depth', type=int, default=100)
    p.add_argument('--scope', choices=['inpool', 'all'], default='inpool')
    p.add_argument('--sample', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.set_defaults(func=cmd_pairs)

    f = sub.add_parser('fuse')
    f.add_argument('--dataset', required=True)
    f.add_argument('--split', required=True, choices=['valid', 'test'])
    f.add_argument('--queries', required=True)
    f.add_argument('--trec', required=True)
    f.add_argument('--scores', required=True)
    f.add_argument('--valid-queries')
    f.add_argument('--valid-trec')
    f.add_argument('--valid-scores')
    f.add_argument('--alpha', type=float, default=None, help='bỏ qua chọn α trên valid')
    f.add_argument('--norm', choices=['zscore', 'minmax'], default='zscore')
    f.add_argument('--depth', type=int, default=100)
    f.add_argument('--out-trec', required=True)
    f.add_argument('--out-eval', required=True)
    f.add_argument('--alpha-json', default=None)
    f.set_defaults(func=cmd_fuse)

    a = ap.parse_args()
    if a.cmd == 'fuse' and a.alpha is None and not (a.valid_queries and a.valid_trec and a.valid_scores):
        ap.error('fuse: cần --alpha hoặc đủ --valid-queries/--valid-trec/--valid-scores')
    a.func(a)


if __name__ == '__main__':
    main()
