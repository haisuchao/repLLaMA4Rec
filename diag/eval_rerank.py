"""
Đánh giá reranker trên top-K của retriever — raw và history-filtered (mode full: loại mọi item user
mua trước item test). Luôn in kèm retriever baseline trên cùng tập ứng viên.

Dùng: python diag/eval_rerank.py beauty aug-5 R0=path/to/R0_test_scores.txt R2=... [--out table.md]
File scores: qid<TAB>docid<TAB>score (output của tevatron.reranker.driver.rerank)
"""
import argparse
import math
import os
from collections import defaultdict

from analyze_negatives import load_sequences, load_trec

KS = (5, 10, 20)


def metrics(ranked, positives):
    out = defaultdict(float)
    for qid, pos in positives.items():
        docs = ranked.get(qid, [])
        r = docs.index(pos) + 1 if pos in docs else None
        for k in KS:
            if r is not None and r <= k:
                out[f'ndcg_{k}'] += 1 / math.log2(r + 1)
                out[f'hr_{k}'] += 1
                out[f'mrr_{k}'] += 1 / r
    return {k: v / len(positives) for k, v in out.items()}


def load_scores(path):
    scores = defaultdict(list)
    with open(path) as f:
        for line in f:
            qid, docid, s = line.rstrip('\n').split('\t')
            scores[qid].append((float(s), docid))
    return {q: [d for _, d in sorted(v, key=lambda x: -x[0])] for q, v in scores.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('rtag')
    ap.add_argument('runs', nargs='*', help='NAME=scores.txt')
    ap.add_argument('--model-tag', default='qwen3-embedding-0.6b')
    ap.add_argument('--depth', type=int, default=100)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    diag_dir = f'{root}/output/{args.dataset}/{args.model_tag}-{args.rtag}/diag'
    seqs = load_sequences(f'{root}/dataset/dataset/recbole/{args.dataset}/{args.dataset}.inter')

    positives, histories = {}, {}
    for u, s in seqs.items():
        qid = f'{u}_test'
        positives[qid] = s[-1]
        histories[qid] = set(s[:-1]) - {s[-1]}

    systems = {'retriever': {q: d[:args.depth] for q, d in load_trec(f'{diag_dir}/test_rank.trec').items()}}
    for spec in args.runs:
        name, path = spec.split('=', 1)
        systems[name] = load_scores(path)

    cols = [f'{m}_{k}' for k in KS for m in ('ndcg', 'hr', 'mrr')]
    header = '| System | Candidates | ' + ' | '.join(cols) + ' |'
    lines = [header, '|' + '---|' * (len(cols) + 2)]
    for name, ranked in systems.items():
        filt = {q: [d for d in docs if d not in histories.get(q, ())] for q, docs in ranked.items()}
        for mode, rk in (('raw', ranked), ('filtered', filt)):
            m = metrics(rk, positives)
            lines.append(f'| {name} | {mode} | ' + ' | '.join(f'{m.get(c, 0):.4f}' for c in cols) + ' |')
    table = '\n'.join(lines)
    print(table)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(table + '\n')


if __name__ == '__main__':
    main()
