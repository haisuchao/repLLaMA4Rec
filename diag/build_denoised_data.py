"""
Dữ liệu reranker cho các chiến lược negative sampling sau R1 (xem diag/README.md). Tất cả chỉ giữ query có
positive trong top-DEPTH (kết luận của thí nghiệm R0/R0s/R1/R2).

  R3_train.jsonl    : query của R1; negatives = top-DEPTH trừ item bị gắn cờ false-negative theo hành vi:
                      future (user mua sau), dup (title Jaccard ≥ 0.6 với positive),
                      co-buy (≥ 2 user khác mua cả hai), next-of-last (≥ 1 user khác mua ngay sau item cuối query).
                      Item lịch sử giữ lại (R2 cho thấy bỏ chúng không giúp).
  R4_train.jsonl    : query của R1; SimANS — lấy trước N_EPOCH×(G−1) negatives không lặp với
                      p ∝ exp(−a·(s_neg − s_pos − b)²) theo score cosine của retriever.
                      Tevatron xáo cố định rồi lấy (G−1) negatives khác nhau mỗi epoch → mỗi negative dùng đúng 1 lần.
  R1aug_train.jsonl : như R1 nhưng trên query augmented (tevatron/<ds>-aug/train.jsonl, aug_train_rank.trec).

Dùng: python diag/build_denoised_data.py beauty aug-5 [--simans-sigma 0.01]
"""
import argparse
import json
import os

import numpy as np

from analyze_negatives import (COBUY_MIN_USERS, DUP_JACCARD, load_sequences, tokens)
from build_rerank_data import load_jsonl
from fuse_scores import load_trec_scores
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('rtag')
    ap.add_argument('--model-tag', default='qwen3-embedding-0.6b')
    ap.add_argument('--depth', type=int, default=100)
    ap.add_argument('--group-size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--simans-sigma', type=float, default=0.01,
                    help='độ rộng kernel SimANS: a = 1/(2σ²); s_neg − s_pos có std ≈ 0.019 trên Beauty')
    ap.add_argument('--simans-b', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    diag = f'{root}/output/{args.dataset}/{args.model_tag}-{args.rtag}/diag'
    out_dir = f'{diag}/rerank'
    data_dir = f'{root}/dataset/dataset/tevatron/{args.dataset}'
    seqs = load_sequences(f'{root}/dataset/dataset/recbole/{args.dataset}/{args.dataset}.inter')
    with open(f'{data_dir}/corpus.jsonl') as f:
        corpus = {r['docid']: r['text'] for r in map(json.loads, f)}
    title_tok = {d: tokens(t[len('Passage: '):] if t.startswith('Passage: ') else t) for d, t in corpus.items()}

    # Co-occurrence chỉ từ phần train (bỏ 2 item cuối), loại chính user — giống analyze_negatives.py
    item_users = defaultdict(set)
    trans = defaultdict(lambda: defaultdict(set))
    for u, s in seqs.items():
        tr = s[:-2]
        for i in tr:
            item_users[i].add(u)
        for a, b in zip(tr, tr[1:]):
            trans[a][b].add(u)

    def flagged(u, pos, last, future, d):
        if d in future:
            return 'future'
        ta, tb = title_tok.get(pos, set()), title_tok.get(d, set())
        if ta and tb and len(ta & tb) / len(ta | tb) >= DUP_JACCARD:
            return 'dup'
        ua, ub = item_users.get(pos, ()), item_users.get(d, ())
        if len(ua) > len(ub):
            ua, ub = ub, ua
        if sum(1 for x in ua if x != u and x in ub) >= COBUY_MIN_USERS:
            return 'co-buy'
        if any(x != u for x in trans.get(last, {}).get(d, ())):
            return 'next-of-last'
        return None

    def psg(d):
        return {'docid': d, 'title': '', 'text': corpus[d]}

    def write(f, qid, query, pos, negs):
        f.write(json.dumps({'query_id': qid, 'query': query, 'positive_passages': [psg(pos)],
                            'negative_passages': [psg(d) for d in negs]}) + '\n')

    # ── R3, R4 trên query của R1 ─────────────────────────────────────────────
    queries = load_jsonl(f'{data_dir}/train.jsonl')
    ret = load_trec_scores(f'{diag}/train_rank.trec', args.depth)
    a = 1.0 / (2 * args.simans_sigma ** 2)
    n_take = args.epochs * (args.group_size - 1)
    stats = defaultdict(int)
    with open(f'{out_dir}/R3_train.jsonl', 'w') as f3, open(f'{out_dir}/R4_train.jsonl', 'w') as f4:
        for qid, q in queries.items():
            u = qid.rsplit('_', 1)[0]
            s = seqs[u]
            pos = s[-3]
            docs = {d: sc for d, sc in ret.get(qid, {}).items() if d in corpus}
            if pos not in docs:
                continue
            last, future = s[-4], set(s[-2:]) - {pos}
            negs = [d for d in docs if d != pos]
            stats['queries'] += 1
            stats['neg_total'] += len(negs)

            clean = []
            for d in negs:
                flag = flagged(u, pos, last, future, d)
                stats[f'removed_{flag}' if flag else 'kept'] += 1
                if not flag:
                    clean.append(d)
            if clean:
                write(f3, qid, q['query'], pos, clean)

            diff = np.array([docs[d] - docs[pos] - args.simans_b for d in negs])
            p = np.exp(-a * diff ** 2)
            p /= p.sum()
            k = min(n_take, len(negs))
            pick = rng.choice(len(negs), size=k, replace=False, p=p)
            write(f4, qid, q['query'], pos, [negs[i] for i in pick])
            stats['simans_neg_rank_sum'] += sum(i + 1 for i in pick)
            stats['simans_neg_n'] += k

    kept = stats['kept'] / stats['neg_total']
    print(f"R3/R4: {stats['queries']:,} queries | R3 giữ {100 * kept:.1f}% negatives, bỏ: " +
          ', '.join(f"{k[8:]}={100 * v / stats['neg_total']:.1f}%" for k, v in stats.items() if k.startswith('removed_')))
    print(f"R4: SimANS σ={args.simans_sigma} (a={a:.0f}), {n_take} negatives/query, "
          f"rank trung bình của negative được chọn = {stats['simans_neg_rank_sum'] / stats['simans_neg_n']:.1f}")

    # ── R1aug ────────────────────────────────────────────────────────────────
    aug_path = f'{diag}/aug_train_rank.trec'
    if not os.path.exists(aug_path):
        print(f'[skip R1aug] chưa có {aug_path}')
        return
    queries = load_jsonl(f'{data_dir}-aug/train.jsonl')
    ret = load_trec_scores(aug_path, args.depth)
    n = n_all = 0
    with open(f'{out_dir}/R1aug_train.jsonl', 'w') as f:
        for qid, q in queries.items():
            n_all += 1
            pos = q['positive_passages'][0]['docid']
            docs = [d for d in ret.get(qid, {}) if d in corpus]
            if pos not in docs:
                continue
            write(f, qid, q['query'], pos, [d for d in docs if d != pos])
            n += 1
    print(f'R1aug: {n:,}/{n_all:,} augmented queries có positive trong top-{args.depth}')


if __name__ == '__main__':
    main()
