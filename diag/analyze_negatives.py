"""
Chẩn đoán chất lượng hard negatives cho reranker (đầu vào: output của diag/mine_ranks.sh).

Với mỗi split (train/valid/test), đo:
  1. Rank của positive trong top-K của retriever — raw và sau history filter (full)
  2. Thành phần negatives theo dải rank:
       history   : item user đã mua trước positive (đúng loại mà history filter loại bỏ)
       future    : item user mua SAU positive (chắc chắn là false negative về mặt hành vi)
       dup       : title gần trùng positive (Jaccard từ >= 0.6) — nhiều khi là cùng sản phẩm khác ASIN
       co-buy    : >= 2 user KHÁC cùng mua positive và negative (item-item co-occurrence)
       next-of-last : >= 1 user KHÁC mua negative ngay sau item cuối của query (transition)
  3. Baseline so sánh: cùng các thống kê trên negatives lấy ngẫu nhiên đều từ corpus

Mọi thống kê co-occurrence chỉ dùng phần train của chuỗi (bỏ 2 item cuối — valid/test),
và loại trừ chính user đang xét → không leak.

Dùng: python diag/analyze_negatives.py beauty aug-5
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict

import numpy as np

BANDS = [(1, 10), (11, 20), (21, 50), (51, 100), (101, 200)]
DUP_JACCARD = 0.6
COBUY_MIN_USERS = 2


def load_sequences(inter_path):
    seqs = defaultdict(list)
    with open(inter_path) as f:
        next(f)
        for line in f:
            u, i, t = line.rstrip('\n').split('\t')
            seqs[u].append((float(t), i))
    return {u: [i for _, i in sorted(v)] for u, v in seqs.items()}


def load_trec(path):
    ranked = defaultdict(list)
    with open(path) as f:
        for line in f:
            qid, _, docid, rank, _score, _ = line.split()
            ranked[qid].append((int(rank), docid))
    out = {}
    for qid, lst in ranked.items():
        seen, docs = set(), []
        for _, d in sorted(lst):
            if d not in seen:
                seen.add(d)
                docs.append(d)
        out[qid] = docs
    return out


def load_titles(corpus_path):
    titles = {}
    with open(corpus_path) as f:
        for line in f:
            r = json.loads(line)
            t = r['text']
            if t.startswith('Passage: '):
                t = t[len('Passage: '):]
            titles[r['docid']] = t
    return titles


def tokens(t):
    return set(re.findall(r'[a-z0-9]+', t.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('rtag')
    ap.add_argument('--model-tag', default='qwen3-embedding-0.6b')
    ap.add_argument('--depth', type=int, default=100,
                    help='depth dùng làm pool negatives của reranker (mặc định 100 như train_reranker.sh)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    diag_dir = f'{root}/output/{args.dataset}/{args.model_tag}-{args.rtag}/diag'
    seqs = load_sequences(f'{root}/dataset/dataset/recbole/{args.dataset}/{args.dataset}.inter')
    titles = load_titles(f'{root}/dataset/dataset/tevatron/{args.dataset}/corpus.jsonl')
    title_tok = {d: tokens(t) for d, t in titles.items()}
    all_items = list(titles)

    # ── Co-occurrence chỉ từ phần train (bỏ 2 item cuối) ─────────────────────
    item_users = defaultdict(set)
    trans = defaultdict(lambda: defaultdict(set))   # a -> b -> {users}
    for u, s in seqs.items():
        tr = s[:-2]
        for i in tr:
            item_users[i].add(u)
        for a, b in zip(tr, tr[1:]):
            trans[a][b].add(u)

    def cobuy(u, a, b):
        ua, ub = item_users.get(a, ()), item_users.get(b, ())
        if len(ua) > len(ub):
            ua, ub = ub, ua
        return sum(1 for x in ua if x != u and x in ub)

    def next_of(u, a, b):
        us = trans.get(a, {}).get(b, ())
        return sum(1 for x in us if x != u)

    def dup(a, b):
        ta, tb = title_tok.get(a, set()), title_tok.get(b, set())
        if not ta or not tb:
            return False
        return len(ta & tb) / len(ta | tb) >= DUP_JACCARD

    # positive index trong chuỗi: train = N-2 (index -3), valid = -2, test = -1
    pos_index = {'train': -3, 'valid': -2, 'test': -1}
    report = {}

    for split in ('train', 'valid', 'test'):
        trec = f'{diag_dir}/{split}_rank.trec'
        if not os.path.exists(trec):
            print(f'[skip] {trec} không tồn tại')
            continue
        ranked = load_trec(trec)
        K = max(len(v) for v in ranked.values())

        ranks_raw, ranks_filt = [], []
        # band -> category -> count ; band -> total negatives
        comp = {b: defaultdict(int) for b in BANDS}
        comp_tot = {b: 0 for b in BANDS}
        rand_comp, rand_tot = defaultdict(int), 0
        n_hist_in_pool, n_pool = [], []
        n_q = 0

        for qid, docs in ranked.items():
            u = qid.rsplit('_', 1)[0]
            s = seqs.get(u)
            if s is None:
                continue
            pi = len(s) + pos_index[split]
            pos = s[pi]
            history = set(s[:pi])
            future = set(s[pi + 1:]) - {pos}
            last = s[pi - 1]
            n_q += 1

            r = docs.index(pos) + 1 if pos in docs else None
            ranks_raw.append(r)
            filt = [d for d in docs if d not in history or d == pos]
            ranks_filt.append(filt.index(pos) + 1 if pos in filt else None)

            pool = [d for d in docs[:args.depth] if d != pos]
            n_pool.append(len(pool))
            n_hist_in_pool.append(sum(d in history for d in pool))

            def classify(d):
                cats = []
                if d in history:
                    cats.append('history')
                if d in future:
                    cats.append('future')
                if dup(pos, d):
                    cats.append('dup')
                if cobuy(u, pos, d) >= COBUY_MIN_USERS:
                    cats.append('co-buy')
                if next_of(u, last, d) >= 1:
                    cats.append('next-of-last')
                if d not in history and d not in future and not cats:
                    cats.append('clean')
                return cats

            for rank, d in enumerate(docs, 1):
                if d == pos:
                    continue
                for b in BANDS:
                    if b[0] <= rank <= b[1]:
                        comp_tot[b] += 1
                        for c in classify(d):
                            comp[b][c] += 1
                        break

            # Baseline: 10 negatives ngẫu nhiên đều từ corpus
            for _ in range(10):
                d = rng.choice(all_items)
                if d == pos:
                    continue
                rand_tot += 1
                for c in classify(d):
                    rand_comp[c] += 1

        def rank_stats(rs):
            arr = np.array([x if x is not None else 10 ** 9 for x in rs])
            found = arr < 10 ** 9
            out = {f'R@{k}': float(np.mean(arr <= k)) for k in (1, 10, 20, 50, 100, 200) if k <= K}
            out['MRR@K'] = float(np.mean(np.where(found, 1.0 / arr, 0.0)))
            out['median_rank_if_found'] = float(np.median(arr[found])) if found.any() else None
            return out

        cats = ['history', 'future', 'dup', 'co-buy', 'next-of-last', 'clean']
        report[split] = {
            'n_queries': n_q,
            'positive_rank_raw': rank_stats(ranks_raw),
            'positive_rank_history_filtered': rank_stats(ranks_filt),
            f'history_items_in_top{args.depth}_pool_mean': float(np.mean(n_hist_in_pool)),
            'composition_by_band': {
                f'{b[0]}-{b[1]}': {c: comp[b][c] / comp_tot[b] for c in cats} if comp_tot[b] else {}
                for b in BANDS
            },
            'composition_random_negatives': {c: rand_comp[c] / rand_tot for c in cats},
        }

    out_path = f'{diag_dir}/negative_diagnosis.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    # ── In bảng gọn ──────────────────────────────────────────────────────────
    for split, rep in report.items():
        print(f'\n═══ {split.upper()} ({rep["n_queries"]:,} queries) ═══')
        for kind in ('positive_rank_raw', 'positive_rank_history_filtered'):
            st = rep[kind]
            print(f'  {kind:32s} ' + '  '.join(
                f'{k}={v:.4f}' if isinstance(v, float) and k != 'median_rank_if_found' else f'{k}={v}'
                for k, v in st.items()))
        print(f'  history items trong pool top-{args.depth}: '
              f'{rep[f"history_items_in_top{args.depth}_pool_mean"]:.2f}/query')
        print(f'  {"band":>9s} ' + ' '.join(f'{c:>13s}' for c in cats))
        for band, d in rep['composition_by_band'].items():
            if d:
                print(f'  {band:>9s} ' + ' '.join(f'{100 * d[c]:12.2f}%' for c in cats))
        d = rep['composition_random_negatives']
        print(f'  {"random":>9s} ' + ' '.join(f'{100 * d[c]:12.2f}%' for c in cats))
    print(f'\n→ {out_path}')


if __name__ == '__main__':
    main()
