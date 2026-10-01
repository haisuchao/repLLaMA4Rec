"""
Sinh dữ liệu cho thí nghiệm kiểm chứng negative sampling của reranker (xem diag/README.md).

Từ top-DEPTH của retriever (diag/<split>_rank.trec do mine_ranks.sh tạo), sinh:
  R0_train.jsonl : như prepare_rerank_data.py — mọi query, negatives = top-DEPTH trừ positive
  R1_train.jsonl : chỉ query có positive nằm trong top-DEPTH
  R2_train.jsonl : R1 + loại item lịch sử (mua trước positive) khỏi pool negatives
  test_pairs.jsonl : flat (query, candidate) cho top-DEPTH của test (không lọc — lọc lúc evaluate)
  valid_pairs_sample.jsonl : như trên cho VALID_SAMPLE query valid có positive trong pool (chọn α khi fuse)

Dùng: python diag/build_rerank_data.py beauty aug-5 [--depth 100]
"""
import argparse
import json
import os
import random

from analyze_negatives import load_sequences, load_trec


def load_jsonl(path):
    with open(path) as f:
        return {r['query_id']: r for r in map(json.loads, f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('rtag')
    ap.add_argument('--model-tag', default='qwen3-embedding-0.6b')
    ap.add_argument('--depth', type=int, default=100)
    ap.add_argument('--valid-sample', type=int, default=2000,
                    help='số query valid (có positive trong pool) dùng để chọn α khi fuse')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    diag_dir = f'{root}/output/{args.dataset}/{args.model_tag}-{args.rtag}/diag'
    out_dir = f'{diag_dir}/rerank'
    os.makedirs(out_dir, exist_ok=True)
    data_dir = f'{root}/dataset/dataset/tevatron/{args.dataset}'

    seqs = load_sequences(f'{root}/dataset/dataset/recbole/{args.dataset}/{args.dataset}.inter')
    with open(f'{data_dir}/corpus.jsonl') as f:
        corpus = {r['docid']: r['text'] for r in map(json.loads, f)}

    def psg(d):
        return {'docid': d, 'title': '', 'text': corpus[d]}

    # ── Train ────────────────────────────────────────────────────────────────
    queries = load_jsonl(f'{data_dir}/train.jsonl')
    ranked = load_trec(f'{diag_dir}/train_rank.trec')
    outs = {v: open(f'{out_dir}/{v}_train.jsonl', 'w') for v in ('R0', 'R1', 'R2')}
    counts = {v: 0 for v in outs}
    n_negs = {v: 0 for v in outs}

    for qid, q in queries.items():
        pos_ids = [p['docid'] for p in q['positive_passages'] if p['docid'] in corpus]
        if not pos_ids:
            continue
        u = qid.rsplit('_', 1)[0]
        s = seqs[u]
        pos = s[-3]
        assert pos in pos_ids, (qid, pos, pos_ids)
        history = set(s[:-3])

        cands = [d for d in ranked.get(qid, [])[:args.depth] if d in corpus]
        in_pool = pos in cands
        negs_all = [d for d in cands if d not in pos_ids]
        negs_nohist = [d for d in negs_all if d not in history]

        variants = {'R0': negs_all}
        if in_pool:
            variants['R1'] = negs_all
            variants['R2'] = negs_nohist
        for v, negs in variants.items():
            if not negs:
                continue
            outs[v].write(json.dumps({
                'query_id': qid,
                'query': q['query'],
                'positive_passages': [psg(d) for d in pos_ids],
                'negative_passages': [psg(d) for d in negs],
            }) + '\n')
            counts[v] += 1
            n_negs[v] += len(negs)
    for f in outs.values():
        f.close()
    for v in outs:
        print(f'{v}: {counts[v]:,} queries, {n_negs[v] / max(counts[v], 1):.1f} negatives/query')

    # ── Test pairs ───────────────────────────────────────────────────────────
    queries = load_jsonl(f'{data_dir}/test.jsonl')
    ranked = load_trec(f'{diag_dir}/test_rank.trec')
    n = 0
    with open(f'{out_dir}/test_pairs.jsonl', 'w') as f:
        for qid, q in queries.items():
            for d in ranked.get(qid, [])[:args.depth]:
                if d in corpus:
                    f.write(json.dumps({'query_id': qid, 'query': q['query'], 'docid': d,
                                        'title': '', 'text': corpus[d]}) + '\n')
                    n += 1
    print(f'test_pairs: {n:,} pairs')

    # ── Valid pairs (mẫu query có positive trong pool) — để chọn α khi fuse score ──
    queries = load_jsonl(f'{data_dir}/valid.jsonl')
    ranked = load_trec(f'{diag_dir}/valid_rank.trec')
    inpool = [qid for qid in queries
              if seqs[qid.rsplit('_', 1)[0]][-2] in ranked.get(qid, [])[:args.depth]]
    random.Random(args.seed).shuffle(inpool)
    sample = sorted(inpool[:args.valid_sample])
    n = 0
    with open(f'{out_dir}/valid_pairs_sample.jsonl', 'w') as f:
        for qid in sample:
            q = queries[qid]
            for d in ranked[qid][:args.depth]:
                if d in corpus:
                    f.write(json.dumps({'query_id': qid, 'query': q['query'], 'docid': d,
                                        'title': '', 'text': corpus[d]}) + '\n')
                    n += 1
    print(f'valid_pairs_sample: {len(sample):,}/{len(inpool):,} in-pool queries, {n:,} pairs')


if __name__ == '__main__':
    main()
