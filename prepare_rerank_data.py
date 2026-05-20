#!/usr/bin/env python3
"""
prepare_rerank_data.py
Chuẩn bị dữ liệu cho reranker từ output của retriever.

Mode train:
  Input:  <split>.jsonl + <split>_rank.trec + corpus.jsonl
  Output: reranker_train.jsonl
  Format: {query_id, query, positive_passages, negative_passages}
          (negative_passages = hard negatives từ top retrieved, loại bỏ positive)

Mode infer:
  Input:  <split>.jsonl + <split>_rank.trec + corpus.jsonl
  Output: flat pairs jsonl — 1 dòng cho mỗi (query, candidate) pair
  Format: {query_id, query, docid, title, text}
"""
import argparse
import json
from collections import defaultdict

from tqdm import tqdm


def load_corpus(path):
    corpus = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            corpus[d['docid']] = d.get('text', '')
    return corpus


def load_queries(path):
    """Returns {query_id: {'query': str, 'positive_docids': set}}"""
    queries = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            queries[d['query_id']] = {
                'query': d['query'],
                'positive_docids': {p['docid'] for p in d.get('positive_passages', [])},
            }
    return queries


def load_trec(path):
    """Returns {query_id: [docid_rank1, docid_rank2, ...]} sorted by rank."""
    retrieved = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            qid, _, docid, rank, _score, _run = parts
            retrieved[qid].append((int(rank), docid))
    return {qid: [d for _, d in sorted(docs)] for qid, docs in retrieved.items()}


def main():
    p = argparse.ArgumentParser(description='Chuẩn bị dữ liệu cho reranker')
    p.add_argument('--mode',    choices=['train', 'infer'], required=True,
                   help='train: training data với hard negatives; infer: flat pairs để rerank')
    p.add_argument('--queries', required=True, help='Path đến split.jsonl')
    p.add_argument('--corpus',  required=True, help='Path đến corpus.jsonl')
    p.add_argument('--trec',    required=True, help='Path đến rank.trec từ retriever')
    p.add_argument('--output',  required=True, help='Output jsonl path')
    p.add_argument('--depth',   type=int, default=100,
                   help='Số candidates tối đa mỗi query (mặc định: 100)')
    args = p.parse_args()

    print('Loading corpus ...', flush=True)
    corpus = load_corpus(args.corpus)
    print(f'  {len(corpus):,} items')

    print('Loading queries ...', flush=True)
    queries = load_queries(args.queries)
    print(f'  {len(queries):,} queries')

    print('Loading retrieval results ...', flush=True)
    retrieved = load_trec(args.trec)
    print(f'  {len(retrieved):,} queries have results')

    n_written = n_skip = 0

    with open(args.output, 'w') as out:
        for qid, q in tqdm(queries.items(), desc=f'[{args.mode}]'):
            query   = q['query']
            pos_ids = q['positive_docids']
            cands   = retrieved.get(qid, [])[:args.depth]

            if args.mode == 'train':
                valid_pos = [
                    {'docid': d, 'title': '', 'text': corpus[d]}
                    for d in pos_ids if d in corpus
                ]
                hard_negs = [
                    {'docid': d, 'title': '', 'text': corpus[d]}
                    for d in cands if d not in pos_ids and d in corpus
                ]
                if not valid_pos or not hard_negs:
                    n_skip += 1
                    continue
                out.write(json.dumps({
                    'query_id':          qid,
                    'query':             query,
                    'positive_passages': valid_pos,
                    'negative_passages': hard_negs,
                }) + '\n')
                n_written += 1

            else:  # infer
                for docid in cands:
                    if docid not in corpus:
                        continue
                    out.write(json.dumps({
                        'query_id': qid,
                        'query':    query,
                        'docid':    docid,
                        'title':    '',
                        'text':     corpus[docid],
                    }) + '\n')
                    n_written += 1

    print(f'\nMode {args.mode}: {n_written:,} records → {args.output}')
    if n_skip:
        print(f'Skipped {n_skip} queries (no positive in corpus or no hard negatives)')


if __name__ == '__main__':
    main()
