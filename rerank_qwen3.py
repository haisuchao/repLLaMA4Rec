#!/usr/bin/env python3
"""
rerank_qwen3.py — Reranking với Qwen3-Reranker (zero-shot hoặc fine-tuned).

Input:  pairs JSONL (từ prepare_rerank_data.py --mode infer)
        mỗi dòng: {"query_id":..., "query":"Query: item1, item2 </s>", "docid":..., "text":"item title"}
Output: qid\tdocid\tscore (một dòng / pair)

Score = P("yes") từ logit tại vị trí cuối: softmax([logit_no, logit_yes])[1]

Cách dùng (thường gọi qua rerank_qwen3.sh):
  # Zero-shot
  python rerank_qwen3.py --pairs test_pairs.jsonl --output test_reranked.txt

  # Fine-tuned (LoRA)
  python rerank_qwen3.py --pairs test_pairs.jsonl --output test_reranked.txt \
      --lora-path output/beauty/qwen3-embedding-0.6b-qwen3-reranker-0.6b
"""
import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


TASK = (
    "A user recently purchased the items listed in the Query (in chronological order). "
    "Determine whether the Document is the item the user purchased immediately after. "
    "The correct next item should be a new product not already in the purchase history "
    "that fits the user's demonstrated preferences and shopping pattern."
)

PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    '<|im_start|>user\n'
)
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def strip_query(raw: str) -> str:
    q = raw.strip()
    if q.startswith('Query: '):
        q = q[len('Query: '):]
    if q.endswith('</s>'):
        q = q[:-4].strip().rstrip(',').strip()
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs',      required=True,
                        help='Input pairs JSONL (prepare_rerank_data.py --mode infer)')
    parser.add_argument('--output',     required=True,
                        help='Output: qid\\tdocid\\tscore mỗi dòng')
    parser.add_argument('--model',      default='Qwen/Qwen3-Reranker-0.6B')
    parser.add_argument('--lora-path',  default=None,
                        help='Path đến LoRA weights (từ train_reranker_qwen3.sh). '
                             'Để trống = zero-shot.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Số pairs xử lý mỗi batch (mặc định: 8)')
    parser.add_argument('--max-length', type=int, default=256,
                        help='Max input tokens mỗi pair (mặc định: 256)')
    parser.add_argument('--task',       default=TASK)
    args = parser.parse_args()

    mode = 'fine-tuned' if args.lora_path else 'zero-shot'
    print(f'Model     : {args.model}  [{mode}]')
    if args.lora_path:
        print(f'LoRA path : {args.lora_path}')

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    )

    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()   # merge weights → inference speed giống base model

    model = model.eval().cuda()

    token_false_id = tokenizer.convert_tokens_to_ids('no')
    token_true_id  = tokenizer.convert_tokens_to_ids('yes')
    print(f'Token IDs : yes={token_true_id}, no={token_false_id}')

    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
    max_body   = args.max_length - len(prefix_ids) - len(suffix_ids)

    print('Loading pairs ...')
    pairs = []
    with open(args.pairs) as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f'  {len(pairs):,} pairs')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, 'w') as out_f, torch.no_grad():
        for start in tqdm(range(0, len(pairs), args.batch_size), desc='Reranking'):
            batch = pairs[start : start + args.batch_size]

            encoded = []
            for p in batch:
                query    = strip_query(p['query'])
                doc      = p['text'].strip()
                body     = f'<Instruct>: {args.task}\n<Query>: {query}\n<Document>: {doc}'
                body_ids = tokenizer.encode(body, add_special_tokens=False)[:max_body]
                encoded.append({'input_ids': prefix_ids + body_ids + suffix_ids})

            padded = tokenizer.pad(encoded, padding=True, return_tensors='pt')
            padded = {k: v.cuda() for k, v in padded.items()}

            logits = model(**padded).logits[:, -1, :]          # [batch, vocab]
            stack  = torch.stack([logits[:, token_false_id],
                                  logits[:, token_true_id]], dim=1)
            scores = torch.nn.functional.log_softmax(stack, dim=1)[:, 1].exp().tolist()

            for p, score in zip(batch, scores):
                out_f.write(f"{p['query_id']}\t{p['docid']}\t{score:.6f}\n")

    print(f'Done → {args.output}')


if __name__ == '__main__':
    main()
