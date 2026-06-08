#!/usr/bin/env python3
"""
train_reranker_qwen3.py — Fine-tune Qwen3-Reranker cho sequential recommendation.

Loss: InfoNCE trên raw yes-logits.
  - score[i] = logits[i, -1, token_yes]            (1 scalar / pair)
  - scores.view(batch, group_size)                  ([batch, 1_pos + N_negs])
  - loss = cross_entropy(scores, labels=0)          (InfoNCE: maximize yes-logit của positive)

Training data: reranker_train.jsonl (sinh bởi train_reranker_qwen3.sh hoặc train_reranker.sh)
  {"query_id":..., "query":"Query: ...",
   "positive_passages":[{"docid":..., "text":...}],
   "negative_passages":[...]}
"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm


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


def encode_pair(query, doc_text, prefix_ids, suffix_ids, tokenizer, task, max_length):
    max_body = max_length - len(prefix_ids) - len(suffix_ids)
    body     = f'<Instruct>: {task}\n<Query>: {query}\n<Document>: {doc_text}'
    body_ids = tokenizer.encode(body, add_special_tokens=False)[:max_body]
    return prefix_ids + body_ids + suffix_ids


class RerankerDataset(Dataset):
    def __init__(self, path, tokenizer, task, prefix_ids, suffix_ids, group_size, max_length):
        self.tokenizer  = tokenizer
        self.task       = task
        self.prefix_ids = prefix_ids
        self.suffix_ids = suffix_ids
        self.group_size = group_size
        self.max_length = max_length
        self.examples   = []

        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d.get('positive_passages') and d.get('negative_passages'):
                    self.examples.append(d)

        print(f'  {len(self.examples):,} training examples loaded')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        d     = self.examples[idx]
        query = strip_query(d['query'])

        pos  = random.choice(d['positive_passages'])
        negs = d['negative_passages']
        n    = self.group_size - 1
        if len(negs) >= n:
            negs = random.sample(negs, n)
        else:
            negs = negs + random.choices(negs, k=n - len(negs))

        # Positive always at index 0 → labels = zeros
        return [
            encode_pair(query, p['text'], self.prefix_ids, self.suffix_ids,
                        self.tokenizer, self.task, self.max_length)
            for p in [pos] + negs
        ]


def collate(batch, pad_id):
    seqs = [seq for group in batch for seq in group]
    max_len        = max(len(s) for s in seqs)
    input_ids      = []
    attention_mask = []
    for s in seqs:
        pad = max_len - len(s)
        input_ids.append([pad_id] * pad + s)           # left padding
        attention_mask.append([0] * pad + [1] * len(s))
    return {
        'input_ids':      torch.tensor(input_ids,      dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data',   required=True)
    parser.add_argument('--output_dir',   required=True)
    parser.add_argument('--model',        default='Qwen/Qwen3-Reranker-0.6B')
    parser.add_argument('--task',         default=TASK)
    parser.add_argument('--group_size',   type=int,   default=4)
    parser.add_argument('--max_length',   type=int,   default=256)
    parser.add_argument('--epochs',       type=int,   default=3)
    parser.add_argument('--lr',           type=float, default=5e-5)
    parser.add_argument('--per_batch',    type=int,   default=1,
                        help='Queries per GPU step (mặc định 1 — mỗi step xử lý group_size pairs)')
    parser.add_argument('--grad_accum',   type=int,   default=32)
    parser.add_argument('--warmup_steps', type=int,   default=100)
    parser.add_argument('--save_steps',   type=int,   default=500)
    parser.add_argument('--lora_r',       type=int,   default=16)
    parser.add_argument('--lora_alpha',   type=int,   default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Model      : {args.model}')
    print(f'Group size : {args.group_size}  (1 pos + {args.group_size - 1} hard negs)')
    print(f'Max length : {args.max_length} tokens')
    print(f'Epochs     : {args.epochs}')
    print(f'LR         : {args.lr}')

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'down_proj', 'up_proj', 'gate_proj'],
        bias='none',
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = model.cuda()
    model.print_trainable_parameters()

    token_true_id  = tokenizer.convert_tokens_to_ids('yes')
    token_false_id = tokenizer.convert_tokens_to_ids('no')
    print(f'Token IDs  : yes={token_true_id}, no={token_false_id}\n')

    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)

    print('Loading training data...')
    dataset    = RerankerDataset(args.train_data, tokenizer, args.task,
                                  prefix_ids, suffix_ids, args.group_size, args.max_length)
    dataloader = DataLoader(
        dataset, batch_size=args.per_batch, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    optimizer   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = len(dataloader) * args.epochs // args.grad_accum
    scheduler   = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    print(f'Total optimizer steps : {total_steps}  '
          f'({len(dataloader)} batches × {args.epochs} epochs ÷ {args.grad_accum} accum)\n')

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for step, batch in enumerate(pbar):
            batch = {k: v.cuda() for k, v in batch.items()}

            # logits: [per_batch * group_size, seq_len, vocab]
            logits     = model(**batch).logits[:, -1, :]           # [N, vocab]
            yes_logits = logits[:, token_true_id]                  # [N]
            scores     = yes_logits.view(args.per_batch, args.group_size)  # [B, G]

            # InfoNCE: positive at index 0
            labels = torch.zeros(args.per_batch, dtype=torch.long, device=scores.device)
            loss   = F.cross_entropy(scores, labels) / args.grad_accum
            loss.backward()

            running_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                pbar.set_postfix({'loss': f'{running_loss / (step + 1):.4f}',
                                  'step': global_step})

                if global_step % args.save_steps == 0:
                    ckpt = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                    model.save_pretrained(ckpt)
                    pbar.write(f'  → Saved {ckpt}')

        avg = running_loss / len(dataloader)
        print(f'Epoch {epoch + 1} — avg loss: {avg:.4f}')

    model.save_pretrained(args.output_dir)
    print(f'\nFinal model saved → {args.output_dir}')


if __name__ == '__main__':
    main()
