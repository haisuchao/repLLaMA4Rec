# Kế hoạch Thiết kế lại Reranker

> Tài liệu này mô tả thiết kế chi tiết cho từng thành phần thay đổi.
> Đọc và góp ý trước khi implement.

---

## Tổng quan các thử nghiệm

| ID | Thành phần | Files thay đổi | Bắt buộc train lại? |
|---|---|---|---|
| **A1** | Pre-filter history items | `prepare_rerank_data.py`, `rerank.sh`, `rerank_qwen3.sh` | Không |
| **C1** | Augmented training data | `train_reranker.sh` | Có |
| **D1** | Score combination (retriever + reranker) | `rerank.sh`, `rerank_qwen3.sh` | Không |
| **D3** | Listwise sliding window | `rerank_listwise.py` (new), `rerank_listwise.sh` (new) | Không |

**Workflow các tổ hợp:**

```bash
# A1 alone — dùng model đã train sẵn, thêm filter
./rerank.sh beauty --tag aug-5 --filter-history

# C1 alone — train lại reranker với aug data, không filter
./train_reranker.sh beauty --tag aug-5 --aug-tag aug-3
./rerank.sh beauty --tag aug-5 --aug-tag aug-3

# D3 alone — listwise reranking
./rerank_listwise.sh beauty --retriever-tag aug-5

# A1 + C1
./train_reranker.sh beauty --tag aug-5 --aug-tag aug-3
./rerank.sh beauty --tag aug-5 --aug-tag aug-3 --filter-history

# A1 + D1
./rerank.sh beauty --tag aug-5 --filter-history --combine-alpha 0.5

# A1 + C1 + D1
./train_reranker.sh beauty --tag aug-5 --aug-tag aug-3
./rerank.sh beauty --tag aug-5 --aug-tag aug-3 --filter-history --combine-alpha 0.5
```

---

## A1: Pre-filter History Items

### Vấn đề giải quyết

Retriever xếp history items (items đã trong query) lên rank 1-2 vì cosine similarity
cao giữa query embedding và item embedding. Cross-encoder thấy exact token match
giữa query `"item1, item2, item3"` và candidate `"item1"` → cho điểm cao nhầm.

### Cơ chế

Trong `prepare_rerank_data.py` mode `infer`, thêm `--filter-history` flag:

1. **Parse history từ query string:**
   ```
   "Query: Pineapple Peel, LAVANILA, Waxelene </s>"
   → ["Pineapple Peel", "LAVANILA", "Waxelene"]
   ```

2. **Matching logic — substring containment (case-insensitive):**
   ```python
   cand = candidate_text.lower().strip()
   for h in history_items:
       if cand == h or cand in h or h in cand:
           return True  # skip this candidate
   ```
   Lý do dùng substring thay vì exact match: corpus item có thể có title dài hơn
   version trong query (bị truncate). Ví dụ query có "Waxelene 2oz jar" nhưng
   corpus có "Waxelene 2oz jar - Natural Petrolatum Alternative".

3. **Report stats:** In ra số candidates bị filter mỗi query (để debug).

### Files thay đổi

#### `prepare_rerank_data.py`

Thêm 2 hàm helper + flag `--filter-history`:

```python
def extract_history_texts(query_str):
    # "Query: a, b, c </s>" → ["a", "b", "c"]
    q = query_str.strip()
    if q.startswith('Query: '): q = q[7:]
    for suffix in (' </s>', '</s>'):
        if q.endswith(suffix): q = q[:-len(suffix)].strip(); break
    return [t.strip() for t in q.split(', ') if t.strip()]

def is_history_item(candidate_text, history_texts):
    cand = candidate_text.lower().strip()
    for h in history_texts:
        h_norm = h.lower().strip()
        if cand == h_norm or cand in h_norm or h_norm in cand:
            return True
    return False
```

Trong main loop, mode infer:
```python
if args.filter_history:
    history = extract_history_texts(query)
    cands = [d for d in cands if not is_history_item(corpus.get(d, ''), history)]
```

#### `rerank.sh` và `rerank_qwen3.sh`

Thêm `--filter-history` flag → pass vào `prepare_rerank_data.py`.

Khi `--filter-history` bật, output files được đặt tên khác để không ghi đè:
- `test_pairs.jsonl` → `test_pairs-fh.jsonl`
- `eval_test_reranked.txt` → `eval_test_reranked-fh.txt`

### Kỳ vọng

Với 22K test queries và retriever xếp history items lên rank 1-2 trong nhiều queries,
filter sẽ đẩy positive item lên ít nhất 1-2 rank. Nếu positive ở rank 11 (vừa miss HR@10)
và rank 1-2 là history items → sau filter positive lên rank 9 → hit HR@10.

Theo ceiling analysis: không cần train thêm gì, chỉ cần bước này.

---

## C1: Augmented Training Data cho Reranker

### Vấn đề giải quyết

Reranker hiện train với 22,363 samples (1 per user). Dataset `beauty-aug-3` có
131,413 samples (5.9× nhiều hơn). Retriever aug-5 đã cho thấy augmentation giúp
từ NDCG 0.0372 → 0.0390 (+4.8%), logic tương tự nên áp dụng cho reranker.

### Cơ chế

Thêm `--aug-tag TAG` vào `train_reranker.sh`. Khi set:
- Step 2: Encode train queries từ `dataset/tevatron/{dataset}-{aug_tag}/train.jsonl`
- Step 3: FAISS search để mine hard negatives từ augmented queries
- Step 4: `prepare_rerank_data.py` dùng augmented train.jsonl

Available aug datasets:
- `beauty-aug-3` (131K samples) ← khuyến nghị
- `beauty-cs5-aug` (context_size=5 + aug)
- `sports-aug-3`, `ml-1m-aug-3`

### Files thay đổi

#### `train_reranker.sh`

```bash
# Tham số mới
aug_tag=""
--aug-tag)   aug_tag="$2";   shift 2 ;;

# Output dir có suffix
RERANKER_DIR="${RETRIEVER_DIR}-reranker"
[ -n "${aug_tag}" ] && RERANKER_DIR="${RERANKER_DIR}-${aug_tag}"

# Training data
if [ -n "${aug_tag}" ]; then
  AUG_DATA_DIR="./dataset/dataset/tevatron/${dataset}-${aug_tag}"
  [ ! -d "${AUG_DATA_DIR}" ] && echo "Lỗi: không tìm thấy ${AUG_DATA_DIR}" && exit 1
  TRAIN_JSONL="${AUG_DATA_DIR}/train.jsonl"
else
  TRAIN_JSONL="${DATA_DIR}/train.jsonl"
fi
```

#### `rerank.sh`

Thêm `--aug-tag TAG` để trỏ đến đúng reranker directory:
```bash
aug_tag=""
--aug-tag)   aug_tag="$2";   shift 2 ;;

RERANKER_DIR="${RETRIEVER_DIR}-reranker"
[ -n "${aug_tag}" ] && RERANKER_DIR="${RERANKER_DIR}-${aug_tag}"
```

### Naming convention

```
# Retriever: aug-5 (đã có)
output/beauty/qwen3-embedding-0.6b-aug-5/

# Reranker cũ (trained with standard data, 22K):
output/beauty/qwen3-embedding-0.6b-aug-5-reranker/

# Reranker C1 (trained with aug-3 data, 131K):
output/beauty/qwen3-embedding-0.6b-aug-5-reranker-aug-3/
```

### Thời gian train ước tính

Standard reranker: 22K × 3 epochs = 67K samples → ~30-40 phút
C1 reranker: 131K × 3 epochs = 393K samples → ~3-4 giờ

### Kỳ vọng

Với 5.9× training data và hard negatives từ augmented queries (đa dạng hơn về context),
reranker nên học được discriminative signal tốt hơn. Kỳ vọng cải thiện tương tự như
augmentation đã giúp retriever (~4-5% NDCG).

---

## D1: Score Combination (Retriever + Reranker)

### Vấn đề giải quyết

Retriever và reranker capture different signals:
- Retriever: global semantic similarity (embedding-based)
- Reranker: fine-grained cross-attention matching

Combining có thể tận dụng strength của cả hai. Đặc biệt khi reranker sau A1 đã
loại bỏ history items, combination giúp calibrate ranking tốt hơn.

### Cơ chế

Sau bước reranking, thêm bước combination (inline Python trong shell script):

```python
# 1. Load retriever scores từ trec file
# 2. Load reranker scores từ reranked file
# 3. Min-max normalize per query
# 4. Combined = alpha * ret_norm + (1-alpha) * rer_norm
# 5. Re-sort → new trec file
```

**Normalization:** Min-max per query (không dùng global normalization vì scores
của các queries không so sánh được trực tiếp):
```python
vals = list(scores.values())
mn, mx = min(vals), max(vals)
norm = {d: (s - mn) / (mx - mn + 1e-8) for d, s in scores.items()}
```

**Alpha values để test:** 0.0 (pure reranker), 0.3, 0.5, 0.7, 1.0 (pure retriever)
→ tune trên valid set, report kết quả tốt nhất trên test.

### Files thay đổi

#### `rerank.sh`

Thêm `--combine-alpha FLOAT` flag. Khi set:
- Step 3 thêm sub-step: combination trước khi evaluate
- Output file: `eval_test_reranked-ca{alpha}.txt` hoặc `eval_test_reranked-fh-ca{alpha}.txt`

```bash
combine_alpha=""
--combine-alpha) combine_alpha="$2"; shift 2 ;;

# Variant suffix cho output files
VARIANT=""
[ "${filter_history}" = "true" ] && VARIANT="${VARIANT}-fh"
[ -n "${combine_alpha}" ] && VARIANT="${VARIANT}-ca$(echo ${combine_alpha} | tr -d '.')"
```

Inline Python cho combination step (sau khi có RERANKED_TREC):
```python
# Load retriever + reranker scores
# Normalize per query
# Combine: alpha*ret + (1-alpha)*rer
# Write combined_trec → dùng thay vì RERANKED_TREC khi evaluate
```

### Lưu ý

- Khi dùng cùng A1 (filter history): retriever trec vẫn có scores cho tất cả 100 candidates,
  reranked trec chỉ có scores cho filtered candidates (ít hơn). Combination chỉ apply cho
  candidates có trong cả hai → OK vì đây là tập cần rank.

- Không thêm D1 vào `rerank_qwen3.sh` lần này để đơn giản — `rerank_qwen3.sh` đã có
  separate design. Có thể thêm sau nếu cần.

---

## D3: Listwise Sliding Window Reranking

### Vấn đề giải quyết

Pointwise reranker (hiện tại) score mỗi (query, candidate) pair độc lập — không
so sánh candidates với nhau. Listwise reranker nhìn NHIỀU candidates cùng lúc →
forced comparison → calibrated ranking.

**Quan trọng:** Khi model thấy tất cả candidates trong một context, nó có thể:
- Nhận ra candidates nào trùng với history (intra-window signal)
- So sánh items với nhau trực tiếp ("item A vs item B, which is more likely next?")

### Cơ chế: Sliding Window (RankGPT style)

```
Candidates: [1, 2, 3, 4, ..., 100]  (sorted by retriever score)

Window passes (window_size=20, step=10):
  Pass 1: rank [81-100] → LLM reorders → apply to list
  Pass 2: rank [71-90]  → LLM reorders → apply
  Pass 3: rank [61-80]  → LLM reorders → apply
  ...
  Pass 9: rank [1-20]   → LLM reorders → apply

Result: fully reranked list
```

Tại sao bottom-to-top? Vì items ở top (sau pass cuối) được rerank tốt nhất.
Mỗi pass, "winners" của window trước có cơ hội được đẩy lên cao hơn.

### Prompt format (Qwen3 chat template)

```
<|im_start|>system
You are an expert at predicting sequential purchases.<|im_end|>
<|im_start|>user
A user recently purchased (in chronological order): {history}

Rank the following {N} candidate items from MOST to LEAST likely to be the user's
immediate NEXT purchase. The next purchase must be a NEW item not in the history.

[1] candidate_title_1
[2] candidate_title_2
...
[N] candidate_title_N

Reply with ONLY the re-ranked item numbers separated by ">", most likely first.
Example: 3>1>7>2>...
<|im_end|>
<|im_start|>assistant
```

**Tại sao ">"?** Dễ parse, ít bị lẫn với số có dấu phẩy trong item titles.

### Output parsing (robust)

```python
def parse_ranking(output_text, n_items):
    # Try ">" format: "3>1>5>2"
    nums = re.findall(r'\d+', output_text.replace('>', ' ').replace(',', ' '))
    indices, seen = [], set()
    for n in nums:
        idx = int(n) - 1   # 1-indexed → 0-indexed
        if 0 <= idx < n_items and idx not in seen:
            indices.append(idx); seen.add(idx)
    # Fill missing indices at the end (model có thể bỏ qua một số items)
    for i in range(n_items):
        if i not in seen: indices.append(i)
    return indices
```

### Batching

Để không chạy quá chậm, batch nhiều windows từ nhiều queries trong cùng 1 forward pass:

```python
# Collect ALL windows from ALL queries
all_windows = []  # (qid, window_start_idx, window_items)
for qid, ranked_items in all_queries.items():
    for start in sliding_window_positions(len(ranked_items), window_size, step_size):
        all_windows.append((qid, start, ranked_items[start:start+window_size]))

# Process in batches of batch_size
for i in range(0, len(all_windows), batch_size):
    batch = all_windows[i:i+batch_size]
    prompts = [build_prompt(q_history[qid], items) for qid, start, items in batch]
    rankings = generate_batch(model, tokenizer, prompts)
    # Apply rankings back
    for (qid, start, items), ranking in zip(batch, rankings):
        reorder_in_place(all_queries[qid], start, ranking)
```

### Tốc độ ước tính

Với Beauty test set (22K queries), window_size=20, step=10:
- 9 windows per query × 22K queries = ~198K windows
- batch_size=8 → ~25K batches
- Với Qwen3-0.6B, mỗi batch ~0.5s → ~12K giây → ~3.5 giờ

**Quá chậm cho test nhanh!** Giải pháp:

1. `--num-queries N` parameter (default: None = all) để test nhanh với subset.
   Ví dụ: `--num-queries 500` → ~8 phút.

2. `--window-size N` và `--step-size N` parameters để tune speed/quality tradeoff.
   Ví dụ: window=10, step=10 → chỉ ~10 passes, nhanh hơn 2×.

3. **Full evaluation:** Chạy overnight hoặc với `batch_size=16`.

### Model mặc định

`Qwen/Qwen3-Reranker-0.6B` — đã có sẵn trong cache, nhỏ, hỗ trợ Qwen3 chat format.

Mặc dù được fine-tune cho yes/no scoring, kiến trúc Qwen3 vẫn có thể generate text
theo instruction. Nếu không follow format tốt, thử `Qwen/Qwen2.5-1.5B` (cũng đã cache).

### Files mới

#### `rerank_listwise.py`

```
Input:  pairs jsonl (từ prepare_rerank_data.py --mode infer)
Output: qid\tdocid\tscore (score = 1/rank)
Flags:
  --pairs PATH          : input pairs jsonl
  --output PATH         : output scores file
  --model MODEL         : generative model (default: Qwen/Qwen3-Reranker-0.6B)
  --window-size N       : items per ranking window (default: 20)
  --step-size N         : window step (default: 10)
  --batch-size N        : windows per GPU batch (default: 8)
  --max-new-tokens N    : max tokens to generate per window (default: 80)
  --num-queries N       : limit to first N queries (default: all)
  --filter-history      : skip candidates in purchase history before ranking
```

#### `rerank_listwise.sh`

Pipeline:
1. Tìm retriever trec file (same as rerank.sh)
2. Prepare pairs (reuse nếu có, same pairs.jsonl as rankLLaMA/qwen3)
3. `python rerank_listwise.py` → `test_listwise.txt`
4. Convert → trec + evaluate
5. So sánh với retriever

```
Flags:
  <dataset>
  --retriever-model MODEL
  --retriever-tag TAG
  --gen-model MODEL      (generative model cho listwise, default: Qwen/Qwen3-Reranker-0.6B)
  --window-size N
  --step-size N
  --batch-size N
  --num-queries N
  --filter-history
  --split SPLIT
  --force
```

Output dir: `output/{dataset}/{retriever_tag}-listwise/`

### Note về D3 vs A1+D3

Nếu kết hợp `--filter-history` với listwise: trước khi ranking window, đã loại bỏ
history items → model không cần "học" downrank history, chỉ cần focus phân biệt
true next item vs semantic negatives.

---

## Tóm tắt output naming

| Command | Output dir | Eval file |
|---|---|---|
| `./rerank.sh beauty --tag aug-5` | `...aug-5-reranker/inference/` | `eval_test_reranked.txt` |
| `./rerank.sh beauty --tag aug-5 --filter-history` | same dir | `eval_test_reranked-fh.txt` |
| `./rerank.sh beauty --tag aug-5 --combine-alpha 0.5` | same dir | `eval_test_reranked-ca05.txt` |
| `./rerank.sh beauty --tag aug-5 --filter-history --combine-alpha 0.5` | same dir | `eval_test_reranked-fh-ca05.txt` |
| `./train_reranker.sh beauty --tag aug-5 --aug-tag aug-3` | `...aug-5-reranker-aug-3/` | — |
| `./rerank.sh beauty --tag aug-5 --aug-tag aug-3` | `...aug-5-reranker-aug-3/inference/` | `eval_test_reranked.txt` |
| `./rerank.sh beauty --tag aug-5 --aug-tag aug-3 --filter-history` | same aug-3 dir | `eval_test_reranked-fh.txt` |
| `./rerank.sh beauty --tag aug-5 --aug-tag aug-3 --filter-history --combine-alpha 0.5` | same aug-3 dir | `eval_test_reranked-fh-ca05.txt` |
| `./rerank_listwise.sh beauty --retriever-tag aug-5` | `...aug-5-listwise/` | `eval_test_listwise.txt` |
| `./rerank_listwise.sh beauty --retriever-tag aug-5 --filter-history` | same listwise dir | `eval_test_listwise-fh.txt` |

---

## Thứ tự thực hiện thực nghiệm (đề xuất)

```
1. A1 alone (~5 phút)
   → ./rerank.sh beauty --tag aug-5 --filter-history
   → ./rerank.sh beauty --filter-history  (standard retriever)

2. A1 + D1 sweep alpha (~15 phút)
   → ./rerank.sh beauty --tag aug-5 --filter-history --combine-alpha 0.3
   → ./rerank.sh beauty --tag aug-5 --filter-history --combine-alpha 0.5
   → ./rerank.sh beauty --tag aug-5 --filter-history --combine-alpha 0.7

3. D3 quick test (~10 phút với --num-queries 500)
   → ./rerank_listwise.sh beauty --retriever-tag aug-5 --num-queries 500

4. C1 train + eval (~3-4 giờ train + ~30 phút eval)
   → ./train_reranker.sh beauty --tag aug-5 --aug-tag aug-3
   → ./rerank.sh beauty --tag aug-5 --aug-tag aug-3

5. A1 + C1 (~30 phút eval, model từ bước 4)
   → ./rerank.sh beauty --tag aug-5 --aug-tag aug-3 --filter-history

6. A1 + C1 + D1 (~30 phút eval)
   → ./rerank.sh beauty --tag aug-5 --aug-tag aug-3 --filter-history --combine-alpha 0.5

7. D3 full (~3-4 giờ)
   → ./rerank_listwise.sh beauty --retriever-tag aug-5
```

---

## Câu hỏi cần xác nhận trước khi implement

1. **A1 matching logic:** Dùng substring containment (A in B or B in A) hay chỉ exact match?
   - Substring dễ bị false positive nếu titles ngắn (ví dụ: "Oil" match với "Tea Tree Oil")
   - Đề xuất: **exact match first, then substring chỉ khi len > 10 chars**

2. **C1 aug-tag:** Dùng `beauty-aug-3` (context=3 + aug, 131K samples)?
   Hay `beauty-cs5-aug` (context=5 + aug)? Hay thêm option cho người dùng chọn?
   - Đề xuất: mặc định `aug-3` (đã có sẵn, dataset lớn nhất hiện có)

3. **D1 alpha:** Test tất cả [0.1, 0.3, 0.5, 0.7, 0.9] trên valid set rồi chọn best?
   Hay chỉ test một vài giá trị?
   - Đề xuất: test 3 giá trị đại diện (0.3, 0.5, 0.7) + report tất cả

4. **D3 model:** Dùng `Qwen/Qwen3-Reranker-0.6B` hay `Qwen/Qwen2.5-1.5B` cho listwise?
   - Đề xuất: **thử cả hai** — thêm `--gen-model` flag để switch

5. **D3 full evaluation:** Bạn có muốn chạy D3 trên full 22K test queries không?
   (~3-4 giờ) Hay chỉ cần test nhanh với 500-1000 queries để đánh giá tiềm năng?

6. **rerank_qwen3.sh:** Có muốn thêm `--filter-history` cho Qwen3-Reranker pipeline
   không? (cùng cơ chế, chỉ cần vài dòng)
   - Đề xuất: có, vì A1 nên test trên tất cả rerankers

7. **D3 + A1:** Khi filter-history trong listwise, filter trước khi tạo windows
   hay filter từng window? 
   - Đề xuất: **filter trước**, tức bỏ history items ra khỏi candidate list hoàn toàn,
     sau đó mới apply sliding window trên filtered list.
