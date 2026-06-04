# Kế hoạch Cải thiện repLLaMA — Recall & Query Format

> Tổng hợp từ session brainstorm ngày 2026-06-04.
> Mục tiêu: cải thiện Recall@K của retriever, chủ yếu qua thiết kế lại query/document format.

---

## 1. Tình trạng hiện tại

### 1.1 Kết quả tốt nhất đang có (Beauty test set)

| Model | NDCG@10 | HR@10 | HR@20 | MRR@10 |
|---|---|---|---|---|
| qwen3-embedding-0.6b-aug-5 | **0.0390** | **0.0905** | **0.1409** | 0.0235 |
| qwen3-embedding-0.6b (standard) | 0.0372 | 0.0861 | 0.1318 | 0.0224 |
| zero-shot | 0.0135 | 0.0315 | 0.0488 | 0.0081 |

Ceiling retrieval: **Recall@100 = 0.2673** → chỉ 26.7% queries có positive item trong top-100.

### 1.2 Nhận xét về kết quả ablation

- **Group size lớn hơn giúp** (gs32 > gs8): nhiều negatives per batch → tốt hơn
- **Augmentation giúp** (aug-5 > aug-3 > standard): ~7× training data so với 1 sample/user
- **BM25 negatives phá hủy hoàn toàn** (NDCG=0.0166 < random 0.0329): false negative poisoning
  - BM25 tìm items title-similar → đây là true near-items (cùng category/brand) → label nhầm thành negative → model học tránh retrieve items có title tương tự
- **Context size lớn hơn không luôn giúp**: cs10-gs32 (0.0356) < standard-gs32 (0.0360) — có thể do `--query-max-len=256` vẫn chưa đủ, hoặc format hiện tại không biểu diễn context dài hiệu quả

---

## 2. Phân tích nguyên nhân Recall thấp

### 2.1 History contamination — vấn đề nghiêm trọng nhất

**Số liệu đo trực tiếp** trên aug-5 best checkpoint, Beauty test set (22,363 queries):

| Chỉ số | Giá trị |
|---|---|
| History item ở **rank 1** | **92.5%** queries |
| ≥1 history item trong **top-5** | **99.5%** queries |
| ≥1 history item trong **top-10** | **99.8%** queries |
| Slots bị waste trung bình trong top-5 | **2.23 / 5 (44.6%)** |
| Slots bị waste trung bình trong top-10 | **2.72 / 10 (27.2%)** |

**Phân phối waste slots trong top-5:**

| Số history items trong top-5 | Queries | Tỷ lệ |
|---|---|---|
| 0 | 103 | 0.5% |
| 1 | 4,852 | 21.7% |
| **2** | **9,471** | **42.4% ← mode** |
| 3 | 6,235 | 27.9% |
| 4 | 1,129 | 5.0% |
| 5 (toàn history) | 573 | 2.6% |

**Nguyên nhân**: bi-encoder encode `"Title: item1, item2, item3"` → query embedding gần với embedding của item1, item2, item3 vì exact text overlap → FAISS trả về history items ở rank 1-2.

### 2.2 So sánh với SASRec — fairness analysis

- **SASRec trong RecBole**: `history_index = None` với sequential models → **KHÔNG** explicitly filter history items. Nhưng SASRec implicit tránh seen items vì ID-based architecture học sequence pattern.
- **repLLaMA**: structural bias — text similarity giữa query và history item titles → luôn có cosine similarity cao.
- **Kết luận**: filter history items khi evaluate là **fair và cần thiết**, không phải "cheating", vì bù đắp structural disadvantage của text-based retrieval. Cần document rõ trong paper.

### 2.3 Format query hiện tại có nhiều vấn đề

```
Query: item1, item2, item3 </s>
```

1. **Comma ambiguity**: item titles chứa dấu phẩy → không phân biệt được ranh giới items
2. **No task signal**: model không biết đây là recommendation task
3. **Thiếu metadata**: chỉ dùng title, bỏ qua category và brand
4. **Symmetric format**: query và document cùng format, trong khi Qwen3-Embedding thiết kế cho asymmetric retrieval
5. **`" </s>"` thừa**: Qwen3's EOS là `<|im_end|>`, được append bởi `--append_eos_token`. Literal `" </s>"` chỉ là noise tokens.

### 2.4 Training data sparse

- repLLaMA (standard): **22,363 samples** (1 sample/user)
- SASRec: sử dụng **tất cả positions** trong sequence (~7× more)
- Augmentation (aug) giải quyết một phần nhưng chưa triệt để

---

## 3. Quyết định thiết kế

### 3.1 Metadata available

**Amazon Beauty & Sports (2014 — filtered corpus 12,101 items):**

| Field | Coverage trong filtered corpus |
|---|---|
| `title` | 100% |
| `categories` | 100% — full hierarchy, e.g., `["Beauty", "Skin Care", "Face", "Peels"]` |
| `description` | 90% — truncated 200 chars |
| `brand` | **83%** (tốt hơn nhiều so với 49% trong raw data do 5-core filter) |
| `price` | 73% |
| `salesRank` | 98% — không có ý nghĩa semantic |

→ **Dùng**: title (100%) + categories full path (100%) + brand khi có (83%)

**MovieLens 1M:**

| Field | Coverage | Ghi chú |
|---|---|---|
| `title` | 100% | Bao gồm năm, e.g., `"Toy Story (1995)"` |
| `genres` | 100% | Flat list, 18 genres, pipe-separated |
| director/actor | **Không có** | Standard ML-1M chỉ có movies.dat + ratings.dat + users.dat |

→ **Dùng**: title (với năm) + genres (comma-separated)

→ Để có director/actor cần augment từ IMDb/TMDB — out of scope hiện tại.

### 3.2 Cold-start

5-core filter đảm bảo mọi user có ≥5 interactions → sau dedup N≥5 → train context tối thiểu 2 items (seq[:N-3] với N=5). **Không cần xử lý cold-start.**

---

## 4. Format mới — Đặc tả đầy đủ

### 4.1 Amazon Beauty & Sports

**Query format:**

```
Below is a customer's purchase history on Amazon, listed in chronological order (earliest to latest). Each item is represented by the following format: Title: <item title>. Category: <item category 1> > <item category 2> > .... Brand: <item brand>. (Brand field is omitted when unavailable.) Based on this history, predict only one item the customer is most likely to purchase next, described in the same format.
Purchase history:
Title: CND Shellac Top .5oz and Base .42oz. Category: Beauty > Nail Care > Nail Polish > Base & Top Coats. Brand: CND.
Title: CND Shellac Gel Nail Polish Zillionaire .25 Oz. Category: Beauty > Nail Care > Nail Polish > Gel Polish. Brand: CND.
Title: Creative Nail Shellac Gold Vip Status. Category: Beauty > Nail Care > Nail Polish > Gel Polish. Brand: Creative Nail Design.
```
*(Tevatron tự append `<|im_end|>` qua `--append_eos_token`)*

**Document format:**

```
Title: Travalo iPump5 Silver. Category: Beauty > Tools & Accessories > Travel Accessories. Brand: Travalo.
```
*(khi có brand)*

```
Title: White Plastic Jar with Dome Lid 2 Oz. Category: Beauty > Skin Care > Packaging > Jars.
```
*(khi không có brand — bỏ hoàn toàn field Brand, KHÔNG ghi "Brand: unknown")*

**Lý do bỏ "Passage: " prefix**: Document format mới dùng "Title: " là anchor, consistent với items trong query. "Passage: " là artifact của repLLaMA gốc (Llama-2 asymmetry), không cần thiết với Qwen3-Embedding last-token pooling.

### 4.2 MovieLens 1M

**Query format:**

```
Below is a customer's movie watch history, listed in chronological order (earliest to latest). Each item is represented by the following format: Title: <movie title>. Genre: <genre 1>, <genre 2>, .... Based on this history, predict only one movie the customer is most likely to watch next, described in the same format.
Watch history:
Title: Toy Story (1995). Genre: Animation, Children's, Comedy.
Title: Jumanji (1995). Genre: Adventure, Children's, Fantasy.
Title: Grumpier Old Men (1995). Genre: Comedy, Romance.
```

**Document format:**

```
Title: Toy Story 2 (1999). Genre: Animation, Children's, Comedy.
```

**Lý do dùng "Genre:" thay "Category:"**: ML-1M genres là flat list (không hierarchical) → `>` path không phù hợp. Genres dùng `,` separator.

### 4.3 Lý do thiết kế instruction text

**"predict" thay vì "retrieve":**
- "predict" capture được tính SEQUENTIAL/TEMPORAL ("what comes NEXT")
- "retrieve" là neutral IR framing, không capture "next item" signal
- Dù từ khác nhau, fine-tuning sẽ dạy model mapping đúng

**"described in the same format" là key:**
- Tạo format alignment: query embedding kỳ vọng "answer" có dạng `"Title: X. Category: Y. Brand: Z."`
- Document cũng có đúng dạng đó → cosine similarity tự nhiên cao hơn với đúng document
- Đây là kỹ thuật **format alignment** trong dense retrieval

**Mô tả format trong instruction:**
- Explicit format description: `"Title: <item title>. Category: <item category 1> > .... Brand: <item brand>. (Brand field is omitted when unavailable.)"`
- Dạy model hiểu structure của từng item, kể cả khi Brand vắng mặt

### 4.4 Token budget

**Instruction overhead:**
- Amazon instruction: **94 tokens**
- ML-1M instruction: **75 tokens**

**Per-item tokens (Amazon):** ~36 tokens (title≈15 + "Title: "=4 + category≈12 + " Category: "=4 + brand≈3 + " Brand: "=4 + separators≈3)

**Document tokens:** Amazon P95=55 tok, ML-1M P95=26 tok

| Dataset | Context size | Query tokens | `--query-max-len` | `--passage-max-len` |
|---|---|---|---|---|
| Beauty/Sports | cs=3 | ~194 tok | **256** | **128** |
| Beauty/Sports | cs=5 | ~266 tok | **320** | **128** |
| Beauty/Sports | cs=10 | ~446 tok | **512** | **128** |
| ML-1M | cs=3 | ~148 tok | **192** | **64** |
| ML-1M | cs=5 | ~188 tok | **256** | **64** |

> **Lưu ý**: Passage-max-len có thể giữ ở 128 cho tất cả để an toàn (không gây OOM, overhead nhỏ).

---

## 5. Kế hoạch implement

### 5.1 Các hàm cần sửa

#### `dataset/preprocess.py` — `build_item_text()`

**Hiện tại:**
```python
def build_item_text(item_id: str, item_meta: dict) -> str:
    meta = item_meta.get(item_id, {})
    title = meta.get("title") or ""
    if isinstance(title, list):
        title = " ".join(str(t) for t in title)
    title = str(title).strip()
    return title if title else item_id
```

**Cần đổi thành** (trả về full structured text):
```python
def build_item_text(item_id: str, item_meta: dict) -> str:
    meta = item_meta.get(item_id, {})
    
    # Title
    title = meta.get("title") or ""
    if isinstance(title, list):
        title = " ".join(str(t) for t in title)
    title = str(title).strip() or item_id
    
    # Category — dùng full path
    cat = str(meta.get("categories") or "").strip()
    
    # Brand — chỉ thêm khi có
    brand = str(meta.get("brand") or "").strip()
    
    text = f"Title: {title}."
    if cat:
        text += f" Category: {cat}."
    if brand:
        text += f" Brand: {brand}."
    return text
```

> **Lưu ý**: Hàm này được dùng ở NHIỀU nơi (corpus, query items, negatives). Thay đổi ở đây sẽ tự động apply nhất quán.

> **Lưu ý**: Với ML-1M, `load_movielens()` hiện đã lưu `"categories"` là genres string (e.g., `"Animation Children's Comedy"`). Cần sửa để dùng `"Genre: "` prefix thay vì `"Category: "` — cần thêm `dataset_type` context hoặc sửa `load_movielens()` để lưu field `"genres"` riêng.

**Hướng xử lý ML-1M**: Thêm optional field `"dataset_type"` vào item_meta hoặc sửa `build_item_text()` nhận thêm tham số `field_names` để customize label.

#### `dataset/export_tevatron.py` — `build_query()`

**Hiện tại:**
```python
def build_query(context: list, item_meta: dict) -> str:
    if not context:
        return "Query: </s>"
    texts = [build_item_text(iid, item_meta) for iid in context]
    return "Query: " + ", ".join(texts) + " </s>"
```

**Cần đổi thành** (instruction format):
```python
AMAZON_INSTRUCTION = (
    "Below is a customer's purchase history on Amazon, listed in chronological order "
    "(earliest to latest). Each item is represented by the following format: "
    "Title: <item title>. Category: <item category 1> > <item category 2> > .... "
    "Brand: <item brand>. (Brand field is omitted when unavailable.) "
    "Based on this history, predict only one item the customer is most likely to "
    "purchase next, described in the same format."
)

ML1M_INSTRUCTION = (
    "Below is a customer's movie watch history, listed in chronological order "
    "(earliest to latest). Each item is represented by the following format: "
    "Title: <movie title>. Genre: <genre 1>, <genre 2>, .... "
    "Based on this history, predict only one movie the customer is most likely to "
    "watch next, described in the same format."
)

def build_query(context: list, item_meta: dict, dataset_name: str = "amazon") -> str:
    if dataset_name == "ml-1m":
        instruction = ML1M_INSTRUCTION
        history_header = "Watch history:"
    else:
        instruction = AMAZON_INSTRUCTION
        history_header = "Purchase history:"
    
    items_text = "\n".join(
        build_item_text(iid, item_meta) for iid in context
    )
    return f"{instruction}\n{history_header}\n{items_text}"
    # Không có " </s>" — Tevatron append <|im_end|> qua --append_eos_token
```

> **Lưu ý quan trọng**: Bỏ `" </s>"` ở cuối. EOS thực là `<|im_end|>` được append bởi `--append_eos_token`. Literal `" </s>"` chỉ là noise tokens từ thời repLLaMA gốc.

#### `dataset/export_tevatron.py` — `make_passage()`

**Hiện tại:**
```python
def make_passage(item_id: str, item_meta: dict) -> dict:
    text = build_item_text(item_id, item_meta)
    return {"docid": item_id, "title": "", "text": f"Passage: {text}"}
```

**Cần đổi thành** (bỏ "Passage: " prefix):
```python
def make_passage(item_id: str, item_meta: dict) -> dict:
    text = build_item_text(item_id, item_meta)
    return {"docid": item_id, "title": "", "text": text}
    # text đã có dạng "Title: X. Category: Y. [Brand: Z.]"
    # "Passage: " không cần thiết với Qwen3-Embedding last-token pooling
```

**Corpus.jsonl** cũng cần update (hiện dùng `f"Passage: {text}"` trực tiếp):
```python
# Trong corpus export loop:
f.write(json.dumps(
    {"docid": item_id, "text": build_item_text(item_id, item_meta)},
    ensure_ascii=False,
) + "\n")
```

### 5.2 Cập nhật `train.sh` và `eval.sh`

Thêm mapping dataset → query-max-len mới:

```bash
# train.sh — query-max-len defaults mới theo dataset và context_size
# cs=3 (default): beauty/sports=256, ml-1m=192
# cs=5:           beauty/sports=320, ml-1m=256
# cs=10:          beauty/sports=512, ml-1m=320
```

Passage-max-len: có thể giữ 196 (không gây vấn đề) hoặc giảm xuống 128.

### 5.3 Thứ tự thực hiện thực nghiệm sau implement

```
1. Export data mới với format mới (beauty, sports, ml-1m)
   cd dataset && python export_tevatron.py beauty --tag v2
   cd dataset && python export_tevatron.py beauty --augment --tag v2-aug

2. Zero-shot eval (baseline không fine-tune)
   ./eval.sh beauty base --tag v2
   # So sánh: zero-shot hiện tại HR@10=0.0315 → kỳ vọng tăng đáng kể

3. Fine-tune với format mới
   ./train.sh beauty --data-variant v2 --tag v2 --query-max-len 256
   ./eval.sh beauty --tag v2

4. Fine-tune với augmentation
   ./train.sh beauty --data-variant v2-aug --tag v2-aug --query-max-len 256
   ./eval.sh beauty --tag v2-aug

5. So sánh kết quả qua show_results.py
```

---

## 6. Các hướng cải thiện khác (chưa implement, để tham khảo)

### 6.1 Filter history items — post-processing (đã có plan ở `reranker_redesign_plan.md`)

**Số liệu hỗ trợ**: 44.6% top-5 bị waste bởi history items.

Implement đơn giản: sau FAISS search, loại bỏ items có text match với query history trước khi trả về top-K. Không cần train lại. Xem `reranker_redesign_plan.md` section A1 để biết matching logic chi tiết.

**Cần implement song song**: cả ở retriever evaluation (`eval.sh`) lẫn reranker input (`prepare_rerank_data.py`).

### 6.2 Multi-position training

SASRec dùng toàn bộ positions trong sequence, repLLaMA chỉ 1 sample/user. Augmentation (`--augment`) giải quyết một phần nhưng chưa cover đủ. Cần đảm bảo augmented positions không leak valid/test items.

**Hiện trạng**: `--augment` trong `export_tevatron.py` đã implement sliding window nhưng dùng `range(1, N-2)` (không leak). Đây là đúng rồi, chỉ cần dùng.

### 6.3 Hard negative mining từ model

Sau khi train với random negatives, mine top-K từ trained model làm hard negatives cho epoch tiếp theo. Tránh false negative problem của BM25 vì mine từ embedding space thay vì text similarity.

### 6.4 Tăng retrieval depth K = 200

Recall@100 = 0.2673. Recall@200 ước tính ~35%. Không cần train lại — chỉ sửa `--depth` trong `eval.sh` và reranker. Free improvement.

---

## 7. Checklist implement

### Phase 1 — Format redesign (ưu tiên cao nhất)
- [ ] Sửa `build_item_text()` trong `preprocess.py`
- [ ] Xử lý ML-1M dataset type (Genre vs Category label)
- [ ] Sửa `build_query()` trong `export_tevatron.py`
- [ ] Sửa `make_passage()` trong `export_tevatron.py`
- [ ] Sửa corpus export loop trong `export_tevatron.py`
- [ ] Update `--query-max-len` defaults trong `train.sh`
- [ ] Export data mới: beauty, sports, ml-1m (standard + augmented variants)
- [ ] Chạy zero-shot eval để đo baseline improvement ngay lập tức
- [ ] Train + eval: beauty v2, beauty v2-aug

### Phase 2 — History filtering
- [ ] Implement filter-history post-processing trong `eval.sh` / evaluation pipeline
- [ ] Implement A1 trong `reranker_redesign_plan.md`
- [ ] Đo improvement khi filter history (kỳ vọng HR@10 tăng đáng kể)

### Phase 3 — Scaling và ablation
- [ ] Thử cs=5 với format mới
- [ ] Thử K=200 retrieval depth
- [ ] So sánh Beauty vs Sports vs ML-1M với format mới
- [ ] Chạy reranker pipeline (A1+C1+D1 từ `reranker_redesign_plan.md`)

---

## 8. Files liên quan trong project

| File | Vai trò | Cần sửa? |
|---|---|---|
| `dataset/preprocess.py` | `build_item_text()` — tạo text từ metadata | **CÓ** |
| `dataset/export_tevatron.py` | `build_query()`, `make_passage()`, corpus loop | **CÓ** |
| `train.sh` | `--query-max-len` defaults | **CÓ** |
| `eval.sh` | `--query-max-len` defaults | **CÓ** |
| `reranker_redesign_plan.md` | Plan A1 (filter history), C1, D1, D3 | Đọc để implement Phase 2 |
| `experiments.md` | Bảng kết quả, cập nhật sau khi train | Thêm experiments mới |
| `show_results.py` | Tổng hợp kết quả tự động | Không cần sửa |
