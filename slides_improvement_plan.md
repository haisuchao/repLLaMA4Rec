---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 22px;
    padding: 40px 50px;
  }
  section.lead {
    text-align: center;
    justify-content: center;
  }
  h1 { color: #1a237e; font-size: 1.8em; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }
  h2 { color: #1565c0; font-size: 1.4em; }
  h3 { color: #0277bd; font-size: 1.1em; margin-top: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
  th { background: #1a237e; color: white; padding: 7px 10px; }
  td { padding: 6px 10px; border-bottom: 1px solid #ddd; }
  tr:nth-child(even) td { background: #f5f7ff; }
  code { background: #f0f4f8; padding: 2px 6px; border-radius: 3px; font-size: 0.85em; }
  pre { background: #f0f4f8; border-left: 4px solid #1565c0; padding: 14px 16px; font-size: 0.78em; border-radius: 4px; }
  .highlight { background: #fff3e0; border-left: 4px solid #f57c00; padding: 10px 14px; border-radius: 4px; margin: 10px 0; }
  .danger { background: #ffebee; border-left: 4px solid #c62828; padding: 10px 14px; border-radius: 4px; margin: 10px 0; }
  .good { background: #e8f5e9; border-left: 4px solid #2e7d32; padding: 10px 14px; border-radius: 4px; margin: 10px 0; }
  strong { color: #c62828; }
  footer { font-size: 0.7em; color: #666; }
---

<!-- _class: lead -->

# repLLaMA — Improvement Plan
## Cải thiện Recall & Thiết kế lại Query/Document Format

**Sequential Recommendation via Dense Retrieval**

2026-06-05


---

# Agenda

1. **Tình trạng hiện tại** — Kết quả và những gì đã thử
2. **Phân tích nguyên nhân** — 4 vấn đề gốc rễ
3. **Giải pháp chính** — Thiết kế lại query & document format
4. **Metadata available** — Dữ liệu có thể dùng
5. **Đặc tả format mới** — Amazon và ML-1M
6. **Kế hoạch implement** — Code cụ thể cần sửa
7. **Thực nghiệm & Timeline**
8. **Các hướng cải thiện khác**

---

# 1. Tình trạng hiện tại

## Kết quả tốt nhất — Amazon Beauty (test set, 12,101 items)

| Model | NDCG@10 | HR@10 | HR@20 | MRR@10 |
|---|:---:|:---:|:---:|:---:|
| `qwen3-embedding-0.6b` · **aug-5** | **0.0390** | **0.0905** | **0.1409** | 0.0235 |
| `qwen3-embedding-0.6b` · standard | 0.0372 | 0.0861 | 0.1318 | 0.0224 |
| `qwen3-embedding-0.6b` · zero-shot | 0.0135 | 0.0315 | 0.0488 | 0.0081 |
| `llama-3.2-1b` · fine-tuned | 0.0313 | 0.0736 | 0.1148 | 0.0185 |
| `qwen3-embedding-0.6b` · **bm25** neg | 0.0209 | 0.0388 | 0.0626 | 0.0154 |

> **Recall@100 = 0.2673** — chỉ **26.7%** queries có item đúng trong top-100 retrieved items

---

# 1. Tình trạng hiện tại

## Kết quả Ablation — Những gì đã học được

| Thực nghiệm | NDCG@10 | Nhận xét |
|---|:---:|---|
| gs32 (31 negatives) | 0.0360 | Group size lớn **giúp** (vs gs8: 0.0329) |
| aug-5 (7× training data) | **0.0390** | Augmentation **giúp đáng kể** |
| aug-3 | 0.0378 | Context lớn hơn (5 vs 3) giúp thêm |
| cs10-gs32 | 0.0356 | Context=10 **không giúp** với format cũ |
| **bm25 negatives** | **0.0166** | **Tệ hơn random (0.0329) — phá hủy model** |

<div class="danger">
⚠️ <strong>BM25 negatives là anti-pattern</strong>: BM25 tìm items title-similar → chính là near-positives (cùng category/brand) → label nhầm thành negatives → model học tránh retrieve items liên quan
</div>

---

# 2. Vấn đề: Recall thấp

## Recall Ceiling Analysis

```
Recall@100 = 0.2673
```

Có nghĩa là: **73.3% queries** — item đúng **không có trong top-100** → reranker không thể giúp gì

```
Nếu reranker hoàn hảo và Recall@100 = 1.0:
  HR@10 tiềm năng = 3.1× so với hiện tại (0.0861 → 0.267)
  
Nhưng với Recall@100 chỉ 0.267:
  Phần lớn improvement phải đến từ RETRIEVER, không phải reranker
```

<div class="highlight">
🎯 <strong>Mục tiêu chính</strong>: Cải thiện Recall@K của retriever — đây là bottleneck lớn nhất
</div>

**4 nguyên nhân gốc rễ được xác định:**
- 🔴 **History contamination** (nghiêm trọng nhất)
- 🟠 **Query format không phù hợp**
- 🟡 **Training data sparse**
- 🟡 **BM25 false negative poisoning**

---

# 2a. Nguyên nhân #1 — History Contamination

## Cơ chế

```
Query: "item1, item2, item3 </s>"

→ Bi-encoder encode thành 1 vector
→ Vector này chứa thông tin của item1, item2, item3
→ Cosine similarity với item1, item2, item3 LUÔN cao (exact text overlap)
→ FAISS trả về item1, item2, item3 ở top rank
→ Các items này KHÔNG THỂ là next item (user đã mua rồi)
→ Waste top slots
```

**Số liệu đo trực tiếp** — aug-5 best checkpoint, Beauty test set (22,363 queries):

| Chỉ số | Giá trị |
|---|:---:|
| History item xuất hiện ở **rank 1** | **92.5%** queries |
| ≥1 history item trong **top-5** | **99.5%** queries |
| ≥1 history item trong **top-10** | **99.8%** queries |

---

# 2a. Nguyên nhân #1 — History Contamination

## Phân tích mức độ thiệt hại

<div class="danger">
🚨 Trung bình <strong>2.23 / 5 slots trong top-5 (44.6%) bị chiếm bởi history items</strong>
</div>

**Phân phối số history items trong top-5** (22,363 queries):

| Số history items trong top-5 | Số queries | Tỷ lệ |
|:---:|:---:|:---:|
| 0 — không bị ảnh hưởng | 103 | 0.5% |
| 1 | 4,852 | 21.7% |
| **2 ← phổ biến nhất** | **9,471** | **42.4%** |
| 3 | 6,235 | 27.9% |
| 4 | 1,129 | 5.0% |
| 5 — toàn bộ top-5 là history | 573 | 2.6% |

> Chỉ **103 / 22,363 queries (0.5%)** không bị contaminate trong top-5

---

# 2a. Nguyên nhân #1 — Tại sao SASRec không bị?

## Phân tích fairness — SASRec vs repLLaMA

| | SASRec | repLLaMA |
|---|---|---|
| Biểu diễn item | Integer ID (1 token/item) | Text title (10-20 tokens/item) |
| Biểu diễn user | Toàn bộ history qua attention | 3-5 item titles concat |
| History filtering | Implicit (học qua training) | **Không có** — structural bias |
| FAISS rank-1 là history? | Không (ID khác nhau hoàn toàn) | **92.5%** trường hợp |

**Source code RecBole xác nhận:**
```python
# general_dataloader.py — collate_fn cho sequential models (SASRec):
return transformed_interaction, None, positive_u, positive_i
#                               ↑ None = history_index không được set

# trainer.py:
if history_index is not None:
    scores[history_index] = -np.inf  # → KHÔNG chạy với SASRec
```

<div class="good">
✅ <strong>Kết luận</strong>: Filter history items là FAIR — bù đắp structural bias của text-based retrieval. SASRec tránh seen items implicitly qua architecture ID-based, repLLaMA cần explicit filter.
</div>

---

# 2b. Nguyên nhân #2 — Query Format Không Phù Hợp

## Format hiện tại

```
Query: item1, item2, item3 </s>
```

**5 vấn đề cụ thể:**

| # | Vấn đề | Ví dụ |
|---|---|---|
| 1 | **Comma ambiguity** | Title "CND Shellac, 0.25 oz" → không phân biệt với separator |
| 2 | **No task signal** | Model không biết đây là recommendation, similarity, hay QA |
| 3 | **Chỉ có title** | Bỏ qua category & brand — 100% và 83% items có data này |
| 4 | **Symmetric format** | Query = Document format → Qwen3-Embedding thiết kế cho asymmetric |
| 5 | **`</s>` là noise** | Qwen3 EOS là `<\|im_end\|>`, append bởi `--append_eos_token`. Literal `" </s>"` = 3 noise tokens |

<div class="highlight">
💡 Khi dùng context_size=10, format cũ càng tệ hơn: model thấy 10 titles nhưng không biết chúng là gì, quan hệ thế nào, và phải làm gì với chúng
</div>

---

# 2c. Nguyên nhân #3 — Training Data Sparse

## So sánh với SASRec

| | repLLaMA (standard) | repLLaMA (aug-5) | SASRec |
|---|:---:|:---:|:---:|
| Training samples (Beauty) | 22,363 | **~154,000** | ~154,000+ |
| Samples per user | 1 | ~7 | All positions |
| Context mỗi sample | 3 items | 3 items (sliding) | Full history |

- **1 sample/user** → model thấy mỗi sequence chỉ 1 lần trong 1 góc nhìn
- **Augmentation** (aug) đã giải quyết được phần lớn: aug-5 tăng +4.8% NDCG@10 so với standard
- Sliding window đúng cách — không leak valid/test items (range: 1 đến N-2)

---

# 2d. Nguyên nhân #4 — BM25 False Negative Poisoning

## Tại sao BM25 negatives thất bại hoàn toàn?

```
BM25 logic:
  Query = "Skin Peel, Deodorant, Waxelene"
  Top BM25 results = ["Salicylic Acid Gel Peel", "TCA 15% Peel", "Papaya Enzyme Peel"]
  
  → Đây là EXACT CATEGORY items: cùng loại skin peel
  → Label nhầm thành NEGATIVE
  → Model học: "đừng retrieve Skin Peel items khi query có Skin Peel"
  → Recall về 0 với các items cùng category
```

**Kết quả thực tế:**
- BM25 negatives: NDCG@10 = **0.0166** (tệ hơn **zero-shot 0.0135** không nhiều!)
- Random negatives: NDCG@10 = **0.0329**

<div class="danger">
⚠️ BM25 hard negatives cho IR document retrieval (khác domain) ≠ BM25 hard negatives cho recommendation.
Trong recommendation, items title-similar thường là POSITIVE candidates, không phải negatives.
</div>

---

# 3. Giải pháp Đề xuất

## Tổng quan

```
Vấn đề 1: History contamination    → Phase 2: Filter history post-processing
Vấn đề 2: Query format             → Phase 1: Thiết kế lại format (ưu tiên cao nhất)
Vấn đề 3: Training data sparse     → Đã có augmentation, dùng với format mới
Vấn đề 4: BM25 false negatives     → Đã biết: dùng random negatives hoặc model-mined
```

**Lý do ưu tiên format mới trước:**

1. Ảnh hưởng **đồng thời cả training và evaluation** — không chỉ eval
2. **Zero-shot improvement** có thể đo ngay mà không cần train lại
3. Là nền tảng cho tất cả experiments tiếp theo
4. Code change nhỏ (2-3 hàm) nhưng impact rộng

<div class="good">
💡 Format mới giải quyết <strong>đồng thời</strong> vấn đề #2 (thêm task signal + metadata) và <strong>giảm nhẹ</strong> vấn đề #1 (query format khác với document format hơn → history contamination giảm)
</div>

---

# 4. Metadata Available

## Amazon Beauty & Sports — Filtered Corpus (12,101 items)

| Field | Coverage | Giá trị thực tế |
|---|:---:|---|
| `title` | **100%** | "Pineapple Pumpkin Enzyme Skin Peel" |
| `categories` | **100%** | `["Beauty", "Skin Care", "Face", "Peels"]` → full hierarchy |
| `description` | 90% | Truncated 200 chars — quá dài, skip |
| **`brand`** | **83%** | "Max Factor", "L'Oreal" — cao hơn raw (49%) do 5-core filter |
| `price` | 73% | 19.98 — không có semantic value |
| `salesRank` | 98% | `{"Beauty": 402875}` — không có semantic value |

**Quyết định:**
- ✅ Dùng: `title` + `categories` (full path) + `brand` (khi có)
- ❌ Skip: `description` (dài, noisy), `price`, `salesRank`

> **Lưu ý**: Brand coverage trong raw data chỉ 49%, nhưng sau 5-core filter tăng lên **83%** — vì các items có ít interactions (thường thiếu metadata) đã bị loại bỏ

---

# 4. Metadata Available

## MovieLens 1M

**Files trong dataset:**

| File | Nội dung |
|---|---|
| `movies.dat` | MovieID :: Title (với năm) :: Genres |
| `ratings.dat` | UserID :: MovieID :: Rating :: Timestamp |
| `users.dat` | Gender, Age, Occupation — **user-side**, không dùng cho item |

**Kết luận**: Standard ML-1M **KHÔNG có director/actor** — chỉ có `title` và `genres`.

```
1::Toy Story (1995)::Animation|Children's|Comedy
2::Jumanji (1995)::Adventure|Children's|Fantasy
```

**18 genres**: Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western

**Quyết định:**
- ✅ Dùng: `title` (với năm) + `genres` (comma-separated, dùng label `"Genre:"`)
- ❌ Director/actor: cần augment từ IMDb/TMDB — ngoài scope hiện tại

---

# 5. Format Mới — Nguyên tắc Thiết kế

## 3 nguyên tắc cốt lõi

**① Format Alignment — "described in the same format"**
```
Query: "...predict only one item...described in the SAME FORMAT"
Document: "Title: X. Category: Y. Brand: Z."
→ Model học: query embedding phải gần với document có format giống mô tả item tiếp theo
→ Cosine similarity tự nhiên cao hơn với đúng document
```

**② Task Signal rõ ràng**
```
"predict only one item the customer is most likely to purchase NEXT"
→ Capture tính SEQUENTIAL ("next", không phải "similar")
→ Phân biệt với general similarity retrieval
```

**③ Asymmetric Design — Qwen3-Embedding**
```
Query:    Instruction + structured item list  ← rich context
Document: Pure item description               ← lean, consistent
→ Đúng thiết kế của Qwen3-Embedding (last-token pooling với instruction prefix)
```

---

# 5. Format Mới — Amazon Beauty & Sports

## Query Format (đầy đủ)

```
Below is a customer's purchase history on Amazon, listed in chronological order
(earliest to latest). Each item is represented by the following format:
Title: <item title>. Category: <item category 1> > <item category 2> > ....
Brand: <item brand>. (Brand field is omitted when unavailable.)
Based on this history, predict only one item the customer is most likely to
purchase next, described in the same format.
Purchase history:
Title: CND Shellac Top .5oz and Base .42oz. Category: Beauty > Nail Care > Nail Polish > Base & Top Coats. Brand: CND.
Title: CND Shellac Gel Nail Polish Zillionaire .25 Oz. Category: Beauty > Nail Care > Nail Polish > Gel Polish. Brand: CND.
Title: Creative Nail Shellac Gold Vip Status. Category: Beauty > Nail Care > Nail Polish > Gel Polish. Brand: Creative Nail Design.
```
*+ `<|im_end|>` được append tự động bởi `--append_eos_token`*

**Thay đổi so với format cũ:**
- Mỗi item trên 1 dòng riêng → không còn comma ambiguity
- Thêm `Category:` (full path) và `Brand:` (khi có)
- Instruction prefix → task signal rõ ràng
- Bỏ `" </s>"` (noise token)

---

# 5. Format Mới — Amazon Document

## Document Format

**Khi có brand:**
```
Title: Travalo iPump5 Silver. Category: Beauty > Tools & Accessories > Travel Accessories. Brand: Travalo.
```

**Khi không có brand (bỏ hẳn field, KHÔNG ghi "Brand: unknown"):**
```
Title: White Plastic Jar with Dome Lid 2 Oz. Category: Beauty > Skin Care > Packaging > Jars.
```

**Lý do bỏ `"Passage: "` prefix:**
- `"Passage: "` là artifact từ repLLaMA gốc (Llama-2 asymmetry)
- Qwen3-Embedding dùng last-token pooling `<|im_end|>` — tiền tố không cần thiết
- Document format mới dùng `"Title: "` là anchor, **nhất quán với items trong query**
- "Brand: unknown" là noise — model học associate "unknown" với items thiếu brand

---

# 5. Format Mới — ML-1M

## Query & Document Format

**Query:**
```
Below is a customer's movie watch history, listed in chronological order
(earliest to latest). Each item is represented by the following format:
Title: <movie title>. Genre: <genre 1>, <genre 2>, ....
Based on this history, predict only one movie the customer is most likely to
watch next, described in the same format.
Watch history:
Title: Toy Story (1995). Genre: Animation, Children's, Comedy.
Title: Jumanji (1995). Genre: Adventure, Children's, Fantasy.
Title: Grumpier Old Men (1995). Genre: Comedy, Romance.
```

**Document:**
```
Title: Toy Story 2 (1999). Genre: Animation, Children's, Comedy.
```

**Khác biệt so với Amazon:**
- `"Genre:"` thay vì `"Category:"` — genres là flat list, không hierarchical
- `","` thay vì `">"` làm separator
- `"Watch history:"` thay vì `"Purchase history:"`
- Năm trong title (`1995`) giữ nguyên — có giá trị semantic (era, sequel recognition)

---

# 5. Instruction Text — Phân tích Lựa chọn

## Tại sao "predict" thay vì "retrieve"?

| Từ | Nghĩa | Phù hợp? |
|---|---|:---:|
| `retrieve` | Tìm items tương tự (general IR) | ⚠️ Không capture "NEXT" |
| `find` | Neutral | ⚠️ Không có temporal signal |
| **`predict`** | Dự đoán cái sẽ xảy ra tiếp theo | ✅ **Đúng bản chất bài toán** |

**"described in the same format" — tại sao quan trọng?**

```
Không có: "predict only one item the customer is most likely to purchase next"
  → Model embed query không biết output phải trông như thế nào
  → Gap giữa query embedding và document embedding

Có: "...next, described in the SAME FORMAT"
  → Model hiểu: output = "Title: X. Category: Y. Brand: Z."
  → Document có đúng format đó → embedding space aligned
  → Cosine similarity cao hơn với đúng document
```

<div class="good">
✅ <strong>Format alignment</strong> là kỹ thuật quan trọng trong asymmetric dense retrieval — instruction báo cho model biết format của "answer" cần retrieve
</div>

---

# 5. Token Budget

## So sánh format cũ vs mới

| Format | Tokens |
|---|:---:|
| **Format cũ** — query 3 items | **63 tok** |
| Format mới — query 3 items (title only, không metadata) | 124 tok |
| **Format mới — query 3 items (title + cat + brand)** | **188 tok** |
| Format mới — query 5 items (title + cat + brand) | 230 tok |
| Format cũ — document | 20 tok |
| **Format mới — document (có brand)** | **~42 tok** (P95 = 55 tok) |
| Format mới — ML-1M document | ~20 tok (P95 = 26 tok) |

**Instruction overhead:** Amazon = 94 tok · ML-1M = 75 tok

## `--query-max-len` và `--passage-max-len` mới

| Dataset | cs=3 | cs=5 | cs=10 | passage-max-len |
|---|:---:|:---:|:---:|:---:|
| Beauty / Sports | **256** | **320** | **512** | **128** |
| ML-1M | **192** | **256** | **320** | **64** |

---

# 6. Kế hoạch Implement

## Tổng quan — Chỉ 3 hàm cần sửa

```
dataset/preprocess.py
  └── build_item_text()      ← THÊM category + brand vào item text

dataset/export_tevatron.py
  ├── build_query()          ← THAY format cũ bằng instruction + structured list
  ├── make_passage()         ← BỎ "Passage: " prefix
  └── corpus export loop     ← BỎ f"Passage: {text}", dùng build_item_text() trực tiếp

train.sh / eval.sh
  └── --query-max-len        ← UPDATE giá trị mặc định theo bảng trên
```

> Tất cả nơi khác dùng `build_item_text()` sẽ tự động được update (corpus, query items, negative passages)

---

# 6. Implement — `build_item_text()`

## `dataset/preprocess.py`

**Trước:**
```python
def build_item_text(item_id: str, item_meta: dict) -> str:
    meta = item_meta.get(item_id, {})
    title = meta.get("title") or ""
    if isinstance(title, list):
        title = " ".join(str(t) for t in title)
    title = str(title).strip()
    return title if title else item_id   # ← chỉ title
```

**Sau:**
```python
def build_item_text(item_id: str, item_meta: dict) -> str:
    meta = item_meta.get(item_id, {})
    title = meta.get("title") or ""
    if isinstance(title, list):
        title = " ".join(str(t) for t in title)
    title = str(title).strip() or item_id

    cat   = str(meta.get("categories") or "").strip()   # full path
    brand = str(meta.get("brand") or "").strip()        # omit khi rỗng

    text = f"Title: {title}."
    if cat:
        text += f" Category: {cat}."
    if brand:
        text += f" Brand: {brand}."
    return text
```

> ⚠️ ML-1M: `load_movielens()` lưu genres vào field `"categories"` với space-separated. Cần sửa để lưu vào field riêng `"genres"` và dùng label `"Genre:"` thay `"Category:"` khi build text.

---

# 6. Implement — `build_query()`

## `dataset/export_tevatron.py`

**Trước:**
```python
def build_query(context: list, item_meta: dict) -> str:
    if not context:
        return "Query: </s>"
    texts = [build_item_text(iid, item_meta) for iid in context]
    return "Query: " + ", ".join(texts) + " </s>"   # ← comma, " </s>" noise
```

**Sau:**
```python
AMAZON_INSTRUCTION = (
    "Below is a customer's purchase history on Amazon, listed in chronological "
    "order (earliest to latest). Each item is represented by the following format: "
    "Title: <item title>. Category: <item category 1> > <item category 2> > .... "
    "Brand: <item brand>. (Brand field is omitted when unavailable.) "
    "Based on this history, predict only one item the customer is most likely to "
    "purchase next, described in the same format."
)
ML1M_INSTRUCTION = (
    "Below is a customer's movie watch history, listed in chronological order "
    "(earliest to latest). Each item is represented by the following format: "
    "Title: <movie title>. Genre: <genre 1>, <genre 2>, .... "
    "Based on this history, predict only one movie the customer is most likely "
    "to watch next, described in the same format."
)

def build_query(context: list, item_meta: dict, dataset_name: str = "amazon") -> str:
    inst   = ML1M_INSTRUCTION if dataset_name == "ml-1m" else AMAZON_INSTRUCTION
    header = "Watch history:" if dataset_name == "ml-1m" else "Purchase history:"
    items  = "\n".join(build_item_text(iid, item_meta) for iid in context)
    return f"{inst}\n{header}\n{items}"   # ← không có " </s>"
```

---

# 6. Implement — `make_passage()` & corpus loop

## `dataset/export_tevatron.py`

**`make_passage()` — trước:**
```python
def make_passage(item_id: str, item_meta: dict) -> dict:
    text = build_item_text(item_id, item_meta)
    return {"docid": item_id, "title": "", "text": f"Passage: {text}"}
    #                                               ↑ BỎ "Passage: " prefix
```

**`make_passage()` — sau:**
```python
def make_passage(item_id: str, item_meta: dict) -> dict:
    text = build_item_text(item_id, item_meta)
    return {"docid": item_id, "title": "", "text": text}
    # text = "Title: X. Category: Y. Brand: Z." — không cần prefix thêm
```

**Corpus export loop — sau:**
```python
for item_id in tqdm(all_items_list, desc="  corpus"):
    f.write(json.dumps(
        {"docid": item_id, "text": build_item_text(item_id, item_meta)},
        ensure_ascii=False,
    ) + "\n")
    # BỎ f"Passage: {text}" — thay bằng build_item_text() trực tiếp
```

---

# 7. Thứ tự Thực nghiệm

## Roadmap sau khi implement

```
Bước 1 — Export data mới (Beauty, Sports, ML-1M)
  cd dataset
  python export_tevatron.py beauty --tag v2
  python export_tevatron.py beauty --augment --tag v2-aug
  python export_tevatron.py sports --tag v2
  python export_tevatron.py ml-1m --tag v2

Bước 2 — Zero-shot eval (đo baseline ngay, không cần train)
  ./eval.sh beauty base              # zero-shot hiện tại: HR@10 = 0.0315
  ./eval.sh beauty base --tag v2     # kỳ vọng tăng đáng kể
  → Nếu zero-shot tăng: format alignment đang hoạt động

Bước 3 — Fine-tune với format mới
  ./train.sh beauty --data-variant v2 --tag v2 --query-max-len 256
  ./eval.sh beauty --tag v2

Bước 4 — Fine-tune với augmentation + format mới
  ./train.sh beauty --data-variant v2-aug --tag v2-aug --query-max-len 256
  ./eval.sh beauty --tag v2-aug

Bước 5 — So sánh tổng hợp
  python show_results.py
```

---

# 7. Thực nghiệm — Ablation Matrix đề xuất

## So sánh có hệ thống

| Experiment | Data | Format | Kỳ vọng |
|---|---|---|---|
| `v2` standard | 22K samples | **Mới** | Baseline format mới |
| `v2-aug` | 154K samples | **Mới** | Best candidate |
| `v2-cs5` | 22K, 5 items | **Mới** | Context dài hơn |
| `v2-aug-cs5` | 154K, 5 items | **Mới** | Combination mạnh nhất |
| `aug-5` *(đang là best)* | 154K samples | Cũ | Baseline so sánh |

**Metric chính để theo dõi**: HR@10, NDCG@10, **Recall@100**

> Recall@100 là chỉ số quan trọng nhất — phản ánh trực tiếp khả năng retrieval trước khi rerank

---

# 8. Cải thiện Khác — History Filtering

## Post-processing (không cần train lại)

**Cơ chế đơn giản:**
```
1. FAISS search top-(K + len(history)) thay vì top-K
2. Sau search, loại bỏ items có text match với history của user
3. Giữ lại đúng top-K candidates sạch

Matching logic (substring containment, case-insensitive):
  is_history_item = (cand == hist) OR (len(hist)>15 AND (cand in hist OR hist in cand))
```

**Kỳ vọng impact:**
- 92.5% queries có history item ở rank 1 → sau filter, rank 1 = item thật sự mới
- Nhiều queries có positive item ở rank 11-12 (miss HR@10) → sau filter lên rank 9-10 → HIT

**Implement ở 2 nơi:**
- `eval.sh` — evaluation pipeline của retriever
- `prepare_rerank_data.py` — input cho reranker (đã có plan A1 trong `reranker_redesign_plan.md`)

<div class="highlight">
⚡ Đây là "free lunch" — không cần train lại, có thể chạy với model hiện tại ngay
</div>

---

# 8. Cải thiện Khác — Tăng Depth K và Hard Negatives

## Tăng Retrieval Depth K = 200

```
Recall@100 = 0.2673
Recall@200 ≈ 0.35 (ước tính)

→ Reranker có thêm ~8% queries thêm với item đúng để rerank
→ Không cần train lại, chỉ sửa --depth trong eval.sh
→ Reranker pipeline: FAISS(200) → filter history → rerank top-200 → report top-10
```

## Hard Negative Mining từ Model (curriculum learning)

```
Vấn đề: Random negatives tốt hơn BM25 vì không bị false negative
Giải pháp tốt hơn random: Mine từ chính model đang train

Epoch 1: Train với random negatives
  ↓
Mine top-K từ model sau epoch 1
  ↓
Epoch 2-3: Train với model-mined hard negatives
  → Negatives là items SEMANTICALLY SIMILAR trong embedding space
  → Khó hơn random, nhưng đúng là negatives (không như BM25)
```

> Cần viết script riêng để mine và tạo lại training data giữa các epochs

---

# 8. Tổng hợp — Tất cả Cải thiện

## Pipeline đầy đủ theo phase

```
Phase 1 — Format redesign (implement ngay)
  ✦ build_item_text(): title + category + brand
  ✦ build_query(): instruction format
  ✦ make_passage(): bỏ "Passage: " prefix
  ✦ Token budget update

Phase 2 — History filtering (không cần train lại)
  ✦ Post-filter history items từ FAISS results
  ✦ Tăng K = 200
  ✦ Reranker với filter (A1 trong reranker_redesign_plan.md)

Phase 3 — Scaling experiments
  ✦ Format mới + augmentation + cs=5
  ✦ Reranker: C1 (aug train data) + D1 (score combination)
  ✦ Hard negative mining từ model

Phase 4 — Cross-dataset
  ✦ Sports + ML-1M với format mới
  ✦ Ablation: category vs no category, brand vs no brand
```

---

# Tóm tắt — Key Takeaways

## Vấn đề được xác định với số liệu cụ thể

| Vấn đề | Số liệu | Giải pháp |
|---|:---:|---|
| History contamination | **92.5% rank-1** bị waste | Filter post-processing |
| Top-5 bị chiếm | **44.6% slots** (2.23/5) | Filter + format mới |
| Recall ceiling | **26.7%** @100 | Format + filter + K=200 |
| Query format | 5 vấn đề cùng lúc | Instruction + metadata |
| BM25 negatives | NDCG **0.0166** (random=0.0329) | Dùng random / model-mined |

## Thay đổi code nhỏ, impact lớn

```
3 hàm Python sửa:  build_item_text() + build_query() + make_passage()
2 shell script:    train.sh + eval.sh (--query-max-len)
→ Ảnh hưởng: tất cả experiments, tất cả datasets, cả training lẫn evaluation
```

<div class="good">
🎯 <strong>Bước tiếp theo ngay lập tức</strong>: Implement format mới → chạy zero-shot eval → so sánh với zero-shot cũ (HR@10 = 0.0315) để đo baseline improvement trước khi train lại
</div>

---

# Phụ lục — Checklist Implement

## Phase 1 — Format Redesign

- [ ] Sửa `build_item_text()` trong `preprocess.py` — thêm category + brand
- [ ] Sửa `load_movielens()` trong `preprocess.py` — lưu genres vào field riêng
- [ ] Thêm `AMAZON_INSTRUCTION` + `ML1M_INSTRUCTION` vào `export_tevatron.py`
- [ ] Sửa `build_query()` — instruction format, bỏ `" </s>"`
- [ ] Sửa `make_passage()` — bỏ `"Passage: "` prefix
- [ ] Sửa corpus export loop — dùng `build_item_text()` trực tiếp
- [ ] Update `--query-max-len` defaults trong `train.sh` và `eval.sh`
- [ ] Export data: `beauty --tag v2`, `beauty --augment --tag v2-aug`
- [ ] Chạy zero-shot eval — đo baseline trước khi train
- [ ] Train + eval: beauty v2, beauty v2-aug

## Phase 2 — History Filtering

- [ ] Implement filter-history trong evaluation pipeline
- [ ] Implement A1 theo `reranker_redesign_plan.md`
- [ ] Đo Recall@100 và HR@10 với + không có filter

## Phase 3 — Scaling

- [ ] cs=5 với format mới, K=200, reranker pipeline
