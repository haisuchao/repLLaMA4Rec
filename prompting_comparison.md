# So sánh Prompt Representation: repLLaMA gốc vs. Metadata-Enhanced Prompting

## 1. Tại sao cần cải tiến?

repLLaMA gốc biểu diễn lịch sử tương tác của user **chỉ bằng title** của các item. Cách này có một số hạn chế:

| Hạn chế | Giải thích |
|---|---|
| **Thiếu ngữ cảnh ngữ nghĩa** | Title đơn lẻ không đủ để LLM hiểu *loại sản phẩm* hay *thương hiệu* user đang quan tâm. Ví dụ: `"Phonecase Gold"` — LLM không biết đây là phụ kiện điện thoại hay trang sức. |
| **Không khai thác được metadata sẵn có** | Bộ dữ liệu Amazon cung cấp sẵn category (nhiều mức), brand, description — nhưng repLLaMA gốc bỏ qua hoàn toàn. |
| **Prompt quá đơn giản, không dẫn dắt LLM** | Format `"Query: title1, title2, title3"` không cho LLM biết *nhiệm vụ cần thực hiện* — chỉ là một danh sách tên sản phẩm, thiếu instruction rõ ràng. |
| **Candidate item cũng thiếu thông tin** | Phía document chỉ có title — LLM không thể so khớp category hay brand giữa lịch sử user và candidate. |

**Tóm lại:** repLLaMA gốc chỉ dùng *tên sản phẩm* để matching, trong khi quyết định mua hàng thực tế phụ thuộc vào nhiều yếu tố hơn (ngành hàng, thương hiệu, mức giá, ...). Bổ sung metadata giúp LLM hiểu sâu hơn về *ý định mua sắm* (purchase intent) của user.

---

## 2. So sánh chi tiết

### 2.1 Query — Biểu diễn lịch sử tương tác của user

<table>
<tr>
<th width="50%">repLLaMA gốc</th>
<th width="50%">Metadata-Enhanced (cải tiến)</th>
</tr>
<tr>
<td>

**Format:**

```
Query: <i1's Title>, <i2's Title>, <i3's Title> <EOS>
```

**Ví dụ:**

```
Query: Iphone13 Promax, Phonecase Gold,
Charger Ugreen <EOS>
```

</td>
<td>

**Format:**

```
Below is a customer's purchase history on Amazon,
listed in chronological order (earliest to latest).
Each item is represented by the following format:
Title: <item title>. Category: <item category 1>
> <item category 2> > .... Brand: <item brand>.
(Brand field is omitted when unavailable.)
Based on this history, predict only one item the
customer is most likely to purchase next, described
in the same format.

Purchase history:
Title: <i1's Title>. Category: <i1's categories>.
Brand: <i1's brand>.
Title: <i2's Title>. Category: <i2's categories>.
Brand: <i2's brand>.
Title: <i3's Title>. Category: <i3's categories>.
Brand: <i3's brand>.
<EOS>
```

**Ví dụ:**

```
Below is a customer's purchase history on Amazon,
listed in chronological order (earliest to latest).
Each item is represented by the following format:
Title: <item title>. Category: <item category 1>
> <item category 2> > .... Brand: <item brand>.
(Brand field is omitted when unavailable.)
Based on this history, predict only one item the
customer is most likely to purchase next, described
in the same format.

Purchase history:
Title: Iphone13. Category: Electricity > Mobile
> iphone. Brand: Apple.
Title: Phonecase Gold. Category: Accessories
> Phonecase. Brand: None.
Title: Charger Ugreen. Category: Accessories
> charger. Brand: Ugreen.
<EOS>
```

</td>
</tr>
</table>

### 2.2 Document — Biểu diễn candidate item

<table>
<tr>
<th width="50%">repLLaMA gốc</th>
<th width="50%">Metadata-Enhanced (cải tiến)</th>
</tr>
<tr>
<td>

**Format:**

```
Passage: <candidate item's title> <EOS>
```

**Ví dụ:**

```
Passage: Airpod 3 White <EOS>
```

</td>
<td>

**Format:**

```
Title: <item's Title>. Category: <item's categories>.
Brand: <item's brand>. <EOS>
```

**Ví dụ:**

```
Title: Airpod 3 White. Category: Accessories
> earbud. Brand: Apple. <EOS>
```

</td>
</tr>
</table>

### 2.3 Cơ chế tính score (giữ nguyên)

Cả hai phiên bản đều sử dụng cùng cơ chế scoring — **không thay đổi kiến trúc model**:

```
1. v_q = LLM(query)  → lấy vector tại vị trí <EOS>
2. v_d = LLM(document) → lấy vector tại vị trí <EOS>
3. score = v_q · v_d   (dot product)
```

Cải tiến chỉ nằm ở **nội dung text đầu vào**, không thay đổi model hay pipeline.

---

## 3. Metadata được bổ sung và lý do

### 3.1 Các metadata có sẵn trong bộ Amazon

| Metadata | Mô tả | Sử dụng? |
|---|---|---|
| **Title** | Tên sản phẩm | Có (giữ nguyên từ bản gốc) |
| **Category** | Phân cấp ngành hàng (nhiều mức) | **Có — MỚI** |
| **Brand** | Thương hiệu sản phẩm | **Có — MỚI** |
| Description | Mô tả chi tiết sản phẩm | Chưa (quá dài, tốn token) |

### 3.2 Tại sao chọn Category và Brand?

| Metadata | Lý do bổ sung |
|---|---|
| **Category** (nhiều mức) | **Phản ánh ý định mua sắm theo ngành hàng.** Ví dụ: user mua `Iphone13` (Electricity > Mobile > iphone) rồi `Phonecase Gold` (Accessories > Phonecase) — category cho thấy user đang trong hệ sinh thái *điện thoại + phụ kiện*. LLM có thể suy luận item tiếp theo cũng thuộc hệ sinh thái này (ví dụ: tai nghe, miếng dán màn hình). Category nhiều mức (hierarchical) giúp biểu diễn từ tổng quát đến chi tiết: `Accessories > earbud` vừa cho biết đây là phụ kiện, vừa cho biết cụ thể là tai nghe. |
| **Brand** | **Phản ánh sự trung thành thương hiệu (brand loyalty).** User mua `Iphone13` (Apple) → khả năng cao sẽ mua tiếp sản phẩm Apple (Airpod, Apple Watch) hơn là sản phẩm Samsung. Brand cũng giúp phân biệt sản phẩm cùng loại: `Charger Ugreen` vs `Charger Anker` — hai candidate cùng category nhưng khác brand, user có thể ưu tiên brand quen thuộc. |

### 3.3 Tại sao chưa dùng Description?

- **Quá dài:** Description có thể dài hàng trăm từ, nhân với 3 items trong query sẽ vượt giới hạn `--query-max-len` (128-160 tokens).
- **Nhiễu:** Description chứa nhiều thông tin marketing không liên quan đến ý định mua sắm.
- **Diminishing returns:** Category + Brand đã cung cấp tín hiệu chính yếu; thêm description tăng chi phí tính toán nhưng lợi ích chưa rõ ràng.

---

## 4. Tổng hợp khác biệt

| Khía cạnh | repLLaMA gốc | Metadata-Enhanced |
|---|---|---|
| **Thông tin item trong query** | Chỉ title | Title + Category + Brand |
| **Thông tin candidate item** | Chỉ title | Title + Category + Brand |
| **Task instruction** | Không có | Có — hướng dẫn LLM nhiệm vụ cụ thể (dự đoán item tiếp theo) |
| **Cấu trúc prompt** | Phẳng, nối bằng dấu phẩy | Có cấu trúc rõ ràng (mỗi item một dòng, các trường phân tách) |
| **Prefix query** | `"Query: "` | Đoạn instruction dài, kết thúc bằng `"Purchase history:"` |
| **Prefix document** | `"Passage: "` | `"Title: ... Category: ... Brand: ..."` |
| **Kiến trúc model** | Không đổi | Không đổi |
| **Cơ chế scoring** | `v_q · v_d` (dot product) | `v_q · v_d` (dot product) — giữ nguyên |
| **Chi phí token** | Thấp (~50-70 tokens/query) | Cao hơn (~120-200 tokens/query) |

---

## 5. Minh họa trực quan

```
═══════════════════════════════════════════════════════════════
                     repLLaMA GỐC
═══════════════════════════════════════════════════════════════

  User history                    Candidate
  ┌─────────────────────┐         ┌──────────────────┐
  │ Query: Iphone13     │         │ Passage: Airpod  │
  │ Promax, Phonecase   │         │ 3 White <EOS>    │
  │ Gold, Charger       │         │                  │
  │ Ugreen <EOS>        │         │                  │
  └─────────┬───────────┘         └────────┬─────────┘
            │                              │
            ▼                              ▼
        ┌───────┐                      ┌───────┐
        │  LLM  │                      │  LLM  │
        └───┬───┘                      └───┬───┘
            │                              │
            ▼                              ▼
          v_q ─────── dot product ─────── v_d  →  score
           (chỉ biết TÊN sản phẩm)

═══════════════════════════════════════════════════════════════
                  METADATA-ENHANCED
═══════════════════════════════════════════════════════════════

  User history                    Candidate
  ┌─────────────────────┐         ┌──────────────────┐
  │ [Instruction...]    │         │ Title: Airpod 3  │
  │ Purchase history:   │         │ White.           │
  │                     │         │ Category:        │
  │ Title: Iphone13.    │         │ Accessories >    │
  │ Category: Elec >    │         │ earbud.          │
  │ Mobile > iphone.    │         │ Brand: Apple.    │
  │ Brand: Apple.       │         │ <EOS>            │
  │                     │         │                  │
  │ Title: Phonecase    │         │                  │
  │ Gold.               │         │                  │
  │ Category: Acc >     │         │                  │
  │ Phonecase.          │         │                  │
  │ Brand: None.        │         │                  │
  │                     │         │                  │
  │ Title: Charger      │         │                  │
  │ Ugreen.             │         │                  │
  │ Category: Acc >     │         │                  │
  │ charger.            │         │                  │
  │ Brand: Ugreen.      │         │                  │
  │ <EOS>               │         │                  │
  └─────────┬───────────┘         └────────┬─────────┘
            │                              │
            ▼                              ▼
        ┌───────┐                      ┌───────┐
        │  LLM  │                      │  LLM  │
        └───┬───┘                      └───┬───┘
            │                              │
            ▼                              ▼
          v_q ─────── dot product ─────── v_d  →  score
     (biết TÊN + NGÀNH HÀNG + THƯƠNG HIỆU)
```

**Ví dụ suy luận mà LLM có thể thực hiện với metadata:**

- User mua toàn hệ sinh thái *điện thoại + phụ kiện* → candidate `Airpod 3 White` (Accessories > earbud, **Apple**) phù hợp vì cùng brand Apple và cùng nhóm phụ kiện.
- Nếu candidate là `Samsung Galaxy Buds` (Accessories > earbud, Samsung) — cùng category nhưng khác brand → score thấp hơn do user thể hiện brand loyalty với Apple.
- Với repLLaMA gốc, LLM chỉ thấy tên `"Airpod 3 White"` và `"Samsung Galaxy Buds"` — không có tín hiệu brand/category rõ ràng để phân biệt.
