# So sánh Query Representation: repLLaMA gốc vs. Metadata-Enhanced Query representation

## 1. Tại sao cần cải tiến?

repLLaMA gốc biểu diễn lịch sử tương tác của user **chỉ bằng title** của các item. Cách này có một số hạn chế:

| Hạn chế | Giải thích |
|---|---|
| **Thiếu ngữ cảnh ngữ nghĩa** | Title đơn lẻ không đủ để LLM hiểu *loại sản phẩm* hay *thương hiệu* user đang quan tâm. Ví dụ: `"Phonecase Gold"` — LLM không biết đây là phụ kiện điện thoại hay trang sức. |
| **Không khai thác được metadata sẵn có** | Bộ dữ liệu Amazon cung cấp sẵn category (nhiều mức), brand, description — nhưng repLLaMA gốc bỏ qua hoàn toàn. |
| **Prompt quá đơn giản, không dẫn dắt LLM** | Format `"Query: title1, title2, title3"` không cho LLM biết *nhiệm vụ cần thực hiện* — chỉ là một danh sách tên sản phẩm, thiếu instruction rõ ràng. |
| **Candidate item cũng thiếu thông tin** | Phía document chỉ có title — LLM không thể so khớp category hay brand giữa lịch sử user và candidate. |

---

## 2. So sánh chi tiết

### 2.1 Query — Biểu diễn lịch sử tương tác của user
Giả sử một user <b>u</b> có lịch sử tương tác với 3 item lần lượt theo thứ tự <i>i1 -> i2 -> i3</i>, mỗi một item <i>i</i> sẽ có các thông tin bao gồm <b><i>title, categories, brand</i></b>. Chúng ta sẽ biểu diễn query của user <b>u</b> theo hai cách như sau:
<table>
<tr valign="top">
<th width="50%" >repLLaMA gốc</th>
<th width="50%" >Metadata-Enhanced (cải tiến)</th>
</tr>
<tr valign="top">
<td>

**Format:**

```
Query: <i1's Title>, <i2's Title>, <i3's Title> <EOS>
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
</td>
</tr>
<tr valign="top">
<td>

**Ví dụ:**

```
Query: Iphone13 Promax, Phonecase Gold,
Charger Ugreen <EOS>
```
</td>
<td>

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
Tương tư như vậy, với mỗi candidate item <b>i</b> sẽ có các thông tin bao gồm <b><i>title, categories, brand</i></b>, chúng ta có 2 cách biểu diễn như sau:
<table>
<tr valign="top">
<th width="50%" >repLLaMA gốc</th>
<th width="50%" >Metadata-Enhanced (cải tiến)</th>
</tr>
<tr valign="top">
<td >

**Format:**

```
Passage: <candidate item's title> <EOS>
```

</td>
<td >

**Format:**

```
Title: <item's Title>. Category: <item's categories>.
Brand: <item's brand>. <EOS>
```

</td>
</tr>
<tr valign="top">
<td>

**Ví dụ:**

```
Passage: Airpod 3 White <EOS>
```
</td>
<td>

**Ví dụ:**

```
Title: Airpod 3 White. Category: Accessories
> earbud. Brand: Apple. <EOS>
```
</td>
</tr>
</table>

### 2.3 Cơ chế tính score (giữ nguyên)

Cả hai cách đều sử dụng cùng cơ chế scoring — **không thay đổi kiến trúc model**:

```
1. v_q = LLM(query)  → lấy vector tại vị trí <EOS> của query
2. v_d = LLM(document) → lấy vector tại vị trí <EOS> của document
3. score = v_q · v_d   (dot product)
```

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

- **Quá dài:** Description có thể dài hàng trăm từ, nhân với 3 items trong query sẽ tăng số token trong query dẫn tới tăng thời gian huấn luyện.
- **Nhiễu:** Description chứa nhiều thông tin marketing không liên quan đến ý định mua sắm.

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
| **Cơ chế scoring** | Không đổi | Không đổi |
| **Chi phí token** | ~50-70 tokens/query | ~120-200 tokens/query (tốn thời gian training hơn) |