# Paper Outline — FADE: Rethinking Sequential Recommendation as Filter-Augmented Dense Retrieval

> Dựng từ khung bản thảo LNCS (`FADE_tmp.pdf`) của tác giả + số liệu/quyết định đã chốt trong
> `paper_report.md` và các buổi thảo luận trước. Đánh dấu **[SỬA]** ở những chỗ khác với bản thảo gốc, kèm lý
> do. Nguồn số liệu: `paper_report.md` (mục tham chiếu ghi trong ngoặc).

---

## Title & Authors

**[SỬA]** Title: ~~"Retrieval-based Sequential Recommendation using fine-tuning Large Language Model"~~ →
**"FADE: Rethinking Sequential Recommendation as Filter-Augmented Dense Retrieval"**
(đã chốt ở buổi trước — xem lý do đặt tên trong phần trao đổi title/model name)

Authors: Nguyen Do Hai¹'², Tu Minh Phuong² — giữ nguyên.

Running title cần set riêng khi vào LaTeX (`\titlerunning{FADE: Filter-Augmented Dense Retrieval for SR}`)
để tránh lỗi "Title Suppressed Due to Excessive Length" đang thấy trong PDF nháp.

Keywords: Sequential Recommendation · Dense Retrieval · Fine-tuning LLM · Information Retrieval
**[SỬA]** thêm "Dense Retrieval" vào keyword vì đây là framing chính của paper.

---

## Abstract

Giữ cấu trúc bullet 5 ý như bản nháp, sửa 2 chỗ:

1. Motivation: LLM trong SR, xu hướng IR-based SR (giữ nguyên).
2. **[SỬA]** Vấn đề #1: không còn nói "cần 2 encoder khác nhau" như một *fact đã chứng minh* — diễn đạt lại
   thành câu hỏi/động lực: kiến trúc IR-based SR hiện tại (kể cả RepLLaMA gốc) dùng **chung 1 encoder**
   (tied) cho query/document dù bản chất 2 loại text khác nhau về độ dài/nội dung — đây là điểm paper này
   **áp dụng và đánh giá có hệ thống** (không claim đã tách encoder, vì thực tế FADE vẫn dùng tied bi-encoder
   — xem Method).
3. Vấn đề #2 (data sparsity) → giải pháp augmentation kiểu SBR (giữ nguyên, khớp C3).
4. Vấn đề #3 (history contamination) → giải pháp post-hoc filter (giữ nguyên, khớp C2).
5. **[SỬA]** Kết quả: ~~"tốt hơn các mô hình SR sử dụng LLM khác"~~ → **"vượt trội so với các baseline
   ID-based SR mạnh (SASRec, GRU4Rec, SRGNN) trên Amazon Beauty, Amazon Sports và ML-1M"** — bản nháp ghi so
   sánh với "LLM-based SR khác" nhưng thực tế paper không có baseline LLM-based nào khác được reproduce (chỉ
   trích literature TIGER/S³-Rec chưa xác minh — §6 `paper_report.md`), so với ID-based mới là claim có bằng
   chứng đầy đủ 3/3 dataset.

---

## 1 Introduction

**[ĐÃ VIẾT ĐẦY ĐỦ trong `paper_draft.md` §1]** — outline dưới đây giữ lại làm tài liệu tham khảo/lý do quyết
định, nhưng bản văn chính thức đã ở `paper_draft.md`. Có thêm **Figure 1** (`figures/fade_example.pdf`) minh
họa ví dụ skincare, dùng chung cho cả đoạn 2 và thách thức #3.

Giữ mạch 3-đoạn của bản nháp, chỉnh nội dung đoạn 2 và 3 theo đúng 3 contribution đã chốt:

- Đoạn 1: giới thiệu SR, hướng ứng dụng LLM cho SR (giữ nguyên ý).
- Đoạn 2: giới thiệu paradigm IR-based SR (query=history, document=candidate item) + **ví dụ minh họa đầy đủ**
  (thay placeholder "Lấy một ví dụ nếu cần" của bản nháp):

  > Giả sử một user trên Amazon Beauty vừa mua tuần tự 3 sản phẩm gần nhất: *"Neutrogena Ultra Sheer
  > Dry-Touch Sunscreen SPF 45"* → *"CeraVe Foaming Facial Cleanser"* → *"The Ordinary Niacinamide 10% + Zinc
  > 1% Serum"*. Trong paradigm IR-based SR, ba tiêu đề này được nối thành một đoạn văn bản đóng vai trò
  > **query** Q. Toàn bộ catalog (hàng nghìn sản phẩm) đóng vai trò tập **document** D, mỗi item là một
  > passage riêng (tiêu đề sản phẩm). Nhiệm vụ retrieval là tìm d ∈ D có độ tương đồng cao nhất với Q trong
  > không gian embedding — về bản chất là dự đoán sản phẩm user nhiều khả năng mua tiếp theo. Một retriever
  > được fine-tune tốt kỳ vọng xếp hạng cao *"CeraVe Moisturizing Cream"* — bước hợp lý tiếp theo trong
  > routine skincare (chống nắng → làm sạch → serum → dưỡng ẩm) — nhờ nắm bắt liên hệ **ngữ nghĩa** về routine
  > chăm sóc da, dù item này chưa từng đồng xuất hiện với 3 item trên trong bất kỳ giao dịch nào trước đó của
  > user nào khác. Đây chính là điểm mạnh của cách tiếp cận text-based so với mô hình ID-based thuần túy dựa
  > vào thống kê đồng xuất hiện (co-occurrence).

  Ví dụ này được **tái sử dụng** ở đoạn 3 (thách thức #3 — contamination) để giữ mạch xuyên suốt, thay vì mỗi
  đoạn dùng một ví dụ rời rạc.

- Đoạn 3 (thách thức): 3 thách thức, sắp theo mạch logic dẫn thẳng vào 3 thành phần của FADE:
  1. **[MỚI — thay cho tied-vs-untied đã bỏ]** *Động lực chọn kiến trúc bi-encoder*: catalog thực tế có thể
     lên tới hàng chục nghìn–hàng triệu item; muốn dùng LLM để đánh giá độ liên quan, cách chính xác nhất là
     joint-encode (cross-encoder) từng cặp (query, candidate) — nhưng điều này **không khả thi** cho bước
     retrieval trên toàn catalog (phải chạy forward pass cho mọi cặp tại mỗi lần truy vấn). Đây là lý do
     kiến trúc **bi-encoder** (encode Q và D độc lập, precompute embedding toàn bộ catalog, so khớp bằng
     ANN/FAISS) là lựa chọn bắt buộc để dense retrieval khả thi ở quy mô catalog thực tế — không phải chỉ vì
     "DPR/RepLLaMA đã làm vậy" mà vì đây là ràng buộc hiệu năng nội tại của bài toán. → dẫn vào Method §3.2.
  2. **Data sparsity khi fine-tune**: mỗi user trong SR chỉ cho ra 1 cặp (query, positive) theo leave-one-out
     mặc định — quá ít để fine-tune hiệu quả một LLM. → dẫn vào §3.3 (augmentation).
  3. **History contamination** (dùng lại ví dụ ở đoạn 2, bằng chứng mạnh nhất: 92.5% query bị history item
     chiếm rank 1, `improvement_plan.md` §2.1): vì Q chứa nguyên văn *"CeraVe Foaming Facial Cleanser"*, một
     bi-encoder train bằng contrastive loss thông thường có xu hướng xếp chính item này (hoặc item tên gần
     giống) lên rank 1 — do loss huấn luyện **chưa từng dạy mô hình phân biệt "lặp lại" với "liên quan"** (không
     có negative nào trùng verbatim với query trong training set). → dẫn vào §3.4 (filter).
- Đoạn 4 (đề xuất): **[SỬA]** bỏ câu "fine-tune 2 encoder song song" — thay bằng: "Chúng tôi đề xuất **FADE**
  — áp dụng kiến trúc tied bi-encoder LLM (Qwen3-Embedding + LoRA) cho SR, kết hợp (1) data augmentation
  kiểu sliding-window để bù đắp mật độ supervision, và (2) post-hoc history filter để loại bỏ contamination —
  đánh giá có hệ thống trên 3 dataset thực tế và 2 quy mô model (0.6B/4B)."
- Đoạn 5: kết quả tóm tắt. **[SỬA — cập nhật theo bảng Overall Results mới ở §4.5]** SASRec dùng trong bảng
  headline giờ là số liệu **literature** (Overall.pdf) chứ không phải bản tự-reproduce, nên margin đổi thành
  **+93.7% NDCG@10 (Beauty)**, **+93.2% NDCG@10 (Sports)** so với SASRec (literature). Số liệu tự-reproduce
  cũ (+49.5%/+68.6%) vẫn **giữ lại** làm robustness note ở Discussion §4.9 ("ngay cả so với SASRec tự train
  lại dưới cùng protocol/hardware, FADE vẫn vượt +49.5–68.6% NDCG@10") — cho thấy kết quả không phụ thuộc vào
  việc chọn nguồn baseline. Cộng thêm finding phụ đáng chú ý về model scale trên ML-1M (§4.8).

**[SỬA]** 4 bullet contribution cuối Introduction — thay bản nháp bằng:

- **C1**: Áp dụng có hệ thống kiến trúc tied bi-encoder LLM (RepLLaMA/DPR-style) cho SR, benchmark trên 3
  dataset (sequence ngắn→dài) và 2 scale model, phát hiện trade-off scale-vs-kiến trúc trên sequence dài.
- **C2**: Chẩn đoán và lý giải cơ chế "history contamination" (do training objective không tạo negative
  trùng verbatim với query) + fix bằng post-hoc filter không cần train lại.
- **C3**: Data augmentation kiểu sliding-window, lý giải như cách khôi phục mật độ supervision mà causal
  architecture (SASRec) có sẵn miễn phí ở mọi vị trí sequence.
- **C4**: Thực nghiệm toàn diện trên 3 dataset real-world, so sánh với 3 baseline ID-based mạnh + zero-shot.

(Bỏ hẳn bullet "dual encoder 2 model riêng biệt" theo quyết định giữ tied.)

---

## 2 Related Work

**[ĐÃ VIẾT ĐẦY ĐỦ trong `paper_draft.md` §2]** — mở rộng hơn outline dưới đây: nhóm "LLM-based SR" giờ chi
tiết hóa cả 6 baseline mới có citation (TIGER, ActionPiece, EAGER-LLM, LlamaRec, P5, E4SRec) + 1 đoạn riêng
so sánh trực tiếp với GLoSS (related work gần nhất). Thêm citation Caser (Tang & Wang, WSDM 2018) cho đoạn
data augmentation — outline dưới đây để trống citation, đã lấp bằng bản draft.

Chưa có nội dung trong bản nháp — đề xuất 4 nhóm con (không cần đặt subsection riêng nếu paper ngắn, có thể
viết liền mạch 4 đoạn):

1. **Sequential Recommendation (ID-based)**: GRU4Rec (RNN), SRGNN (GNN), SASRec (self-attention/causal) —
   nền tảng baseline của paper.
2. **LLM/text-based dense retrieval cho recommendation**: DPR (Karpukhin et al., 2020) — bi-encoder gốc,
   untied encoder. RepLLaMA (Ma et al., SIGIR 2024, arXiv:2310.08319) — bi-encoder LLM đầu tiên, dùng **tied**
   encoder — một backbone LLaMA duy nhất xử lý cả query lẫn document (đã verify trực tiếp qua arXiv), cùng
   [EOS]-pooling thay [CLS]. FADE kế thừa đúng lựa chọn thiết kế này (tied bi-encoder) và áp dụng cho bài
   toán SR — xem động lực kiến trúc ở §1 và Method §3.2.
3. **Generative retrieval cho SR**: TIGER (Rajput et al., NeurIPS 2023) — hướng khác (semantic ID generation)
   so với dense retrieval, dùng để đối chiếu định tính (chưa có số liệu định lượng verified — §6
   `paper_report.md`, cần trích trực tiếp từ paper gốc trước khi đưa vào).
4. **Data augmentation cho SR/SBR**: sliding-window sub-session augmentation (đã dùng phổ biến trong
   Session-based Rec) — paper này áp dụng ý tưởng tương tự cho bối cảnh fine-tune bi-encoder, khác với SBR
   truyền thống ở chỗ mục tiêu là bù supervision cho *retrieval* loss chứ không phải classification loss.
5. **Lexical-overlap bias trong dense retriever** (nền tảng lý thuyết cho C2): Sciavolino et al. (EMNLP 2021,
   "Simple Entity-Centric Questions Challenge Dense Retrievers") — dense retriever dễ bị chi phối bởi overlap
   từ vựng/thực thể khi thiếu hard negative tương ứng trong training.

---

## 3 Method

### 3.1 Problem Statement
Định nghĩa SR chuẩn (user sequence, dự đoán item tiếp theo) + công thức hóa lại thành retrieval: query
function Q(·), corpus D = toàn bộ item, mục tiêu argmax similarity. (Nội dung kỹ thuật, viết mới — có thể lấy
mô tả từ `README.md` mục "Ý tưởng".)

### 3.2 Bi-Encoder Retriever
**[SỬA — khớp thực tế]** Mô tả đúng kiến trúc: 1 backbone Qwen3-Embedding (0.6B/4B) + LoRA (rank 16, alpha
64, target q/k/v/o/gate/up/down), **share weight** giữa encode_query và encode_passage (trích rõ: theo đúng
thiết kế RepLLaMA gốc — 1 decoder xử lý cả 2 luồng, pooling = last-token/[EOS], similarity = cosine sau khi
normalize). Loss = InfoNCE/contrastive với in-batch + explicit negatives (group_size). Không đề cập "2
encoder riêng" ở mục này.

### 3.3 Data Augmentation
Sliding-window trên chuỗi tương tác train, giữ thứ tự thời gian (kiểu SBR sub-session), sinh thêm cặp
(partial-history, next-item). Lý giải cơ chế: bù mật độ supervision so với causal-attention (SASRec được
supervise ở *mọi* vị trí trong 1 forward pass, bi-encoder mặc định chỉ có 1 cặp/example nếu không augment).
Số liệu ablation minh chứng: đưa vào §4.6, không lặp lại số ở đây.

### 3.4 Post-hoc History Filter
**[SỬA tên mục]** ~~"Post filtering and ranker"~~ → **"Post-hoc History Filter"** (không có reranker riêng,
theo xác nhận). Mô tả: sau khi FAISS retrieve top-K, loại các item đã có trong lịch sử user khỏi danh sách
trước khi tính metric — 2 chế độ `ctx` (chỉ loại item trong query window) và `full` (loại toàn bộ lịch sử
trước item test, tương đương protocol ẩn của SASRec). Không cần train lại, không cần GPU. Lý giải cơ chế
contamination (đoạn lý thuyết đã thảo luận: loss không tạo negative trùng verbatim query → không có tín hiệu
phạt việc rank item đã tiêu thụ lên cao).

---

## 4 Experiments

### 4.1 Datasets
Giữ nguyên bảng thống kê đã có trong bản nháp (Beauty/Sports/ML-1M) — khớp `paper_report.md` §1.1. Thêm mô tả
5-core filter + leave-one-out split + full ranking (không sample negative) ngay dưới bảng.

### 4.2 Experimental Setup
**[SỬA — nhận thông số dời từ Method §3.2 theo yêu cầu]** Backbone: **Qwen3-Embedding-0.6B** (cấu hình
chính, dùng cho toàn bộ headline results) và **Qwen3-Embedding-4B** (model-scale study, §4.8). LoRA:
rank $r=16$, scaling $\alpha=64$, dropout $=0.1$, target modules = q/k/v/o/gate/up/down-proj (7 module/layer,
đã verify từ `tevatron/src/tevatron/retriever/arguments.py`). Temperature InfoNCE $\tau=0.01$. Training:
Tevatron v2, DeepSpeed ZeRO-2, learning rate $1\text{e-}4$, 3 epochs, hardware RTX 3060 12GB/32GB RAM (từ
`README.md` mục Yêu cầu phần cứng — nên nêu rõ vì đây là constraint ảnh hưởng trực tiếp đến §4.8 model
scale). Baseline ID-based train qua RecBole 1.2.1, `MAX_ITEM_LIST_LENGTH=200`, cross-entropy full-softmax.

**[CẦN BẠN CUNG CẤP]** Lý do chọn cụ thể **Qwen3-Embedding** làm backbone (ngoài "phù hợp phần cứng hạn chế"
đã ghi trong README — có so sánh với embedding model khác trước khi chọn không, hay chọn vì lý do khác như
đã có sẵn bản 0.6B/4B cùng family để làm scale study?). Cần 1-2 câu cho §4.2 để không bị hỏi "why this
specific backbone" khi review.

### 4.3 Evaluation Metrics
**[SỬA]** ~~Recall@K, MRR@K (K=5,10)~~ → **NDCG@{5,10,20}, HR@{5,10,20} (=Recall@K vì 1 positive/query),
MRR@{5,10,20}** — lý do: toàn bộ số liệu đã tính sẵn ở paper_report.md dùng bộ 9-metric này, đổi về
Recall/MRR@5,10 sẽ mất thông tin NDCG (metric chuẩn nhất trong SR literature để so sánh, vì có tính đến vị
trí rank) mà không tiết kiệm được công sức nào (dữ liệu đã có sẵn). Giữ định nghĩa Recall@K/MRR@K y nguyên
văn phong bản nháp, chỉ bổ sung NDCG@K.

### 4.4 Baseline Methods
**[SỬA — mở rộng theo Overall.pdf]** Nhóm ID-based: GRU4Rec, SASRec, SR-GNN. Nhóm generative-retrieval /
LLM-based (số liệu literature, xem nguồn §4.5): TIGER, ActionPiece, EAGER-LLM, LlamaRec, P5, E4SRec, GLoSS-8B.
Cộng Zero-shot (Qwen3-Embedding, không fine-tune) làm lower-bound.

**[CẦN BẠN CUNG CẤP]** Citation chính xác cho các model sau (chưa đủ tự tin trích đúng venue/năm để đưa vào
bibliography): **ActionPiece**, **EAGER-LLM**, **GLoSS-8B**. Các model còn lại tôi tạm ghi theo hiểu biết,
nhờ bạn double-check trước khi final:
- TIGER: Rajput et al., "Recommender Systems with Generative Retrieval", NeurIPS 2023 (đã có, chắc chắn).
- P5: Geng et al., "Recommendation as Language Processing (P5)", RecSys 2022 (khá chắc).
- E4SRec: Li et al., "E4SRec: An Elegant Effective Efficient Extensible Solution of LLMs for SR" (khá chắc
  về tên/nội dung, chưa chắc chắn 100% venue/năm chính xác).
- LlamaRec: Yue et al., "LlamaRec: Two-Stage Recommendation using LLMs for Ranking" (khá chắc về tên/nội
  dung, chưa chắc chắn 100% venue).

### 4.5 Overall Results
**[SỬA 2026-07-27 — quyết định dưới đây đã bị đảo ngược sau đó, xem `paper_draft.md`/`fade_paper.tex`]**
Ban đầu khuyến nghị tách 3 bảng riêng vì lo ngại quá rộng — nhưng để tiết kiệm diện tích (page limit), tác
giả sau đó yêu cầu **gộp Beauty+Sports thành 1 bảng ngang** (nhóm cột theo dataset, dùng hết chiều rộng
trang). Đã test compile thật (LaTeX `\tiny` + `\tabcolsep` hẹp) — không tràn lề, đọc được. Giữ đoạn dưới đây
làm lý do quyết định ban đầu, không phản ánh bảng hiện tại.

Tách **3 bảng riêng** (Beauty/Sports/ML-1M) —
khuyến nghị thay vì gộp 1 bảng vì 11 method × 6 metric sẽ quá rộng cho khổ trang LNCS 1 cột. Cột: R@5, R@10,
NDCG@5, NDCG@10, MRR@5, MRR@10. Ô trống trong nguồn giữ nguyên "—" (N/A).

**Bảng Beauty** (nguồn: `Overall.pdf`, đã sửa 1 ô — xem flag ở trên):

| Method | R@5 | R@10 | NDCG@5 | NDCG@10 | MRR@5 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| GRU4Rec | 0.0164 | 0.0283 | 0.0099 | 0.0137 | — | — |
| SASRec | 0.0387 | 0.0605 | 0.0249 | 0.0318 | — | — |
| SR-GNN | 0.0363¹ | 0.0552 | 0.0252 | 0.0312 | 0.0215 | 0.0240 |
| TIGER | 0.0454 | 0.0648 | 0.0321 | 0.0384 | — | — |
| ActionPiece | 0.0511 | — | 0.0340 | — | — | — |
| EAGER-LLM | 0.0548 | 0.0830 | 0.0369 | 0.0459 | — | — |
| LlamaRec | 0.0852 | 0.1524 | 0.0543 | 0.0759 | 0.0440 | 0.0529 |
| P5 | 0.0503 | — | 0.0370 | — | — | — |
| E4SRec | 0.0525 | — | 0.0360 | — | — | — |
| GLoSS-8B | 0.0681 | — | 0.0442 | — | — | — |
| **FADE-0.6B (Ours)** | **0.0730** | **0.1088** | **0.0501** | **0.0616** | **0.0427** | **0.0474** |

¹ Sửa từ 0.0182 (lỗi nguồn, vi phạm ràng buộc NDCG@K ≤ R@K) → 0.0363, khớp `paper_report.md`.

**Bảng Sports** (nguồn: `Overall.pdf`, không phát hiện lỗi):

| Method | R@5 | R@10 | NDCG@5 | NDCG@10 | MRR@5 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| GRU4Rec | 0.0129 | 0.0204 | 0.0086 | 0.0110 | — | — |
| SASRec | 0.0233 | 0.0350 | 0.0154 | 0.0192 | — | — |
| SR-GNN | 0.0200 | 0.0310 | 0.0130 | 0.0165 | 0.0107 | 0.0122 |
| TIGER | 0.0264 | 0.0400 | 0.0181 | 0.0225 | — | — |
| ActionPiece | 0.0316 | — | 0.0205 | — | — | — |
| EAGER-LLM | 0.0373 | 0.0569 | 0.0251 | 0.0315 | — | — |
| LlamaRec | — | — | — | — | — | — |
| P5 | 0.0272 | — | 0.0169 | — | — | — |
| E4SRec | 0.0281 | — | 0.0196 | — | — | — |
| GLoSS-8B | 0.0364 | — | 0.0238 | — | — | — |
| **FADE-0.6B (Ours)** | **0.0447** | **0.0684** | **0.0294** | **0.0371** | **0.0244** | **0.0275** |

**Bảng ML-1M** (**[SỬA — dùng `paper_report.md` §2 thay vì Overall.pdf]** — xem flag #3 ở trên; không có
baseline generative-retrieval literature nào vì TIGER gốc không có kết quả ML-1M, §6 `paper_report.md`):

| Method | R@5 | R@10 | NDCG@5 | NDCG@10 | MRR@5 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| SASRec | 0.1303 | 0.2089 | 0.0823 | 0.1077 | 0.0666 | 0.0770 |
| SR-GNN | 0.1522 | 0.2215 | 0.1011 | 0.1235 | 0.0844 | 0.0936 |
| GRU4Rec | 0.1768 | 0.2641 | 0.1210 | 0.1491 | 0.1026 | 0.1142 |
| **FADE-0.6B (Ours)** | 0.1276 | 0.1887 | 0.0846 | 0.1041 | 0.0705 | 0.0785 |

Lưu ý: FADE-0.6B **thua** GRU4Rec/SR-GNN ở ML-1M — không né tránh, dẫn sang §4.8 (FADE-4B đảo ngược kết quả).

**[CẦN FOOTNOTE RÕ NGUỒN khi vào draft]**: Beauty/Sports — GRU4Rec/SASRec + toàn bộ nhóm LLM-based
(TIGER...GLoSS-8B) là số liệu **trích từ literature** (không reproduce trong repo này), SR-GNN + FADE-0.6B là
**tự reproduce**. ML-1M — SASRec/SR-GNN/GRU4Rec + FADE-0.6B đều **tự reproduce**. Việc trộn nguồn
literature/tự-reproduce trong cùng 1 bảng (Beauty/Sports) cần ghi rõ để reviewer không hiểu nhầm là cùng
protocol — nêu caveat này ở đầu §4.5 hoặc trong caption bảng.

### 4.6 Ablation Study
**[SỬA — dùng Ablation-study.pdf, đã sửa header — xem flag #1 ở trên]**

| Variant | Beauty NDCG@10 | Beauty R@10 | Sports NDCG@10 | Sports R@10 | ML-1M NDCG@10 | ML-1M R@10 |
|---|---:|---:|---:|---:|---:|---:|
| **FADE (full)** | **0.0616** | **0.1088** | **0.0371** | **0.0684** | **0.1041** | **0.1887** |
| − History Filter | 0.0390 | 0.0905 | 0.0260 | 0.0574 | 0.0618 | 0.1343 |
| − Augmentation | 0.0581 | 0.1019 | 0.0279 | 0.0523 | 0.0691 | 0.1260 |
| Zero-shot (+ filter)² | 0.0202 | 0.0364 | 0.0102 | 0.0195 | 0.0128 | 0.0214 |

² Ghi rõ là biến thể "zero-shot + filter" (không phải zero-shot raw, số thấp hơn nữa — xem
`paper_report.md` §3) để tránh nhầm lẫn khi so với zero-shot ở bảng §4.5.

Nêu rõ cặp ablation nào "sạch" (cô lập đúng 1 biến) và cặp nào có confound — theo đúng chú thích đã có trong
`paper_report.md` §3 (VD: cặp augmentation ở ML-1M 0.6B đổi cả group_size, không tách biệt tuyệt đối).

### 4.7 Design Choice Validation
**[SỬA 2026-07-27 — mục này đã bị XOÁ khỏi bài để tiết kiệm diện tích]** Nội dung 2 thực nghiệm nhỏ dưới đây
đã được rút gọn thành 1 đoạn ngắn (không bảng) thêm vào cuối §4.2 Experimental Setup — xem `paper_draft.md`
§4.2 / `fade_paper.tex`. §4.8 Model Scale renumber thành §4.7, §4.9 Discussion renumber thành §4.8. Giữ
nguyên nội dung gốc dưới đây làm tài liệu tham khảo/lý do quyết định.

**[MỚI — nhận nội dung dời từ Method §3.2/§3.4 theo yêu cầu]** Hai thực nghiệm nhỏ, xác nhận 2 design choice
đã nêu (nhưng không chứng minh) ở Method — đặt cùng 1 mục vì cùng bản chất "tại sao chọn X thay vì Y":

**(a) Negative sampling: random vs. BM25.** Từ Method §3.2: "explicit negatives sampled uniformly at random
... §4.7 empirically validates this choice against BM25". Bảng (Beauty, ablation sớm của dự án):

| Negative strategy | NDCG@10 (raw) |
|---|---:|
| Random (mặc định, dùng cho mọi kết quả trong paper) | 0.0329 |
| BM25 hard negatives | 0.0166 (−49.5%) |

Lý giải: BM25 ưu tiên tìm item có overlap từ vựng cao với query — thường là item **cùng category/brand**,
tức nhiều khả năng là true positive bị gán nhầm negative ("false-negative poisoning"), khiến model học cách
**tránh** retrieve các item liên quan thay vì tìm chúng. → củng cố thêm luận điểm ở §3.4 về lexical-overlap
bias trong dense retriever, theo hướng ngược lại (âm tính giả thay vì dương tính giả).

**(b) Filter granularity: ctx vs. full.** Từ Method §3.4: "§4.7 additionally examines a lighter variant...".
Bảng (headline model mỗi dataset, từ `paper_report.md` §7):

| Dataset | NDCG@10 raw (không filter) | NDCG@10 ctx | NDCG@10 full (= headline FADE) |
|---|---:|---:|---:|
| Beauty | 0.0390 | 0.0590 | **0.0616** |
| Sports | 0.0260 | 0.0367 | **0.0371** |
| ML-1M | 0.0618 | 0.0850 | **0.1041** |

`full` luôn ≥ `ctx` (full là superset của ctx). Khoảng cách 2 chế độ nhỏ ở Beauty/Sports (context 3-5 item ≈
phần lớn lịch sử vốn đã ngắn, ~8-9 item trung bình) nhưng lớn ở ML-1M (context chỉ là phần nhỏ của trung bình
165.5 item/user) — nhất quán với lý do chọn `full` làm mặc định trong Method.

### 4.8 Effect of Model Scale (0.6B vs 4B)
**[SỬA 2026-07-27 — tạm hoãn viết theo yêu cầu tác giả]** Chưa viết nội dung — chờ số liệu
Qwen3-Embedding-4B đầy đủ cho **cả Beauty và Sports** (hiện chỉ có ML-1M, train trên máy khác, chưa
verify lại được — xem `paper_report.md` §4/Ghi chú phương pháp luận #5) mới có căn cứ so sánh.

**Thay đổi khung trình bày khi viết** (khác hẳn bản cũ):
- **Bỏ hoàn toàn khung "gap"** — không còn so sánh FADE-0.6B thua GRU4Rec/SR-GNN trên ML-1M rồi
  "FADE-4B đảo ngược/vượt qua" baseline. Mục này **không so sánh với baseline ID-based nào cả**.
- Chỉ tập trung đúng 1 thông điệp: **so sánh FADE-0.6B vs FADE-4B**, kết luận đơn giản "model lớn
  hơn cho kết quả tốt hơn" — không cần liên hệ tới GRU4Rec/SR-GNN/SASRec trong mục này.
- **Dùng bar chart thay vì bảng** — biểu diễn NDCG@10 (có thể thêm HR@10) của 0.6B vs 4B, nhóm theo
  dataset (Beauty/Sports/ML-1M), khi đã có đủ 3 dataset.
- Giữ nguyên caveat về nguồn số liệu ML-1M (train máy khác, chưa verify) khi viết.

**Ảnh hưởng tới các phần khác** (đã sửa để nhất quán, xem `paper_draft.md`/`fade_paper.tex`):
- Introduction (đoạn 5): bỏ câu "scaling to 4B reverses this outcome and surpasses all baselines" —
  chỉ giữ quan sát 0.6B thua baseline trên ML-1M (vẫn đúng, có số liệu đầy đủ ở §4.5), đổi phần 4B
  thành câu hỏi đang nghiên cứu thay vì kết luận đã chốt.
- §4.5 (Overall Results): câu dẫn "§4.8 shows that this gap closes, and reverses" đổi thành câu
  trung tính hơn, không hứa trước kết luận mà §4.8 (giờ hoãn) chưa có bằng chứng đầy đủ.

### 4.9 Discussion and Limitations
- **[MỚI]** Robustness của margin so với SASRec: bảng headline §4.5 dùng SASRec literature (Overall.pdf,
  NDCG@10=0.0318 Beauty / 0.0192 Sports). Để tránh nghi ngờ về khác biệt protocol giữa nguồn literature và
  repo này, nêu thêm: khi tự train lại SASRec dưới **cùng protocol/hardware** với FADE (RecBole 1.2.1,
  MAX_ITEM_LIST_LENGTH=200, cùng 5-core/leave-one-out split), FADE vẫn vượt **+49.5% NDCG@10 (Beauty)** /
  **+68.6% NDCG@10 (Sports)** — số nhỏ hơn nhưng vẫn rất lớn, chứng tỏ kết quả không phải do khác biệt
  protocol giữa 2 nguồn baseline (số liệu tự-reproduce đầy đủ ở `paper_report.md` §2, có thể đưa thành bảng
  phụ trong Appendix nếu còn chỗ).
- Mixing nguồn literature (TIGER, ActionPiece, EAGER-LLM, LlamaRec, P5, E4SRec, GLoSS-8B, và GRU4Rec/SASRec ở
  Beauty/Sports) với nguồn tự-reproduce (SR-GNN, FADE, toàn bộ ML-1M) trong bảng §4.5 — cần nêu rõ giới hạn
  này (protocol/preprocessing có thể lệch nhẹ giữa các paper nguồn dù cùng benchmark danh nghĩa).
- Thiếu baseline literature TIGER/S³-Rec verified đầy đủ (numbers trong bảng đã lấy nhưng cần re-check lại
  1 lượt trước khi final so với bản PDF gốc).
- Thiếu 4B đầy đủ cho Beauty/Sports.
- Single-GPU 12GB constraint ảnh hưởng đến scope thực nghiệm scale lớn hơn.

---

## 5 Conclusion
Tóm tắt 3 đóng góp chính + kết quả headline + hướng mở (scale lớn hơn cho Beauty/Sports, literature
baseline).

---

## Việc cần làm trước khi viết draft đầy đủ (checklist)

- [x] Xác nhận outline khớp ý định — đã bỏ hoàn toàn nhánh untied dual-encoder (Introduction, Related Work,
      Discussion, Conclusion), thay bằng thách thức "động lực chọn kiến trúc bi-encoder" ở §1.
- [x] Ví dụ minh họa đầy đủ cho §1 (routine skincare Beauty) — tái dùng xuyên suốt đoạn 2 và thách thức #3.
- [ ] Lấy số liệu TIGER/S³-Rec gốc nếu muốn đưa vào Related Work/so sánh định tính (không bắt buộc).
- [ ] Venue cụ thể — **để sau**, chưa cần quyết định ở giai đoạn này (LNCS + EquinOCS trong bản nháp gợi ý
      Springer proceedings, nhưng không chặn việc viết draft; page limit có thể canh lại khi đã có bản đầy đủ).
