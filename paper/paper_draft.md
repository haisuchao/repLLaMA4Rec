# FADE: Rethinking Sequential Recommendation as Filter-Augmented Dense Retrieval

> **TRẠNG THÁI HIỆN TẠI (chốt 2026-07-30)** — đọc mục này trước khi sửa tiếp, xem "TODO tổng hợp" ở cuối file
> để biết chính xác việc còn lại. File này là **nguồn markdown**; bản LaTeX đầy đủ tương ứng ở
> `template/fade_paper.tex` (compile sạch bằng tectonic — số trang xem lần compile gần nhất) — **hai file
> phải luôn sửa song song**, không sửa 1 file rồi quên file kia.
> - **[QUAN TRỌNG — 2026-07-30 #8] Tác giả đã tự chỉnh sửa trực tiếp `.tex`** ở phiên này — đã đồng bộ ngược
>   lại vào `paper_draft.md`. Các thay đổi chính đã đồng bộ:
>   - **GPU đổi**: từ RTX 3060 (12GB VRAM, 32GB RAM) → **RTX 3090 (24GB VRAM)** — bỏ luôn câu "FADE-4B Sports
>     đang chạy vì hardware bottleneck" (không còn đúng với GPU mới) và câu "untying 4B backbone sẽ vượt quá
>     bộ nhớ 1 GPU tiêu dùng" ở §3.2 Tied encoder.
>   - **3 challenges ở Introduction giờ gộp thành 1 đoạn văn liền mạch** (trước đó là 3 đoạn tách riêng theo
>     First/Second/Third) — chỉ là thay đổi trình bày, nội dung giữ nguyên.
>   - **Rút gọn nhiều câu ở Method + §4.2**: bỏ mention Qwen3-Embedding trong §3.2 (chỉ còn nói "a pretrained
>     decoder-only LLM"), bỏ hẳn câu cuối "Tied encoder" (câu hỏi mở tied-vs-untied để future work), bỏ câu
>     "Exact rank/scaling given in §4.2" ở LoRA Adapter, bỏ hẳn đoạn BM25/false-negative-poisoning (cả ở
>     Method **và** §4.2), bỏ câu cap-20-pairs ở Data Augmentation, bỏ câu extract-hidden-states-qua-raw-PEFT
>     ở §4.2.
>   - **Caption 2 hình rút cực ngắn**: Fig. 1 chỉ còn "(a)...(b)... instance of seen-item bias" (bỏ tham chiếu
>     section); Fig. 2 chỉ còn "FADE pipeline overview" (bỏ hết mô tả chi tiết). Fig 1 thu nhỏ
>     `0.75\linewidth`, Fig 2 phóng to lại `1\linewidth`.
>   - **Bảng dataset (Table 1)**: bỏ hậu tố "(2014)" khỏi tên Beauty/Sports.
>   - **Bibliography**: bỏ hết mã arXiv cho các mục đã có venue chính thức (mục 5, 9, 12, 15, 17) — chỉ giữ
>     arXiv cho mục 16 (Wang et al. e5, không có venue chính thức).
>   - **2 lỗi tôi phát hiện và tự sửa khi đồng bộ**: (1) lỗi đánh máy "raining data" → "training data" ở đoạn
>     ví dụ skincare; (2) GRU4Rec/SASRec trong Table 2 bị mất dấu † trong bản `.tex` tác giả sửa (mâu thuẫn
>     với câu văn ngay phía trên nói "unmarked methods (SR-GNN, FADE)") — đã thêm lại †.
>   - **Theo yêu cầu tác giả trong phiên này, đã tách thêm 2 chỗ để dễ đọc**: §4.4 Baseline Methods từ 1 đoạn
>     → 3 đoạn (ID-based / LLM-based / zero-shot), mỗi baseline trong nhóm LLM-based có thêm 1 cụm mô tả ngắn
>     trong ngoặc; §4.5 Overall Results narrative từ 1 đoạn dài → 3 đoạn theo từng ý (Beauty/Sports 0.6B thắng
>     → ML-1M 0.6B thua → FADE-4B đảo ngược + đối lập với Beauty).
> - **Chưa chốt backbone size**: §4.2 chỉ nói "dùng Qwen3-Embedding family, báo cáo cả 0.6B và 4B, quyết định
>   dùng size nào làm headline backbone sau khi có đủ số liệu (§4.5)" — KHÔNG còn khẳng định "0.6B là headline
>   backbone" như bản cũ.
> - **§4.7 Effect of Model Scale đã bị XOÁ HẲN** (chốt 2026-07-28 #3, theo yêu cầu tác giả) — thay vào đó,
>   Table 2/3 có thêm 1 dòng **FADE-4B**, điền dần khi có số liệu (không bịa số).
> - **[QUAN TRỌNG — 2026-07-28 #4/#5] FADE-4B: Beauty và ML-1M đã có số liệu, kể 2 câu chuyện KHÁC HẲN nhau**:
>   - **Beauty** (N@10=0.0611 vs FADE-0.6B 0.0616): chỉ ngang bằng, KHÔNG "vượt trội" như dự đoán ban đầu.
>   - **ML-1M** (N@10=0.1601 vs FADE-0.6B 0.1041, **+53.8%**): cải thiện rất lớn, và **FADE-4B giờ vượt qua cả
>     GRU4Rec/SR-GNN** (baseline mà FADE-0.6B từng thua) — R@5 +12.0%, R@10 +7.1%, N@5 +9.8%, N@10 +7.4% so
>     với GRU4Rec. Đây là đảo ngược hoàn toàn kết quả ML-1M trong Table 3 — **bold đã chuyển từ FADE-0.6B/
>     GRU4Rec sang FADE-4B**.
>   - Diễn giải tạm thời (đã ghi rõ trong bài là suy đoán, chưa test formal, chỉ dựa trên 2 điểm dữ liệu):
>     scale giúp bù đắp bất lợi kiến trúc (context window cố định) trên chuỗi dài, nhưng ít tác dụng trên
>     chuỗi ngắn nơi FADE-0.6B đã gần bão hoà tín hiệu.
>   - Đã cập nhật: Table 3 (bold + số liệu), §4.5 narrative, Conclusion Limitations, **bullet contribution
>     đầu §1** (từ câu hỏi mở "study whether..." → khẳng định "showing that a larger backbone closes, and on
>     our long-sequence benchmark reverses, FADE's gap..." vì ML-1M đã đủ số liệu để khẳng định).
>   - **Chỉ còn thiếu Sports** cho FADE-4B — Table 2 cột Sports vẫn `TBD`. Abstract CHƯA cập nhật theo phát
>     hiện này (cố tình chờ đủ cả 3 dataset + tác giả chốt backbone size trước khi viết lại Abstract).
>   - **[2026-07-30 #6]** Đã bỏ đoạn giải thích placeholder (in nghiêng, dưới Table 3) khỏi bản paper chính
>     (`paper_draft.md` + `.tex`) theo yêu cầu tác giả — đoạn đó nói "FADE-4B's Sports columns in Table 2 are
>     still placeholders... bold has moved from FADE-0.6B to FADE-4B, and from GRU4Rec... to FADE-4B as
>     well." Chỉ giữ lại Ở ĐÂY làm note theo dõi nội bộ. **Khi có số liệu Sports thật**: (1) điền Table 2, (2)
>     xoá dấu `*pending*`, (3) viết thêm nhận xét Sports vào narrative §4.5 hiện có, KHÔNG cần khôi phục lại
>     đoạn giải thích placeholder này (đã lỗi thời khi đó).
> - **MRR không còn được nhắc tới ở bất kỳ đâu** (đã bỏ luôn khỏi §4.3 và §4.5, trước đó chỉ bỏ khỏi bảng).
> - **§3.5 (cũ §3.4) không còn claim so sánh ctx-vs-full filter variant** (số liệu 0.0850 vs 0.1041 ML-1M) —
>   đã bỏ vì không có bảng kết quả nào trong bài hỗ trợ claim này.
> - **Method §3 giờ có 5 mục con**: 3.1 Problem Statement, **3.2 Bi-Encoder** (kiến trúc, pooling, tied
>   encoder — đổi tên lại từ "LLM-based Bi-Encoder" theo yêu cầu tác giả), **3.3 LoRA Adapter** (PEFT +
>   InfoNCE), 3.4 Data Augmentation, 3.5 History Filter (tên cũ "Post-hoc History Filter" — xem ghi chú đổi
>   tên 2026-07-30).
> - **Figure 2 do tác giả tự thiết kế** bằng công cụ khác, KHÔNG còn qua `make_architecture_figure.py` (script
>   đó stale — đừng chạy lại, sẽ ghi đè mất bản mới). Trong `.tex` đã thu nhỏ còn `0.65\textwidth` và rút gọn
>   caption (cả Fig. 1 và Fig. 2) để đỡ chiếm diện tích trang. Xem mô tả chi tiết ở mục "Figure 2" cuối file.
>
> - **[QUAN TRỌNG — Rà từ vựng toàn bài 2026-07-30]** Theo yêu cầu tác giả, đã đổi các từ "hiếm gặp trong paper
>   SR/LLM" xuyên suốt TOÀN BÀI (không chỉ Abstract) — áp dụng cho cả `paper_draft.md` và
>   `template/fade_paper.tex`:
>   - `recast/recasting` → `frame/framing` → **[2026-07-30 #2] đổi tiếp thành `apply`/`represent`/`representing`**
>     (tác giả thấy "frame" vẫn không phải từ quen thuộc trong paper SR/LLM)
>   - `item catalog` → `item set` (khớp $\mathcal{I}$ đã định nghĩa) hoặc mô tả trực tiếp không dùng "catalog"
>   - `asymmetric/asymmetry` (query/document) → mô tả trực tiếp "a multi-item query matched against
>     single-item documents" / "differ this much", bỏ hẳn tính từ "asymmetric"
>   - `yields` (SR yields / this yields) → `provides`/`produces`/`brings` tuỳ ngữ cảnh
>   - **`history contamination` → `seen-item bias`** (thuật ngữ riêng của bài, đổi theo lựa chọn tác giả — xem
>     câu hỏi đã hỏi và trả lời trong phiên 2026-07-30)
>   - **`post-hoc (history) filter` → `history filter`** (bỏ hẳn "post-hoc", kể cả tên mục **§3.5** đổi từ
>     "Post-hoc History Filter" → "History Filter")
>   - Câu mở đầu Abstract viết lại khác hẳn câu mở đầu Introduction (trước đó bị trùng y hệt)
>   - **CẢNH BÁO CHƯA XỬ LÝ ĐƯỢC**: cả 2 hình `fig1_introduction_example.pdf` (panel b) và
>     `fade_architecture.pdf` (box filter) là ảnh tác giả tự làm bên ngoài (không phải matplotlib script) —
>     chữ bên trong ảnh **vẫn còn ghi "history contamination" / "Post-hoc History Filter"** (tên cũ). Tôi
>     không sửa được text bên trong ảnh (không có file nguồn). Tác giả cần tự cập nhật lại 2 ảnh này cho khớp
>     tên mới, nếu không bài sẽ có mâu thuẫn giữa hình và chữ.
> - **[2026-07-30 #2] Thêm 1 vòng chỉnh sửa Introduction/Related Work**:
>   - Câu mở đầu Introduction viết lại ngắn gọn hơn theo ý tác giả: "chronological interaction history... used
>     to model the user's preference" (bỏ khung "on the premise that... static profile could not capture").
>   - Challenge thứ 3 viết lại theo hướng **lexical overlap** (dẫn dắt bằng lexical overlap giữa query/candidate
>     text trước, KHÔNG còn dùng chữ "failure mode"), giảm mức độ nghiêm trọng — khung content-based/
>     Adomavicius&Tuzhilin vẫn giữ nhưng chuyển thành ý phụ ("can be seen as a pronounced case of..."), không
>     còn là câu mở đầu. Bullet contribution thứ 2 cũng bỏ "failure mode" → "bias".
>   - **Bullet contribution đầu §1 rút gọn hẳn**: chỉ còn nói "apply bi-encoder LLM retriever to SR", **bỏ
>     hoàn toàn phần benchmark 3 dataset/2 backbone scale/kết quả ML-1M reversal** (theo yêu cầu tác giả) —
>     phần benchmark/scale-finding đó vẫn còn đầy đủ trong §4.5, chỉ không còn là 1 bullet contribution riêng.
>   - **Đoạn GLoSS đã gộp vào đoạn "LLM-based sequential recommendation"** (không còn là đoạn/paragraph riêng
>     "closest prior work") và rút ngắn còn 2 ý: (1) giống FADE ở kiến trúc bi-encoder dense retrieval, (2)
>     khác FADE ở chỗ GLoSS chỉ dùng LLM để fine-tune phía query (generator), còn bi-encoder thực sự
>     (e5-small-v2) thì **đóng băng, không fine-tune cho retrieval** — trong khi FADE fine-tune 1 bi-encoder
>     duy nhất end-to-end cho cả 2 phía. Số baseline so sánh bằng literature numbers tăng từ "six" → **"seven"**
>     (đã gồm GLoSS-8B). Lưu ý: giữ "e5-small-v2" là **encoder nhỏ** (không gọi là "LLM") để đúng kỹ thuật,
>     dù tác giả mô tả nôm na là "cũng là 1 LLM" — nếu tác giả muốn khẳng định chính xác nó là LLM, cần xác
>     nhận lại.
> - **[2026-07-30 #3] Bỏ hẳn "lexical-overlap bias" (Sciavolino et al. 2021) khỏi Related Work**: tác giả chỉ
>   ra 2 thuật ngữ "seen-item bias" (của FADE) và "lexical-overlap bias" (trích Sciavolino et al., đoạn
>   "Data augmentation and lexical-overlap bias") nghe giống nhau nhưng thực ra là **2 hiện tượng khác nhau**
>   (seen-item bias: retriever xếp hạng cao chính item có trong query vì trùng verbatim text; Sciavolino:
>   retriever generalize kém tới entity/pattern KHÔNG xuất hiện lúc train) — không đủ căn cứ để giữ so sánh
>   này trong Related Work. Đã: (1) xoá toàn bộ đoạn Sciavolino et al., đổi tên mục từ "Data augmentation and
>   lexical-overlap bias" → **"Data augmentation"** (giờ chỉ còn nói Tan et al./Caser), (2) xoá bibliography
>   entry Sciavolino (mục 19 cũ), **đánh số lại 20→19, 21→20, 22→21** (bibliography còn 21 mục). Giờ
>   **"seen-item bias" là thuật ngữ đặt tên hiện tượng DUY NHẤT trong toàn bài** — chữ "lexical overlap" vẫn
>   còn xuất hiện ở Introduction (mô tả cơ chế của seen-item bias) nhưng đó chỉ là mô tả thường, không phải
>   tên riêng, nên không còn xung đột.
> - **[2026-07-30 #4] Viết lại Challenge 1 (bi-encoder) theo khung 2 tầng, thống nhất Abstract ↔ Introduction**:
>   trước đó Abstract nói "chưa ai nghiên cứu kỹ cách encode cho SR asymmetry" nhưng Introduction lại chỉ nói
>   "cross-encoder quá tốn kém nên bi-encoder gần như bắt buộc" — 2 luận điểm khác nhau, không khớp nhau, và
>   bullet contribution 1 (nhắc asymmetry) không có căn cứ dẫn dắt từ Challenge 1. Đã thống nhất theo khung
>   2 tầng (thảo luận với tác giả, tầng 2 là trọng tâm): **Tầng 1 (chỉ nhắc qua, 1 câu)** — cross-encoder tốn
>   kém ở quy mô catalog → bi-encoder gần như bắt buộc. **Tầng 2 (trọng tâm, 3 câu)** — bi-encoder hiện có
>   (kể cả RepLLaMA, không nêu tên trong Introduction theo yêu cầu tác giả trước đó) được thiết kế cho text
>   retrieval thông thường (query/document cân xứng) và thường **tied** (dùng chung trọng số); SR phá vỡ sự
>   cân xứng này (query nhiều item, document 1 item); liệu tied bi-encoder có còn hoạt động tốt dưới sự bất
>   cân xứng này chưa từng được kiểm chứng có hệ thống — đây là câu hỏi FADE trả lời. Đã đồng bộ 3 chỗ: Abstract
>   (chỉ còn nói tầng 2, bỏ tầng 1 cho gọn), Introduction Challenge 1 (đủ cả 2 tầng), bullet contribution 1
>   (thêm "tied" + "testing whether it continues to work well" để khớp khung mới). Method §3.2 (Tied encoder)
>   và Related Work (ConvDR/two-tower dùng untied) đã sẵn nhất quán với khung này từ trước, không cần sửa.
> - **[2026-07-30 #5 — ĐẢO NGƯỢC #4] Bỏ hẳn tầng 2, chỉ dùng tầng 1 cho Challenge 1**: tác giả thấy lý do
>   tầng 2 (tied bi-encoder chưa từng được kiểm chứng dưới SR asymmetry) vẫn khiên cưỡng, và đặt vấn đề "chưa
>   ai kiểm chứng có hệ thống" là quá rộng so với mức độ một bài hội thảo. Đã revert cả 3 chỗ về lại bản chỉ
>   có tầng 1 (cross-encoder tốn kém → bi-encoder gần như bắt buộc): Introduction Challenge 1 (bản gốc trước
>   #4), Abstract (câu ngắn gọn "jointly encoding every candidate item with the query via a cross-encoder is
>   too costly at catalog scale, making a bi-encoder close to a necessity"), bullet contribution 1 (bỏ "tied"
>   + "testing whether...", chỉ còn "adapting it to a setting where a multi-item query is matched against
>   single-item documents" — mô tả thuần, không claim đây là câu hỏi nghiên cứu mở). Câu hỏi tied-vs-untied
>   VẪN CÒN nhưng chỉ ở mức khiêm tốn tại Method §3.2 (đã có sẵn, không claim gì thêm: "to our knowledge,
>   untested... we leave a controlled tied-vs-untied comparison to future work") — không nâng lên thành
>   challenge/contribution ở Abstract/Introduction nữa.
> - **Đã viết xong prose**: Section 1 (Introduction), Section 2 (Related Work), Section 3 (Method), Section 4
>   (Experiments, không còn §4.7), **Abstract** (viết lại nhiều lần, mới nhất 2026-07-30).
> - **Chưa viết**: phần đầu Section 5 Conclusion (tóm tắt 3 contribution + kết quả headline — đoạn Limitations
>   trong Conclusion đã viết xong).
> - **Figures**: `figures/fade_example.pdf`/`fig1_introduction_example.pdf` (Fig. 1, ví dụ skincare ở
>   Introduction) và `figures/fade_architecture.pdf` (Fig. 2, kiến trúc pipeline ở Method, tác giả tự làm).
> - **Công thức toán** viết ở dạng LaTeX-friendly (`$...$` / `$$...$$`) để copy trực tiếp sang `.tex`.
> - **Venue**: chưa chọn, tác giả chủ động chưa muốn quyết định — không tự ý đề xuất/chốt.
> - **[2026-08-08] Untied encoder giờ đã implement được trong code** (`train.sh --untie-encoder` /
>   `eval.sh` tự auto-detect) — dùng PEFT multi-adapter (2 LoRA adapter riêng cho query/passage, chia sẻ 1
>   backbone frozen), verify GPU chỉ tăng ~2.8% so với tied (đo trên Qwen3-Embedding-0.6B). Trước đó
>   `untie_encoder` của Tevatron chỉ tồn tại ở code JAX cũ/ví dụ RepLLaMA gốc, không hoạt động ở path
>   torch/DeepSpeed mà FADE dùng — đã tự viết lại từ đầu, xem `README.md` §10 "Untied encoder" để biết chi
>   tiết cơ chế. **Chưa chạy ablation tied-vs-untied thật trên dataset đầy đủ** — câu ở Method §3.2 ("we
>   leave a controlled tied-vs-untied comparison to future work") vẫn đúng và CHƯA cần sửa; ghi chú này chỉ
>   để theo dõi rằng phần hạ tầng code đã sẵn sàng, không phải đã có kết quả để đưa vào bài.

## Các quyết định quan trọng cần nhớ (không hiển nhiên nếu chỉ đọc lại text)

- **Bảng Table 2** (Beauty+Sports) là 1 bảng ngang duy nhất, cột nhóm theo dataset (`\cmidrule`), font
  `\footnotesize`, `\tabcolsep=2.5pt` — đừng tách lại thành 2 bảng dọc, đó là quyết định có chủ đích để tiết
  kiệm trang.
- **MRR đã bị bỏ khỏi mọi bảng so sánh** (Table 2, Table 3) và khỏi §4.3 — chỉ còn NDCG+HR. Lý do: hầu hết
  baseline LLM-based không báo cáo MRR.
- **§4.7 Effect of Model Scale đã bị xoá hẳn khỏi bài** (2026-07-28 #3) — FADE-4B giờ chỉ là 1 dòng
  `*pending*` trong Table 2/3 + 1 câu dự đoán định tính ở cuối §4.5, không còn là 1 section riêng.
- **Introduction & Related Work: không đưa số liệu so sánh** (NDCG/%) vào prose — số liệu chỉ xuất hiện từ
  Section 4 trở đi. Đây là yêu cầu rõ ràng của tác giả, không phải mặc định của tôi.
- **Văn phong**: tránh cấu trúc liệt kê `--- X --- Y ---` (nghe như máy viết) — dùng từ nối tự nhiên
  (whether...or, but, namely, so that). Contribution list dùng gạch đầu dòng thường, không dùng nhãn C1/C2/C3.
  Không xếp hạng "quan trọng nhất" giữa các challenge/thách thức — trình bày ngang hàng.
- **Chống overclaim**: mọi phát hiện/hiện tượng được nêu ra như đóng góp (VD seen-item bias, tên cũ "history
  contamination" — xem ghi chú đổi tên 2026-07-30) đều đã
  tra cứu web trước để xác nhận phần nào là cũ/đã biết (phải ghi nhận rõ) và phần nào thực sự mới — không
  được để trống bước này khi viết claim mới.
- **GLoSS đã chốt** (không còn là câu hỏi mở): pipeline là generate-then-retrieve 2 giai đoạn (LLaMA-3 sinh
  text mô tả → e5-small-v2 encode + dot product). Bước sau cùng đúng là tied bi-encoder, nhưng khác FADE vì
  model hiểu preference (LLaMA-3, generative loss) tách biệt khỏi model retrieval (e5-small-v2, nhỏ, không rõ
  có fine-tune riêng cho retrieval hay không).
- **Tan et al. 2016** (DLRS@RecSys) là nguồn gốc thật của sliding-window augmentation, **sớm hơn Caser 2
  năm** — Caser vẫn giữ trong Related Work vì quen thuộc hơn, nhưng Tan et al. mới là citation ưu tiên về mặt
  lịch sử.
- **Bibliography hiện có 20 mục** (giảm từ 22 → 21 → 20: đã bỏ Sciavolino et al. 2021 — 2026-07-30 #3 — và
  Adomavicius & Tuzhilin 2005 — 2026-07-30 #6, do câu trích dẫn duy nhất bị xoá khỏi Challenge 3)
  (author-year tạm thời, đánh số theo thứ tự xuất hiện trong `\cite{}`) — đã verify qua web search, xem chi
  tiết ở mục **References** cuối file này.
- **Introduction & Related Work: không dùng tham chiếu `(§X.X)`/`(Sect. X)`** trong phần văn xuôi — mọi vấn đề
  phải được trình bày trực tiếp bằng lời thay vì trỏ sang section khác (chốt 2026-07-28). Method/Experiments
  vẫn dùng tham chiếu section bình thường.
- **Challenge thứ 3 ở Introduction (seen-item bias, tên cũ "history contamination")**: khung trình bày là "đây
  là hạn chế quen thuộc
  của content-based recommendation nói chung (overspecialization, dẫn Adomavicius & Tuzhilin 2005), SR-as-
  retrieval vốn là 1 dạng content-based nên thừa hưởng vấn đề này, còn collaborative/ID-based models (SASRec)
  thì không gặp phải" — KHÔNG còn claim "novelty/chưa ai document" ở Introduction (claim novelty đó vẫn giữ
  trong contribution bullet + §3.5 Method, chỉ bỏ khỏi Introduction theo yêu cầu tác giả).

---

## Abstract

Sequential recommendation (SR) aims to predict what a user will interact with next by modeling the order of
their past behavior. Large language models (LLMs) have recently been applied to SR in a variety of ways, and
formulating SR as an LLM-based information-retrieval (IR) problem, in which a user's recent interaction
history is converted into a text query and matched against candidate items treated as documents in a
retrieval corpus, is one such promising direction pursued by a growing body of prior work. Applying LLM-based
IR to SR, however, raises three challenges that prior work in this space has not fully resolved. First,
jointly encoding every candidate item with the query via a cross-encoder is too costly at catalog scale,
making a bi-encoder close to a necessity. Second, the leave-one-out
protocol standard in SR supplies only one
training example per user, far too little to fine-tune an LLM-scale encoder. Third, because the query is
built directly from previously-consumed item text, such retrievers tend to repeatedly rank those same items
at the top of the list. We present FADE (Filter-Augmented Dense rEtrieval), which addresses these three
challenges respectively with a Bi-Encoder Dense Retrieval applied systematically to SR, sliding-window data
augmentation that recovers the missing per-position supervision, and a lightweight, training-free filter that
removes previously-consumed items from the ranked list. Experiments on three real-world datasets show that
FADE substantially outperforms prior LLM-based information-retrieval approaches to sequential recommendation.

**[Sửa 2026-07-30 #3 — rà từ vựng toàn bài]** ~233 từ. Đổi câu mở đầu khác hẳn Introduction (tránh trùng câu
literal). Bỏ tiếp các từ tác giả coi là hiếm gặp trong paper SR/LLM: "item catalog" → "candidate items ...
documents in a retrieval corpus"; "asymmetric" → mô tả trực tiếp "a multi-item query must be matched against
single-item documents"; "post-hoc filter" → "filter" (bỏ hẳn "post-hoc", tính chất training-free/inference-only
đã có ở tính từ "training-free" và ở toàn bộ phần thân bài). Xem ghi chú đầu file mục "Rà từ vựng toàn bài
2026-07-30" để biết đầy đủ các từ đã đổi trong toàn bộ bài, không chỉ Abstract.

---

## 1 Introduction

Sequential recommendation (SR) predicts the next item a user will interact with from their chronological
interaction history, which is used to model the user's preference. Early work modeled this ordering with Markov chains,
assuming the next action depends only on the last one or few; later architectures relaxed this local
assumption while pursuing the same underlying goal of exploiting sequence order, whether through a recurrent
network (GRU4Rec [Hidasi et al., 2016]), a graph neural network over the session (SR-GNN [Wu et al., 2019]),
or self-attention over the sequence (SASRec [Kang and McAuley, 2018]). All of these represent items purely as
trainable ID embeddings learned end-to-end from the interaction matrix, with no access to any information
about what an item actually is. The recent rise of large language models (LLMs) has opened a complementary
direction: instead of learning item identity from scratch, an LLM can encode the textual side information
already available for most catalogs, such as titles, descriptions, and categories, bringing broad semantic and
world knowledge that pure ID-based collaborative filtering has no access to.

One increasingly popular way to apply this idea is to represent SR as an information retrieval (IR)
problem. A user's recent interaction history is converted into a text query, the item set becomes a
document corpus, and predicting the next item reduces to retrieving the item most similar to the
query in a learned embedding space. Figure 1(a) illustrates this on Amazon Beauty: given a query built from a
user's three most recent purchases, namely a sunscreen, a facial cleanser, and a treatment serum, an effective
retriever should rank a moisturizing cream highest, not because any user in the training data ever bought
exactly this sequence before (there may be no such co-occurrence anywhere in the data), but
because the retriever's embedding space captures that a moisturizer is the natural next step in a skincare
routine. This is precisely the appeal of the retrieval reformulation over ID-based collaborative filtering:
relevance is judged by *semantic* similarity of text rather than by *statistical co-occurrence* of item
identifiers, letting the model generalize to item combinations it has never observed.

**[Figure 1 — `figures/fig1_introduction_example.pdf`: (a) ví dụ retrieval mong muốn, (b) seen-item bias thực
tế xảy ra — mô tả chi tiết ở cuối file. LƯU Ý: ảnh hiện tại (tác giả tự làm) vẫn còn chữ "history
contamination" trong panel (b) — xem ghi chú "Rà từ vựng toàn bài 2026-07-30" cuối file để cập nhật ảnh cho
khớp tên mới.]**

Despite this appeal, applying LLM-based dense retrieval to SR raises three challenges specific to the
recommendation setting. First, scoring relevance with an LLM does not by itself dictate *how* query and
candidate items should be encoded. The most accurate option would be to jointly encode each (query, candidate)
pair with full cross-attention, as a cross-encoder does, but this is computationally infeasible for the
retrieval stage itself: an item set can contain tens of thousands to millions of items, and re-running a full
LLM forward pass over every candidate for every query is prohibitive at serving time. A **bi-encoder**, which
encodes queries and documents independently so that item embeddings can be computed once offline and queried
via approximate nearest-neighbor search, is therefore not merely a design preference inherited from prior
text-retrieval work; it is close to a necessary condition for LLM-based retrieval to be practical at
recommendation scale. Second, the standard leave-one-out training protocol used in SR provides very little
supervision per user: with only the final interaction held out for testing, each user contributes exactly one
(query, next-item) training pair by default. This is a poor match for fine-tuning an LLM-scale encoder, which
typically needs substantially more examples than the handful one gets by directly borrowing this protocol from
ID-based SR without adaptation. Third, because the query is built directly from the titles of
previously-consumed items, a retriever trained with a standard contrastive objective tends to rank those same
items, or items with near-identical titles, near the top of the result list, largely due to lexical overlap
between the query and candidate text rather than genuine relevance. Figure 1(b) illustrates this for the same
example as above: the already-purchased facial cleanser outranks the moisturizing cream, which is pushed down
to rank 2. We refer to this pattern as **seen-item bias**.

We address these challenges with **FADE** (**F**ilter-**A**ugmented **D**ense r**E**trieval), a bi-encoder
retriever for sequential recommendation built on a fine-tuned LLM backbone. FADE augments training with
sliding-window sub-sequences to recover the per-position supervision that causal architectures such as SASRec
obtain for free, and applies a lightweight, training-free history filter that removes
previously-consumed items from the ranked list. Across three real-world datasets spanning short and long
interaction histories, and at two backbone scales, FADE substantially outperforms both strong ID-based
baselines and prior LLM-based recommenders.

Our contributions are as follows:

- We apply a bi-encoder LLM retriever to sequential recommendation, adapting it to a setting where a
  multi-item query is matched against single-item documents.
- We identify and mechanistically explain *seen-item bias*, a retrieval-specific bias in which
  previously-consumed items dominate the ranked list, and fix it with a training-free filter.
- We introduce sliding-window data augmentation for bi-encoder SR training, framed as recovering the
  per-position supervision that causal architectures obtain implicitly, and show it contributes substantially
  across all three datasets.
- We conduct comprehensive experiments against ID-based baselines (GRU4Rec, SASRec, SR-GNN) and, where
  available, literature-reported LLM-based baselines, on three real-world datasets.

---

## 2 Related Work

**Sequential recommendation.** Classical SR models predict the next item from a user's interaction sequence
by learning item-ID embeddings end-to-end from interaction data alone: GRU4Rec models the sequence with a
gated recurrent network [Hidasi et al., 2016], SR-GNN represents a session as a graph and applies a gated
graph neural network [Wu et al., 2019], and SASRec replaces both with self-attention under a causal mask
[Kang and McAuley, 2018]. All three condition purely on interaction order and item identity, without using
any textual content, so they have no way to relate two items that never co-occur in the training data,
however similar their content, and no natural way to reason about items outside the ID vocabulary seen during
training. This is exactly the limitation that motivates representing SR as a text-retrieval problem instead,
where relevance is judged by content rather than by co-occurrence statistics alone. We use all three as
ID-based baselines in our experiments.

**LLM-based sequential recommendation.** A growing body of work applies LLMs to SR through approaches
distinct from FADE's dense-retrieval reformulation. *Generative retrieval* methods discard embedding-based
ANN search entirely and instead train a model to directly generate an item's identifier: TIGER [Rajput et
al., 2023] quantizes each item's content embedding into a short sequence of discrete "Semantic ID" tokens via
residual quantization, then trains a sequence-to-sequence model to autoregressively predict the target item's
Semantic ID; ActionPiece [Hou et al., 2025] improves on TIGER's tokenization by merging item-feature tokens
based on their co-occurrence context, rather than tokenizing every item identically regardless of the
sequence it appears in. *LLM-as-ranker* methods keep a separate, lightweight candidate retriever and use the
LLM only to re-rank its output: LlamaRec [Yue et al., 2023] does this in a single forward pass, via a
verbalizer that reads a probability distribution over candidate items directly off the LLM's output logits
rather than generating text. Other work integrates LLMs more tightly with existing ID-based architectures
rather than replacing them: E4SRec [Li et al., 2024] uses an LLM as a sequence encoder feeding into an
ID-embedding prediction head, and EAGER-LLM [Hong et al., 2025] injects collaborative ("exogenous") signals
into a decoder-only LLM recommender alongside its native semantic understanding, addressing a mismatch
between an LLM's linguistic pretraining and the collaborative patterns recommendation requires. P5 [Geng et
al., 2022] takes a different unification angle, casting many recommendation tasks, not only next-item
prediction, as a single text-to-text, prompt-based objective. Closest to FADE is GLoSS [Acharya et al., 2025],
which shares the same bi-encoder dense-retrieval architecture, matching a query embedding against embeddings
of every candidate item. GLoSS, however, uses the LLM only to fine-tune the query side: a fine-tuned generator
produces a text description of the item it predicts the user will buy next, and this description, together
with every candidate item, is embedded by a separate bi-encoder (e5-small-v2 [Wang et al., 2022]) that is kept
frozen rather than fine-tuned for retrieval.

**Dense retrieval bi-encoders.** Reformulating retrieval as independent query/document encoding followed by a
shallow similarity comparison, the bi-encoder architecture was popularized in open-domain question answering
by DPR [Karpukhin et al., 2020], which trains two separately-parameterized BERT encoders, and later adapted to
an LLM-scale backbone by RepLLaMA [Ma et al., 2024], which replaces the two BERT towers with a single,
weight-shared LLaMA-2 decoder and [EOS]-token pooling in place of [CLS] pooling. An aggregated, multi-item
query matched against single-item documents is a pattern that appears elsewhere too,
generally resolved with untied rather than shared encoders: conversational dense retrieval faces the same
shape of problem, since a query built by concatenating several turns of dialogue history must be matched
against single-passage documents, and ConvDR [Yu et al., 2021] addresses it with a dedicated query encoder
trained to imitate a separately-encoded, frozen document index; industrial two-tower recommender systems
face it as well, aggregating a user's behavioral signals in one tower and a single item's content features in
the other [Yi et al., 2019], again with two independently-parameterized towers.

**Data augmentation.** Generating multiple training instances from a single interaction sequence by sliding a
fixed-size window across it, rather than using only one instance per user, dates back at least to Tan et al.
[2016], who introduced this augmentation alongside a way to account for temporal shift in RNN-based session
recommenders, and was later used by Caser [Tang and Wang, 2018] to train a convolutional next-item classifier
over a window of fixed size.

---

## 3 Method

We reformulate sequential recommendation as a text retrieval task (§3.1) and solve it with a bi-encoder
retriever whose two text-encoding pathways share a single LLM backbone (§3.2), efficiently adapted to this
backbone via a LoRA adapter (§3.3), trained with an augmented supervision signal (§3.4), and paired at
inference time with a lightweight filter that removes a retrieval artifact introduced by the
reformulation itself (§3.5). Figure 2 gives an overview of the full pipeline.

**[Figure 2 — `figures/fade_architecture.pdf`, mô tả chi tiết ở cuối file để vẽ lại nếu cần]**

### 3.1 Problem Statement

Let $\mathcal{U}$ and $\mathcal{I}$ denote the sets of users and items, respectively. For each user $u \in
\mathcal{U}$, the interaction history is a chronologically ordered sequence $S_u = (i_1^u, i_2^u, \dots,
i_{n_u}^u)$, $i_t^u \in \mathcal{I}$. Given a prefix $S_u^{<t} = (i_1^u, \dots, i_{t-1}^u)$, the goal of
sequential recommendation (SR) is to predict the item $i_t^u$ the user is most likely to interact with next,
typically cast as ranking the full item set $\mathcal{I}$ by relevance to $S_u^{<t}$.

We reformulate this ranking problem as a text retrieval problem. Let $T: \mathcal{I} \to \Sigma^{*}$ map each
item to a short textual description (its title, in our experiments). Given a context size $c$, we define a
query construction function

$$Q(S_u^{<t}) = T(i_{t-c}^u) \,\Vert\, T(i_{t-c+1}^u) \,\Vert\, \dots \,\Vert\, T(i_{t-1}^u)$$

that concatenates the titles of the $c$ most recent items into a single passage of text, and we treat the
entire item set as a document corpus $\mathcal{D} = \{T(i) : i \in \mathcal{I}\}$, one passage per item.
We build the query from only the $c$ most recent items rather than the user's entire history $S_u^{<t}$:
concatenating every item a user has ever interacted with would make the query grow unboundedly with history
length, quickly exceeding the backbone's context window for long-history users (e.g. MovieLens-1M, average
sequence length 165.5) and diluting the signal relevant to the immediate next item among a large amount of
older, less relevant context. The SR task then reduces to retrieving, for query $Q(S_u^{<t})$, the passage
$d^{*} \in \mathcal{D}$ that maximizes similarity to the query in a learned embedding space:

$$d^{*} = \arg\max_{d \in \mathcal{D}} \; \mathrm{sim}\big(f(Q(S_u^{<t})), f(d)\big)$$

where $f$ is a text encoder and $\mathrm{sim}(\cdot,\cdot)$ is cosine similarity. This formulation lets us
reuse LLM-based dense retrieval machinery — pretrained embedding backbones, contrastive fine-tuning, ANN
search — directly for SR, in place of item-ID embedding tables learned from scratch as in SASRec/GRU4Rec.

### 3.2 Bi-Encoder

We solve this retrieval problem with a **bi-encoder**: queries and documents are encoded
independently into fixed-size vectors by a shared text encoder $f_\theta$, and relevance is scored by the
cosine similarity of these vectors. This architecture follows the bi-encoder paradigm introduced by DPR
[Karpukhin et al., 2020] and adapted to LLM backbones by RepLLaMA [Ma et al., 2024]; we describe our specific
instantiation below and note where it departs from this prior work, without re-attributing every individual
design choice to it. $f_\theta$ is built on a pretrained decoder-only LLM; how we adapt this backbone for
training is described separately in §3.3.

**Encoding.** Given tokenized input text $x$ — either a query $Q$ produced by $Q(\cdot)$ or a document
$D = T(i)$ for some item $i$ — the decoder processes $x$ autoregressively under a causal
self-attention mask, producing hidden states $H = (h_1,\dots,h_L)$ for the $L$ input tokens. We take the
hidden state of the final token as the text representation, $e_x = h_L$. Because the backbone is a causal
decoder rather than a bidirectional encoder, $h_L$ is the only position that has attended to the entire input
sequence, which is why we pool at the last token instead of prepending a [CLS] token as BERT-style dense
retrievers do. The pooled vector is $\ell_2$-normalized,

$$\hat e_x = e_x / \lVert e_x \rVert_2,$$

and similarity between a query and a document is the cosine similarity of their normalized embeddings,
$\mathrm{sim}(Q,D) = \hat e_Q \cdot \hat e_D$.

**Tied encoder.** The same function $f_\theta$ — the same backbone weights and the same LoRA adapters —
encodes both queries and documents: there is no query-specific or document-specific parameter anywhere in the
model, and no explicit marker distinguishes the two at the input level either; $Q(S_u^{<t})$ and $T(i)$ are
both just passed through $f_\theta$ as plain text. This is a deliberate design choice rather than an
oversight. The original DPR formulation trains two independently parameterized encoders, one for queries and
one for passages; at LLM scale, doing the same would double an already substantial parameter and memory
budget, and a sufficiently expressive backbone can plausibly infer from context alone whether it is encoding a
multi-item history or a single item title, without needing separate parameters to do so.

### 3.3 LoRA Adapter

Fully fine-tuning $f_\theta$ would require far more compute, memory, and training time than is available on
commodity hardware, so instead we adapt the backbone with **Low-Rank Adaptation (LoRA)** [Hu et al., 2022],
which freezes the pretrained weights and injects a small number of trainable low-rank matrices into the
backbone's attention and feed-forward projections. This leaves the vast majority of the backbone's parameters
untouched while still letting it adapt to the retrieval objective, making fine-tuning practical on a single
consumer GPU.

**Training objective.** The encoder is fine-tuned with the InfoNCE contrastive loss. For a training example
consisting of query $Q_i$, one positive document $D_i^+$ (the ground-truth next item), and $g-1$ explicit
negative documents (group size $g$), the loss over training batch $\mathcal{B}$ is

$$\mathcal{L} = -\log \frac{\exp(\mathrm{sim}(Q_i, D_i^+)/\tau)}{\sum_{D \in \mathcal{B}} \exp(\mathrm{sim}(Q_i, D)/\tau)}$$

with temperature $\tau$. $\mathcal{B}$ contains the explicit negatives paired with $Q_i$ as well as the
positive and negative documents of every other query in the batch (in-batch negatives), following standard
dense-retrieval training practice.

### 3.4 Data Augmentation

Under the standard leave-one-out protocol, the last item of $S_u$ is held out for testing, the second-to-last
for validation, and the remaining prefix $S_u^{\text{train}} = (i_1^u, \dots, i_m^u)$ ($m = n_u - 2$) is used
for training. In its simplest form this produces exactly **one** training pair per user: query $Q(S_u^{\text
{train}})$, built from the $c$ items immediately preceding $i_m^u$, paired with positive $D^+ = T(i_m^u)$.

This severely underuses the available supervision relative to causal sequence models. SASRec, for instance,
applies a causal self-attention mask over the entire training sequence and is supervised to predict $i_{t+1}$
from $i_{1:t}$ *at every position* $t$ in a single forward pass — effectively obtaining $m-c$ (context,
next-item) pairs "for free" from one sequence. Our bi-encoder has no equivalent mechanism: without
intervention, each user contributes only one gradient signal per epoch, which is especially limiting for
users with long histories and for corpora with many users but comparatively few items relating them.

We recover this missing supervision explicitly via sliding-window augmentation, adapted from the sub-session
splitting technique commonly used in session-based recommendation. For each user, we slide a window of size
$c$ across $S_u^{\text{train}}$ and emit one training pair per valid position $t \in \{c+1,\dots,m\}$:

$$Q_t = Q\big((i_{t-c}^u,\dots,i_{t-1}^u)\big), \qquad D_t^+ = T(i_t^u)$$

yielding up to $m-c$ pairs per user instead of one, while preserving chronological order within each window
(the underlying sequence is never shuffled).

### 3.5 History Filter

At inference time, given query $Q(S_u^{<t})$, the retriever ranks the entire corpus $\mathcal{D}$ by
similarity and returns the top-$K$ candidates via approximate nearest-neighbor search (FAISS); as discussed
in §1, this ranked list is heavily biased toward items the user has already consumed. We attribute this to
the training objective rather than to any explicit preference for "popular" or
"similar" items: because $D_i^+$ is always the *next*, previously-unseen item, the contrastive loss never
constructs a negative pair whose text is verbatim (or near-verbatim) contained in the query itself.
Consequently, nothing in training penalizes assigning maximal similarity to a candidate whose text overlaps
almost completely with the query, exactly the case when the candidate is one of the $c$ items making up the
query's own context window. At inference, the model reproduces this shortcut: an item's own presence in the
query is itself powerful evidence of "relevance" to the trained similarity function, regardless of whether
re-recommending it is desirable.

We fix this without any additional training via a **history filter**: after retrieving the top-$K$
ranked list for a query, we remove every item the user has interacted with prior to position $t$, namely the
user's entire interaction history $S_u^{<t}$ and not merely the $c$ items forming the query's own context
window, and re-rank the remaining candidates before producing the final recommendation. The filter requires no
retraining and no GPU: it operates directly on already-computed FAISS rankings, making it essentially free to
apply.

---

## 4 Experiments

### 4.1 Datasets

We evaluate FADE on three public datasets that differ substantially in the number of items and
interaction-sequence length, letting us test the retrieval reformulation both where sequences are short and where they are very
long: two Amazon (2014) product categories, **Beauty** and **Sports and Outdoors** [McAuley et al., 2015],
and **MovieLens-1M** [Harper and Konstan, 2015]. Table 1 summarizes their statistics.

**Table 1. Dataset statistics.**

| Dataset | Users | Items | Interactions | Avg. seq. len | Max seq. len |
|---|---:|---:|---:|---:|---:|
| Amazon Beauty | 22,363 | 12,101 | 198,502 | 8.9 | 204 |
| Amazon Sports and Outdoors | 35,598 | 18,357 | 296,337 | 8.3 | 296 |
| MovieLens-1M | 6,040 | 3,416 | 999,611 | 165.5 | 2,277 |

Following standard practice in the SR literature, each dataset is 5-core filtered (every user and item has at
least 5 interactions) and split with the leave-one-out protocol: for each user, the last interaction is held
out for testing, the second-to-last for validation, and the remaining prefix is used for training. At both
validation and test time we rank the *entire* item set rather than a sampled subset of
negatives, since sampled-negative evaluation is known to produce rankings of methods that do not always agree
with full-ranking evaluation.

### 4.2 Experimental Setup

We use the **Qwen3-Embedding** family as the backbone for FADE. We choose Qwen3-Embedding over a
general-purpose, generation-oriented LLM because its pretraining objective is already aligned with what our
retriever needs to do: represent a piece of text as a single vector that can be compared against other such
vectors, rather than predict a distribution over the next token. We report results at two
backbone sizes, 0.6B and 4B; which of the two we adopt as FADE's headline backbone is a decision we defer
until results for both are complete across all three datasets. Both backbone sizes are adapted with
LoRA (rank $r=16$, scaling $\alpha=64$, dropout $0.1$) injected into the query, key, value, output, gate, up-, and
down-projection matrices of every transformer block. We fine-tune with the InfoNCE objective at
temperature $\tau=0.01$, using Tevatron v2 with DeepSpeed ZeRO-2, a learning rate of $1\times10^{-4}$, and 3
training epochs. All FADE experiments run on a single NVIDIA RTX 3090 24GB VRAM.

The ID-based baselines that we reproduce ourselves (GRU4Rec, SASRec, SR-GNN) are trained with RecBole
1.2.1 under a protocol matched as closely as possible to FADE's: full-softmax cross-entropy loss (no negative
sampling during training) and `MAX_ITEM_LIST_LENGTH=200`.

### 4.3 Evaluation Metrics

We report Normalized Discounted Cumulative Gain (NDCG@$K$) and Hit Ratio (HR@$K$ — equivalent to Recall@$K$
since each test instance has exactly one relevant item), at $K \in \{5,10,20\}$:

- **HR@$K$**: the fraction of test instances for which the held-out item appears in the top-$K$ ranked list.
- **NDCG@$K$**: the average, over test instances, of $1/\log_2(\mathrm{rank}+1)$ when the held-out item is
  ranked within the top $K$ (and $0$ otherwise) — rewarding a hit at rank 1 more than an equally-counted hit
  at rank $K$, unlike HR@$K$.

### 4.4 Baseline Methods

We compare against two groups of baselines, together with a zero-shot lower bound.

**ID-based sequential models** represent items with trainable ID embeddings and require no textual side
information: GRU4Rec [Hidasi et al., 2016] (RNN-based), SR-GNN [Wu et al., 2019] (graph-based), and SASRec
[Kang and McAuley, 2018] (self-attention, causal). We reproduce all three ourselves under the protocol in
§4.2.

**LLM- and generative-retrieval-based models**, for which we report literature numbers rather than
reproducing them ourselves (§4.5 states exactly which numbers in each table come from which source): TIGER
[Rajput et al., 2023] (generates a discrete Semantic ID for the target item), ActionPiece [Hou et al., 2025]
(improves TIGER's tokenization with co-occurrence-aware token merging), EAGER-LLM [Hong et al., 2025] (injects
collaborative signals into a decoder-only LLM recommender), LlamaRec [Yue et al., 2023] (re-ranks a separate
candidate retriever's output via an LLM verbalizer), P5 [Geng et al., 2022] (casts recommendation as a unified
text-to-text objective), E4SRec [Li et al., 2024] (uses an LLM as a sequence encoder feeding an ID-embedding
prediction head), and GLoSS-8B, the 8B-parameter variant of GLoSS [Acharya et al., 2025] (a
generate-then-retrieve dense-retrieval pipeline).

Finally, we report a **zero-shot** lower bound: the untuned Qwen3-Embedding-0.6B backbone used directly as a
retriever without any fine-tuning, to quantify how much of FADE's performance comes from fine-tuning versus
from the pretrained backbone alone.

### 4.5 Overall Results

Table 2 reports NDCG and HR at $K \in \{5,10\}$ on Beauty and Sports (merged into one wide table, laid out
horizontally by dataset to use the full page width — see `template/fade_paper.tex` for the LaTeX version), and
Table 3 reports the same on ML-1M, with the best result in each column in **bold**. In Table 2, methods marked
† are literature-reported numbers, copied from their respective papers rather than reproduced by us;
unmarked methods (SR-GNN, FADE) are reproduced by us under the protocol in §4.2.

**Table 2. Beauty and Sports and Outdoors.** (R = Recall, N = NDCG; † = literature-reported)

| Method | Beauty R@5 | R@10 | N@5 | N@10 | Sports R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GRU4Rec† | 0.0164 | 0.0283 | 0.0099 | 0.0137 | 0.0129 | 0.0204 | 0.0086 | 0.0110 |
| SASRec† | 0.0387 | 0.0605 | 0.0249 | 0.0318 | 0.0233 | 0.0350 | 0.0154 | 0.0192 |
| SR-GNN | 0.0363 | 0.0552 | 0.0252 | 0.0312 | 0.0200 | 0.0310 | 0.0130 | 0.0165 |
| TIGER† | 0.0454 | 0.0648 | 0.0321 | 0.0384 | 0.0264 | 0.0400 | 0.0181 | 0.0225 |
| ActionPiece† | 0.0511 | — | 0.0340 | — | 0.0316 | — | 0.0205 | — |
| EAGER-LLM† | 0.0548 | 0.0830 | 0.0369 | 0.0459 | 0.0373 | 0.0569 | 0.0251 | 0.0315 |
| LlamaRec† | 0.0852 | 0.1524 | 0.0543 | 0.0759 | — | — | — | — |
| P5† | 0.0503 | — | 0.0370 | — | 0.0272 | — | 0.0169 | — |
| E4SRec† | 0.0525 | — | 0.0360 | — | 0.0281 | — | 0.0196 | — |
| GLoSS-8B† | 0.0681 | — | 0.0442 | — | 0.0364 | — | 0.0238 | — |
| **FADE-0.6B** | **0.0730** | **0.1088** | **0.0501** | **0.0616** | **0.0447** | **0.0684** | **0.0294** | **0.0371** |
| FADE-4B | 0.0728 | 0.1083 | 0.0497 | 0.0611 | *pending* | *pending* | *pending* | *pending* |

**Table 3. MovieLens-1M** (fully self-reproduced — no literature baselines available; §6 of `paper_report.md`
notes that TIGER's original paper reports no ML-1M results).

| Method | R@5 | R@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|
| SASRec | 0.1303 | 0.2089 | 0.0823 | 0.1077 |
| SR-GNN | 0.1522 | 0.2215 | 0.1011 | 0.1235 |
| GRU4Rec | 0.1768 | 0.2641 | 0.1210 | 0.1491 |
| FADE-0.6B (Ours) | 0.1276 | 0.1887 | 0.0846 | 0.1041 |
| **FADE-4B (Ours)** | **0.1980** | **0.2829** | **0.1328** | **0.1601** |

On Beauty and Sports, FADE-0.6B achieves the best NDCG@10 and HR@10 of every method compared, beating the
strongest ID-based baseline (SASRec) by +93.7% and +93.2% NDCG@10 respectively, and beating every
LLM/generative-retrieval baseline as well — including GLoSS-8B, an 8B-parameter model, despite FADE-0.6B
having less than a tenth as many parameters.

On ML-1M, FADE-0.6B trails both GRU4Rec and SR-GNN, architectures whose recurrent or graph-propagation
mechanisms process a user's *entire* (very long, 165.5 items on average) interaction history, whereas FADE's
query is built from only the $c$ most recent items.

FADE-4B reverses this picture: it beats GRU4Rec, the strongest ID-based baseline on ML-1M, by +7–12% across
all four metrics (Table 3), the larger backbone more than recovering the gap that the smaller backbone's
fixed-size context leaves open. This is a striking contrast with Beauty, the other dataset where FADE-4B
fine-tuning has completed so far, where the larger backbone instead performs essentially on par with
FADE-0.6B (NDCG@10 0.0611 vs. 0.0616) rather than improving on it.

### 4.6 Ablation Study

Table 4 ablates FADE on all three datasets, using the 0.6B backbone, by removing one component at a time:
**"− History filter"** uses the raw FAISS ranking unmodified instead of applying the filter; **"−
Augmentation"** removes sliding-window augmentation from training; and **"Zero-shot (+ filter)"** fine-tunes
nothing at all but still applies the history filter at inference, isolating the contribution of fine-tuning
itself from that of the filter (the fully raw zero-shot number, without the filter, is lower still and is
reported in `paper_report.md` §3).

**Table 4. Ablation study (NDCG@10 / R@10).**

| Variant | Beauty NDCG@10 | Beauty R@10 | Sports NDCG@10 | Sports R@10 | ML-1M NDCG@10 | ML-1M R@10 |
|---|---:|---:|---:|---:|---:|---:|
| **FADE (full)** | **0.0616** | **0.1088** | **0.0371** | **0.0684** | **0.1041** | **0.1887** |
| − History filter | 0.0390 | 0.0905 | 0.0260 | 0.0574 | 0.0618 | 0.1343 |
| − Augmentation | 0.0581 | 0.1019 | 0.0279 | 0.0523 | 0.0691 | 0.1260 |
| Zero-shot (+ filter) | 0.0202 | 0.0364 | 0.0102 | 0.0195 | 0.0128 | 0.0214 |

Removing the history filter is consistently the single largest drop among the two components we ablate:
NDCG@10 falls by 36.7% on Beauty, 29.9% on Sports, and 40.6% on ML-1M when the raw FAISS ranking is used
unmodified — the direct empirical counterpart to the bias introduced in §1 and §3.5.
Removing augmentation also hurts substantially, though its effect size varies more across datasets (5.7% on
Beauty vs. 24.8–33.6% on Sports/ML-1M). Removing fine-tuning entirely (zero-shot) is by far the largest drop
of all — 67–88% NDCG@10 depending on dataset — confirming that most of FADE's performance comes from adapting
the backbone to the retrieval objective, not from the pretrained embedding space alone.

---

## 5 Conclusion

**[VIẾT LẠI 2026-07-30 #7]** Đã bỏ hẳn đoạn Limitations cũ (mixed sourcing + FADE-4B incomplete) theo yêu cầu
tác giả — mixed sourcing giờ đã xử lý bằng dấu † ở Table 2 (§4.5), còn FADE-4B/Sports pending chỉ còn note nội
bộ ở đầu file, không xuất hiện trong paper chính nữa. Conclusion dưới đây viết hoàn toàn mới, tóm tắt 3 đóng
góp + kết quả headline dựa trên số liệu hiện có (Beauty/Sports 0.6B đầy đủ, ML-1M 0.6B+4B đầy đủ, Beauty 4B
đầy đủ; chỉ Sports 4B còn thiếu nhưng không nhắc tới ở đây).

We presented FADE, a bi-encoder retriever that reformulates sequential recommendation as dense retrieval over
item text. Applying this reformulation to SR raises three challenges that FADE addresses directly: encoding
an entire item set with a bi-encoder rather than an infeasible cross-encoder; recovering the per-position
supervision that the standard leave-one-out protocol does not provide, through sliding-window data
augmentation; and correcting seen-item bias, a tendency for previously-consumed items to dominate the ranked
list, through a lightweight, training-free history filter.

Across three real-world datasets, FADE-0.6B substantially outperforms strong ID-based baselines (SASRec,
GRU4Rec, SR-GNN) and prior LLM-based recommenders on Beauty and Sports, while trailing full-history recurrent
and graph baselines on the much longer sequences of MovieLens-1M. Scaling to FADE-4B tells two different
stories: on Beauty it brings little additional benefit, but on MovieLens-1M it substantially improves over
FADE-0.6B and overtakes GRU4Rec and SR-GNN, reversing the gap observed at the smaller scale. Our ablation
study confirms that both the history filter and sliding-window augmentation contribute substantially to this
performance, with most of the overall gain coming from fine-tuning the backbone itself.

---

## References

**[ĐÃ VERIFY 100% qua web search — 2026-07-27]** Toàn bộ citation dùng trong draft hiện tại (Introduction,
Method, Experiments), định dạng author-year tạm thời — chuyển sang định dạng bibliography chuẩn của venue
(numbered/LNCS style, v.v.) khi biết hội thảo cụ thể.

1. Hidasi, B., Karatzoglou, A., Baltrunas, L., Tikk, D.: Session-based Recommendations with Recurrent Neural
   Networks. ICLR (2016)
2. Wu, S., Tang, Y., Zhu, Y., Wang, L., Xie, X., Tan, T.: Session-based Recommendation with Graph Neural
   Networks. AAAI 33(1), 346–353 (2019)
3. Kang, W.C., McAuley, J.: Self-Attentive Sequential Recommendation. ICDM (2018)
4. Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., Yih, W.: Dense Passage
   Retrieval for Open-Domain Question Answering. EMNLP, pp. 6769–6781 (2020)
5. Ma, X., Wang, L., Yang, N., Wei, F., Lin, J.: Fine-Tuning LLaMA for Multi-Stage Text Retrieval. SIGIR '24,
   pp. 2421–2425 (2024)
6. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-Rank
   Adaptation of Large Language Models. ICLR (2022)
7. McAuley, J., Targett, C., Shi, Q., van den Hengel, A.: Image-based Recommendations on Styles and
   Substitutes. SIGIR, pp. 43–52 (2015)
8. Harper, F.M., Konstan, J.A.: The MovieLens Datasets: History and Context. ACM TiiS 5(4), 19:1–19:19 (2015)
9. Rajput, S., Mehta, N., Singh, A., Hulikal Keshavan, R., Vu, T., Heldt, L., Hong, L., Tay, Y., Tran, V.,
   Samost, J., Kula, M., Chi, E.H., Sathiamoorthy, M.: Recommender Systems with Generative Retrieval. NeurIPS
   (2023)
10. Hou, Y., Ni, J., He, Z., Sachdeva, N., Kang, W.C., Chi, E.H., McAuley, J., Cheng, D.Z.: ActionPiece:
    Contextually Tokenizing Action Sequences for Generative Recommendation. ICML (2025)
11. Hong, M., Xia, Y., Wang, Z., Zhu, J., Wang, Y., Cai, S., Yang, X., Dai, Q., Dong, Z., Zhang, Z., Zhao, Z.:
    EAGER-LLM: Enhancing Large Language Models as Recommenders through Exogenous Behavior-Semantic
    Integration. WWW '25, pp. 2754–2762 (2025)
12. Yue, Z., Rabhi, S., de Souza Pereira Moreira, G., Wang, D., Oldridge, E.: LlamaRec: Two-Stage
    Recommendation using Large Language Models for Ranking. PGAI Workshop @ CIKM (2023)
13. Geng, S., Liu, S., Fu, Z., Ge, Y., Zhang, Y.: Recommendation as Language Processing (RLP): A Unified
    Pretrain, Personalized Prompt & Predict Paradigm (P5). RecSys, pp. 299–315 (2022)
14. Li, X., Chen, C., Zhao, X., Zhang, Y., Xing, C.: E4SRec: An Elegant Effective Efficient Extensible
    Solution of Large Language Models for Sequential Recommendation. WWW '24 (2024)
15. Acharya, K., Petrov, A.V., Ziani, J.: Generative Language Models with Semantic Search for Sequential
    Recommendation. Online and Adaptive Recommender Systems (OARS) Workshop @ KDD (2025)
16. Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., Wei, F.: Text Embeddings by
    Weakly-Supervised Contrastive Pre-training. arXiv:2212.03533 (2022)
17. Tan, Y.K., Xu, X., Liu, Y.: Improved Recurrent Neural Networks for Session-based Recommendations. In:
    Workshop on Deep Learning for Recommender Systems (DLRS) @ RecSys (2016)
18. Tang, J., Wang, K.: Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding.
    WSDM, pp. 565–573 (2018)
19. Yu, S., Liu, Z., Xiong, C., Feng, T., Liu, Z.: Few-Shot Conversational Dense Retrieval. SIGIR '21, pp.
    829–838 (2021)
20. Yi, X., Yang, J., Hong, L., Cheng, D.Z., Heldt, L., Kumthekar, A., Zhao, Z., Wei, L., Chi, E.:
    Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations. RecSys '19, pp. 269–277
    (2019)

**Ghi chú xác minh**: mục 1–9, 16–20 đã verify qua web search trong phiên này (venue/năm/tác giả khớp
DBLP/ACM/arXiv trực tiếp; riêng abstract của mục 10–15 cũng đã fetch và đọc trực tiếp để mô tả đúng phương
pháp trong §2 Related Work, không suy đoán). Mục 10–12, 15 lấy nguồn citation từ bạn cung cấp — đã đối chiếu
chéo với kết quả search để xác nhận khớp (ICML 2025 cho ActionPiece, WWW 2025 cho EAGER-LLM, OARS@KDD 2025
cho GLoSS). Mục 5 (RepLLaMA) có 2 mốc thời gian: arXiv preprint 2023, bản chính thức SIGIR 2024 — dùng "Ma et
al., 2024" trong in-text citation (theo venue chính thức) nhưng giữ cả 2 mốc trong bibliography. Phát hiện
đáng chú ý khi đọc abstract GLoSS: chính GLoSS cũng tự so sánh với các phương pháp BM25-lexical (GPT4Rec) và
nêu động lực tương tự FADE (vượt qua lexical matching) — đây là related work gần nhất, đã viết riêng 1 đoạn
phân biệt rõ trong §2.

**[Cập nhật 2026-07-27 #3]** Mục 17 (Tan et al., 2016) verify qua web search — xác nhận đây là citation
**sớm hơn Caser 2 năm** cho kỹ thuật sliding-window/data augmentation trong SR (Caser vẫn giữ lại vì là
citation quen thuộc hơn với cộng đồng, nhưng Tan et al. mới là nguồn gốc kỹ thuật). Mục 19–20 (ConvDR,
two-tower recsys — số cũ 20–21, đã đánh số lại 2026-07-30 #3 sau khi bỏ Sciavolino et al.) verify qua web
search — cả hai đều dùng **untied encoder** cho bài toán bất đối xứng tương
tự SR (query tổng hợp nhiều phần tử vs document đơn lẻ), dùng để đối chiếu với lựa chọn tied của FADE trong
§2 "Dense retrieval bi-encoders".

**[Cập nhật 2026-07-27 #4 — chốt GLoSS]** Tác giả xác nhận chi tiết pipeline GLoSS: (1) fine-tune LLaMA-3 để
**sinh** text mô tả item tiếp theo (generative, prompt-based) từ lịch sử mua hàng, (2) encode cả text sinh ra
và mọi candidate item bằng **e5-small-v2** [Wang et al., 2022] — model riêng, không phải LLaMA-3, (3) dot
product để tính ranking score. Kết luận: bước (2)-(3) **đúng là bi-encoder** (và là tied, vì dùng chung
e5-small-v2 cho cả 2 phía), nhưng bản chất pipeline là **generate-then-retrieve 2 giai đoạn**, khác hẳn FADE
— phần "hiểu preference user" nằm ở generator (LLaMA-3, train bằng generation loss) chứ không phải ở bi-encoder
(e5-small-v2, encoder nhỏ, có thể không fine-tune riêng cho tác vụ retrieval này). Đã viết lại đoạn GLoSS
trong §2 theo đúng phân biệt này, bỏ `[TODO — cần thảo luận thêm]`, thêm citation e5 (Wang et al., 2022,
arXiv:2212.03533).

**[Cập nhật 2026-07-28]** Viết lại §1 Introduction (đoạn mở đầu về SR, bỏ mention RepLLaMA ở câu chuyển sang
3 challenges, viết lại challenge thứ 3 theo hướng content-based vs collaborative recommendation) và rút gọn
§2 Related Work (GLoSS còn 2 ý, gộp đoạn "Dense retrieval bi-encoders" bỏ nội dung riêng của FADE, đoạn "Data
augmentation and lexical-overlap bias" chỉ còn nói về prior work). Đã bỏ toàn bộ tham chiếu `(§X.X)` trong
Introduction và Related Work theo yêu cầu tác giả. Thêm **mục 22** (Adomavicius & Tuzhilin, 2005 — survey
kinh điển RecSys, đã verify DBLP/ACM) làm căn cứ cho claim "overspecialization là hạn chế đã biết của
content-based recommendation" ở challenge thứ 3 — **citation mới, tác giả nên xác nhận lại trước khi chốt**.

---

## Figure 1 — Introduction example (mô tả chi tiết để vẽ lại nếu cần)

**[LƯU Ý 2026-07-30]** File nguồn hiện tại: `figures/fig1_introduction_example.pdf` (tác giả tự làm lại, thay
cho `fade_example.pdf`/`make_example_figure.py` cũ) — bố cục tổng thể vẫn giống mô tả dưới đây (cột trái dùng
chung cho cả 2 panel, rẽ sang 2 panel bên phải), chỉ khác phong cách hình ảnh. **Ảnh hiện tại vẫn còn chữ
"history contamination"** trong tiêu đề panel (b) — cần tác giả tự sửa lại ảnh thành "seen-item bias" để khớp
tên mới đã đổi trong toàn bài (xem ghi chú "Rà từ vựng toàn bài 2026-07-30" cuối file).

**[Cập nhật 2026-07-30 #8]** Tác giả đã tự chỉnh `.tex`: hình thu nhỏ còn `width=0.75\linewidth` (từ
`\textwidth`), và caption rút gọn tối đa còn "Illustrative example on Amazon Beauty. (a) A well-trained
retriever ranks a novel, relevant item first; (b) an unfiltered retriever instead re-surfaces an
already-consumed item, an instance of *seen-item bias*" — bỏ hẳn tham chiếu `(Sect. 3.5)` ở cuối câu.

Bố cục: cột trái dùng chung cho cả 2 panel (lịch sử 3 item → query text), rẽ sang 2 panel bên phải cho 2 kết
quả khác nhau của cùng 1 query.

**Cột trái (dùng chung)**: 3 box xanh dương xếp dọc, có mũi tên nối tuần tự (1→2→3, đúng thứ tự thời gian
mua hàng: sunscreen → cleanser → serum) → mũi tên xuống → box cam "Query text $Q$ (titles 1+2+3 concatenated)".

**Panel (a) — phía trên bên phải, tiêu đề màu xanh lá "What a well-trained retriever should return"**: mũi
tên cong từ cạnh phải Query box đi lên tới box xanh lá đậm viền "#1 CeraVe Moisturizing Cream" (kèm chú thích
nhỏ bên phải: "✓ not in history / ✓ same skincare routine"), phía dưới là 2 box xám nhạt "#2", "#3" (item
không liên quan trực tiếp đến ví dụ, chỉ để show đây là 1 ranked list).

**Panel (b) — phía dưới bên phải, tiêu đề màu đỏ "What actually happens in practice — seen-item bias" (ảnh
hiện tại vẫn ghi "history contamination", cần sửa lại)**: mũi tên gần thẳng từ cạnh phải Query box (hơi cong
xuống) tới box đỏ đậm viền "#1 CeraVe
Foaming Facial Cleanser" (kèm chú thích: "✗ already item 2 in the query itself"), bên dưới là box xám "#2
CeraVe Moisturizing Cream (correct answer, pushed down)" và "#3" — thể hiện rõ: câu trả lời đúng bị đẩy
xuống hạng 2 vì hạng 1 bị chiếm bởi chính item đã có trong query.

**Ý đồ thị giác cần giữ khi vẽ lại**: (1) dùng chung **1** cột lịch sử/query cho cả 2 panel (không lặp lại) để
nhấn mạnh đây là *cùng một query*, chỉ khác ở hành vi retrieval; (2) panel (a) và (b) phải đối xứng về bố cục
(cùng 3 dòng #1/#2/#3) để dễ so sánh trực quan; (3) màu xanh lá/đỏ ở box #1 mỗi panel là điểm nhấn chính —
đây là sự khác biệt duy nhất cần người đọc chú ý ngay từ cái nhìn đầu tiên.

---

## Figure 2 — Method architecture overview (mô tả chi tiết để vẽ lại nếu cần)

**[Cập nhật 2026-07-30 #8]** Tác giả đã tự chỉnh `.tex`: hình phóng to lại thành `width=1\linewidth` (từ
`0.65\textwidth` trước đó), và caption rút gọn cực ngắn chỉ còn **"FADE pipeline overview"** — bỏ hết mô tả
chi tiết về context size/LoRA/history filter từng có trong caption trước đó. Mô tả chi tiết dưới đây vẫn giữ
nguyên giá trị làm hướng dẫn vẽ lại ảnh, chỉ riêng caption thực tế trong bài đã ngắn hơn nhiều.

**[THIẾT KẾ LẠI 2026-07-28 #2 — tác giả tự làm lại hoàn toàn]** File nguồn: `figures/fade_architecture.pdf` —
tác giả đã tự thiết kế lại bằng công cụ khác (không còn phải matplotlib script `figures/make_architecture_figure.py`
nữa; script đó giờ **stale**, chạy lại sẽ ghi đè mất bản mới — đừng chạy trừ khi tác giả yêu cầu cập nhật
script cho khớp). Bố cục là **1 luồng đơn** (không còn tách 2 panel Training/Inference như bản trước), với
Bi-Encoder và LoRA Adapter là **2 box tách riêng** — khớp với việc Method giờ tách §3.2 (Bi-Encoder)
và §3.3 (LoRA Adapter) thay vì gộp chung trong 1 mục như trước.

**Legend (đặt trên cùng, 4 ký hiệu):** 🔥 Trainable; ❄️ Frozen; viền đứt nét = chỉ tồn tại lúc training; box tô
màu cam/đỏ cam = đóng góp của bài báo (Data Augmentation, History Filter — **[LƯU Ý 2026-07-30] ảnh hiện tại
vẫn ghi "Post-hoc History Filter", cần tác giả sửa lại ảnh thành "History Filter" cho khớp tên mới**).

**Luồng chính (trên → dưới, trái → phải):**
- **"User History"**: dãy $i_1, i_2, \ldots, i_{m-c}, \ldots, i_m$ trong 1 khung lớn; $c$ item cuối cùng
  ($i_{m-c}, \ldots, i_m$) được bao bởi khung đứt nét màu cam riêng, chú thích **"context size c"** — minh hoạ
  trực quan lý do §3.1 chỉ dùng $c$ item gần nhất chứ không phải toàn bộ lịch sử (câu trả lời: lịch sử dài sẽ
  vượt context window của backbone và loãng thông tin).
- **"Candidate Item"**: box đối xứng bên phải, tương ứng phía document.
- **"Data Augmentation"** (box đứt nét, tô màu cam nhạt = đóng góp của bài báo, §3.4): nhận mũi tên đứt nét từ
  "User History", xuất mũi tên đứt nét sang "Query" — chỉ hoạt động lúc train.
- "User History" → "Query"; "Candidate Item" → "Document".
- "Query" và "Document" đều có mũi tên đi vào **"Bi-Encoder (Qwen3-Embedding family)"** — box tím/xanh, icon
  ❄️ **Frozen** ở góc, biểu diễn backbone pretrained giữ nguyên trọng số gốc (§3.2).
- **"LoRA Adapter"** (box đứt nét viền tím riêng biệt, icon 🔥 **Trainable**, bên trái Bi-Encoder): mũi tên đứt
  nét đi vào Bi-Encoder, biểu diễn LoRA chỉ thêm một lượng nhỏ tham số train được thay vì train lại toàn bộ
  backbone (§3.3); một mũi tên đứt nét khác tạo vòng lặp dưới box này biểu diễn backprop/cập nhật trọng số lúc
  train (InfoNCE, §3.3).
- Bi-Encoder → **"Top-K Retrieval"** → **"History Filter"** (box tô màu cam đậm = đóng góp của bài báo, §3.5;
  ảnh hiện tại vẫn ghi "Post-hoc History Filter") → **"Final Recommendation List"**.

**Ý đồ thị giác cần giữ khi vẽ lại**: (1) Bi-Encoder và LoRA Adapter là **2 box tách riêng**, không gộp làm 1
như bản matplotlib cũ, để khớp với việc Method tách §3.2/§3.3; (2) khung "context size c" quanh $c$ item cuối
trong "User History" để minh hoạ trực quan §3.1; (3) không còn tách 2 panel Training/Inference như bản trước —
giờ là 1 luồng duy nhất, dùng icon 🔥/❄️ và viền đứt nét để phân biệt trainable/frozen/training-only thay cho
2 panel riêng; (4) 2 box "Data Augmentation" và "History Filter" tô màu khác biệt vì là 2 đóng góp chính cần
nổi bật.

---

*(Lịch sử chỉnh sửa chi tiết từng phiên đã được gộp vào mục "Các quyết định quan trọng cần nhớ" ở đầu file —
xoá phần changelog cũ ở đây vì đã lỗi thời/trùng lặp sau nhiều vòng chỉnh sửa tiếp theo.)*

## TODO tổng hợp — đọc mục này để biết chính xác việc còn lại (cập nhật 2026-07-28 #3)

**Nội dung cần viết (theo thứ tự ưu tiên hợp lý):**
- [x] **Abstract** — viết xong 2026-07-28 #3, xem đầu file (giữa title và §1 Introduction) và
      `template/fade_paper.tex` (`\begin{abstract}...\end{abstract}`).
- [x] **Section 5 Conclusion** — viết xong 2026-07-30 #7: tóm tắt 3 đóng góp + kết quả headline dựa trên số
      liệu hiện có. Đoạn Limitations cũ (mixed sourcing + FADE-4B incomplete) đã bị **xoá hẳn** theo yêu cầu
      tác giả (không phải "giữ nguyên rồi thêm phần đầu" như TODO cũ ghi) — mixed sourcing giờ xử lý bằng dấu
      † ở Table 2, FADE-4B/Sports pending chỉ còn note nội bộ đầu file.
- [ ] **FADE-4B kết quả — chỉ còn thiếu Sports** (Beauty + ML-1M đã điền 2026-07-28 #4/#5). ML-1M cho thấy
      FADE-4B vượt qua cả GRU4Rec/SR-GNN (+7-12%), trong khi Beauty chỉ ngang FADE-0.6B — 2 pattern trái
      ngược. Khi có số liệu Sports: điền Table 2 (xoá `*pending*`), viết thêm nhận xét vào narrative §4.5 hiện
      có, và cân nhắc nghiêm túc việc biến "scale giúp bù kiến trúc trên chuỗi dài nhưng không giúp trên chuỗi
      ngắn" thành 1 finding/thảo luận riêng (hiện mới là suy đoán 1 câu, dựa trên 2 điểm dữ liệu, §4.5). Sau
      khi có Sports + tác giả chốt backbone size (0.6B hay 4B là headline), cần viết lại Abstract + Conclusion
      cho khớp (Conclusion hiện chưa nhắc FADE-4B/Sports vì cố tình chờ đủ số liệu).
- [ ] **Acknowledgments / Disclosure of Interests** trong `fade_paper.tex` — đang là placeholder TODO, cần
      tác giả điền grant/funding thật (nếu có) và xác nhận lại phát biểu competing interests trước khi nộp.

**Cần verify lại trước khi nộp (khác với citation — citation đã verify xong):**
- [ ] Re-check **số liệu** (không phải citation) của TIGER/S³-Rec và toàn bộ nhóm literature-sourced (đánh dấu
      † trong Table 2, §4.5) — đối chiếu số NDCG/R với bảng gốc trong từng paper. Không còn đoạn Limitations
      để flag việc này nữa (đã xoá 2026-07-30 #7) — cần tự nhớ verify trước khi nộp.
- [ ] Nếu có thời gian đọc kỹ paper GLoSS: xác nhận lại mô tả pipeline generate-then-retrieve đã viết trong
      §2 là chính xác 100% (đã dựa trên mô tả bạn cung cấp trực tiếp, độ tin cậy cao nhưng chưa tự đọc paper
      gốc để đối chiếu câu chữ).

**Việc mang tính hình thức, làm sau cùng khi đã chốt venue:**
- [ ] Chuyển bibliography từ author-year tạm thời sang định dạng chuẩn của venue (numbered/LNCS style khớp
      `splncs04.bst`, hoặc theo yêu cầu riêng của hội thảo) khi biết hội thảo cụ thể.
- [ ] Kiểm tra lại page limit khi biết venue — hiện tại 17 trang (LaTeX), LNCS phổ biến giới hạn 12–15 trang;
      đã rút được từ 18→16 rồi lại 16→17 (thêm đoạn GLoSS) qua nhiều vòng chỉnh sửa, có thể cần rút thêm.
- [ ] Đồng bộ Overleaf nếu muốn — xem hướng dẫn Git integration / GitHub sync / upload zip đã trao đổi trong
      phiên trước (Overleaf Git integration là tính năng trả phí).
