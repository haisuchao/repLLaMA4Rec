"""Generates paper/figures/fade_architecture.pdf — FADE pipeline overview diagram.

Split into two phases, stacked top-to-bottom:
  (a) Training  — bi-encoder (Sect. 3.2) + sliding-window data augmentation (Sect. 3.3)
  (b) Inference — the fine-tuned bi-encoder (frozen) + post-hoc history filter (Sect. 3.4),
                   which only ever runs at inference time and has no effect on training.
A single connector arrow between the two panels shows that the only thing inference reuses
from training is the fine-tuned encoder weights.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(15, 11.6))
ax.set_xlim(0, 15.6)
ax.set_ylim(-0.3, 13.6)
ax.axis("off")

COL_DATA = "#dbe7f5"     # light blue — raw data / text
COL_DATA_EDGE = "#4472a8"
COL_ENC = "#fde3c0"      # amber — shared encoder (highlight)
COL_ENC_EDGE = "#c9781b"
COL_OUT = "#dcefdc"      # light green — retrieval / output
COL_OUT_EDGE = "#4f8f52"
COL_TRAIN = "#eeeeee"    # gray — training-only annotation
COL_TRAIN_EDGE = "#888888"
COL_FILT = "#f6d9d9"     # light red — filter (contribution highlight)
COL_FILT_EDGE = "#b23b3b"
COL_PANEL = "#f7f7f7"

def box(x, y, w, h, text, fc, ec, ls="-", lw=1.6, fs=10.5, weight="normal"):
    b = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=lw, linestyle=ls,
                        edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fs, weight=weight, zorder=4, linespacing=1.35)
    return (x, y, w, h)

def arrow(b1, b2, side1="right", side2="left", style="-|>", color="black",
          lw=1.4, ls="-", connectionstyle="arc3,rad=0.0", zorder=2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    pts1 = {"right": (x1 + w1, y1 + h1 / 2), "left": (x1, y1 + h1 / 2),
            "top": (x1 + w1 / 2, y1 + h1), "bottom": (x1 + w1 / 2, y1)}
    pts2 = {"right": (x2 + w2, y2 + h2 / 2), "left": (x2, y2 + h2 / 2),
            "top": (x2 + w2 / 2, y2 + h2), "bottom": (x2 + w2 / 2, y2)}
    p1, p2 = pts1[side1], pts2[side2]
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                         linewidth=lw, linestyle=ls, color=color, zorder=zorder,
                         connectionstyle=connectionstyle)
    ax.add_patch(a)

# ============================================================================
# Panel backgrounds
# ============================================================================
panel_a = FancyBboxPatch((0.1, 6.2), 15.4, 7.05, boxstyle="round,pad=0.02,rounding_size=0.08",
                          linewidth=0, facecolor=COL_PANEL, zorder=0)
ax.add_patch(panel_a)
panel_b = FancyBboxPatch((0.1, 0.05), 15.4, 5.85, boxstyle="round,pad=0.02,rounding_size=0.08",
                          linewidth=0, facecolor="#ffffff", zorder=0)
ax.add_patch(panel_b)

ax.text(0.35, 12.95, "(a) Training", fontsize=13, weight="bold", ha="left", va="center")
ax.text(2.55, 12.95, "— bi-encoder + sliding-window augmentation (Sect. 3.2, 3.3)",
        fontsize=10, style="italic", color="#444444", ha="left", va="center")
ax.text(0.35, 5.75, "(b) Inference", fontsize=13, weight="bold", ha="left", va="center")
ax.text(2.65, 5.75, "— frozen encoder + post-hoc history filter (Sect. 3.4)",
        fontsize=10, style="italic", color="#444444", ha="left", va="center")

# ============================================================================
# (a) TRAINING — top panel
# ============================================================================
T = 7.35  # y-offset for training panel

b_hist_tr = box(0.4, T + 3.35, 2.3, 0.9, "User history (train)\n$S_u^{\\mathrm{train}}$",
                 COL_DATA, COL_DATA_EDGE)
b_aug = box(0.4, T + 1.9, 2.3, 0.95, "Sliding-window\naugmentation\n(Sect. 3.3)",
            COL_TRAIN, COL_TRAIN_EDGE, ls="--")
b_qtext_tr = box(3.35, T + 3.35, 2.55, 0.9, "Query texts\n$Q_t$ (augmented)",
                  COL_DATA, COL_DATA_EDGE)
arrow(b_hist_tr, b_qtext_tr)
arrow(b_hist_tr, b_aug, side1="bottom", side2="top")
arrow(b_aug, b_qtext_tr, side1="right", side2="left",
      connectionstyle="arc3,rad=-0.25", ls="--", color="#666666")

b_cat_tr = box(0.4, T + 0.1, 2.3, 0.9, "Item catalog\n$\\mathcal{I}$", COL_DATA, COL_DATA_EDGE)
b_dtext_tr = box(3.35, T + 0.1, 2.55, 0.9, "Document texts\n$T(i),\\ i\\in\\mathcal{I}$",
                  COL_DATA, COL_DATA_EDGE)
arrow(b_cat_tr, b_dtext_tr)

b_enc_tr = box(7.1, T + 0.75, 3.1, 2.6,
               "Bi-Encoder $f_\\theta$\nQwen3-Embedding + LoRA\n(SHARED weights, Sect. 3.2)",
               COL_ENC, COL_ENC_EDGE, lw=2.4, fs=11.5, weight="bold")
arrow(b_qtext_tr, b_enc_tr, side1="right", side2="top", connectionstyle="arc3,rad=-0.15")
arrow(b_dtext_tr, b_enc_tr, side1="right", side2="bottom", connectionstyle="arc3,rad=0.15")

b_loss = box(7.4, T - 1.05, 2.5, 0.8, "InfoNCE loss\n(train-time only)",
             COL_TRAIN, COL_TRAIN_EDGE, ls="--", fs=9.5)
arrow(b_loss, b_enc_tr, side1="top", side2="bottom", ls="--", color="#666666")

b_ftenc = box(11.35, T + 1.35, 3.55, 1.4,
              "Fine-tuned weights $\\theta^{*}$\ncarried to inference\n(frozen, no further training)",
              "#ffffff", COL_ENC_EDGE, ls=":", lw=1.6, fs=9.8)
arrow(b_enc_tr, b_ftenc, side1="right", side2="left", color=COL_ENC_EDGE, lw=1.8)

# ============================================================================
# (b) INFERENCE — bottom panel
# ============================================================================
b_hist_inf = box(0.4, 4.6, 2.3, 0.9, "User history (test)\n$S_u^{<t}$", COL_DATA, COL_DATA_EDGE)
b_qtext_inf = box(3.35, 4.6, 2.55, 0.9, "Query text\n$Q(S_u^{<t})$", COL_DATA, COL_DATA_EDGE)
arrow(b_hist_inf, b_qtext_inf)

b_cat_inf = box(0.4, 0.55, 2.3, 0.9, "Item catalog\n$\\mathcal{I}$", COL_DATA, COL_DATA_EDGE)
b_dtext_inf = box(3.35, 0.55, 2.55, 0.9, "Document texts\n$T(i),\\ i\\in\\mathcal{I}$",
                   COL_DATA, COL_DATA_EDGE)
arrow(b_cat_inf, b_dtext_inf)

b_enc_inf = box(7.1, 2.15, 3.1, 2.0,
                "Bi-Encoder $f_{\\theta^{*}}$\n(frozen weights\nfrom training)",
                COL_ENC, COL_ENC_EDGE, lw=2.4, fs=11.5, weight="bold")
arrow(b_qtext_inf, b_enc_inf, side1="right", side2="left", connectionstyle="arc3,rad=-0.2")
arrow(b_dtext_inf, b_enc_inf, side1="right", side2="bottom", connectionstyle="arc3,rad=0.15")

arrow(b_ftenc, b_enc_inf, side1="bottom", side2="top", color=COL_ENC_EDGE, lw=1.8, ls=":",
      connectionstyle="arc3,rad=0.05")

b_qemb = box(10.9, 4.6, 2.2, 0.9, "Query embedding\n$\\hat{e}_Q$", COL_DATA, COL_DATA_EDGE)
b_demb = box(10.9, 0.55, 2.55, 0.9, "Corpus embeddings $\\{\\hat{e}_D\\}$\n(precomputed, FAISS index)",
             COL_DATA, COL_DATA_EDGE)
arrow(b_enc_inf, b_qemb, side1="right", side2="left", connectionstyle="arc3,rad=-0.2")
arrow(b_enc_inf, b_demb, side1="bottom", side2="left", connectionstyle="arc3,rad=-0.15")

b_topk = box(13.55, 2.65, 2.0, 0.9, "Top-$K$\nretrieval", COL_OUT, COL_OUT_EDGE, fs=9.8)
arrow(b_qemb, b_topk, side1="bottom", side2="top", connectionstyle="arc3,rad=-0.2")
arrow(b_demb, b_topk, side1="top", side2="left", connectionstyle="arc3,rad=0.2")

b_filt = box(13.55, 1.4, 2.0, 0.9, "History\nfilter", COL_FILT, COL_FILT_EDGE, lw=2.0, fs=9.8)
arrow(b_topk, b_filt, side1="bottom", side2="top")

b_final = box(13.55, 0.15, 2.0, 0.9, "Final\nTop-$K$ list", COL_OUT, COL_OUT_EDGE, fs=9.8)
arrow(b_filt, b_final, side1="bottom", side2="top")

# ============================================================================
# Legend
# ============================================================================
legend_items = [
    (COL_DATA, COL_DATA_EDGE, "-", "Text / embedding"),
    (COL_ENC, COL_ENC_EDGE, "-", "Shared bi-encoder (Sect. 3.2)"),
    (COL_TRAIN, COL_TRAIN_EDGE, "--", "Training-time only (Sect. 3.2, 3.3)"),
    (COL_FILT, COL_FILT_EDGE, "-", "Post-hoc filter, inference-only (Sect. 3.4)"),
]
ly = 13.3
for i, (fc, ec, ls, label) in enumerate(legend_items):
    lx = 0.4 + i * 3.85
    ax.add_patch(FancyBboxPatch((lx, ly), 0.32, 0.26,
                                 boxstyle="round,pad=0.01,rounding_size=0.03",
                                 linewidth=1.4, linestyle=ls,
                                 edgecolor=ec, facecolor=fc, zorder=4))
    ax.text(lx + 0.44, ly + 0.13, label, ha="left", va="center", fontsize=8.6, zorder=4)

plt.tight_layout()
plt.savefig("/media/administrator/Data1/Projects/Python/repLLaMA/paper/figures/fade_architecture.pdf",
            bbox_inches="tight")
plt.savefig("/media/administrator/Data1/Projects/Python/repLLaMA/paper/figures/fade_architecture_preview.png",
            bbox_inches="tight", dpi=150)
print("saved fade_architecture.pdf")
