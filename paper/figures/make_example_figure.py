"""Generates paper/figures/fade_example.pdf — Introduction illustrative example
(query construction + the two possible retrieval outcomes: desired vs. contaminated)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13.5, 7.2))
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 7.2)
ax.axis("off")

COL_HIST = "#dbe7f5"; COL_HIST_EDGE = "#4472a8"
COL_QUERY = "#fde3c0"; COL_QUERY_EDGE = "#c9781b"
COL_GOOD = "#dcefdc"; COL_GOOD_EDGE = "#4f8f52"
COL_BAD = "#f6d9d9"; COL_BAD_EDGE = "#b23b3b"
COL_NEUTRAL = "#f2f2f2"; COL_NEUTRAL_EDGE = "#999999"

def box(x, y, w, h, text, fc, ec, lw=1.6, fs=9.5, weight="normal", ha="center"):
    b = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.02,rounding_size=0.07",
                        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(b)
    tx = x + w / 2 if ha == "center" else x + 0.12
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs,
             weight=weight, zorder=3, linespacing=1.3)
    return (x, y, w, h)

def arrow(b1, b2, side1="right", side2="left", color="black", lw=1.5,
          connectionstyle="arc3,rad=0.0", style="-|>"):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    pts1 = {"right": (x1 + w1, y1 + h1 / 2), "left": (x1, y1 + h1 / 2),
            "top": (x1 + w1 / 2, y1 + h1), "bottom": (x1 + w1 / 2, y1)}
    pts2 = {"right": (x2 + w2, y2 + h2 / 2), "left": (x2, y2 + h2 / 2),
            "top": (x2 + w2 / 2, y2 + h2), "bottom": (x2 + w2 / 2, y2)}
    a = FancyArrowPatch(pts1[side1], pts2[side2], arrowstyle=style,
                         mutation_scale=13, linewidth=lw, color=color,
                         zorder=1, connectionstyle=connectionstyle)
    ax.add_patch(a)

# ---- Shared left column: user history -> query ---------------------------
ax.text(0.15, 6.85, "User's most recent purchases (chronological)",
        fontsize=10, weight="bold")
b_h1 = box(0.15, 6.0, 3.15, 0.7, "1. Neutrogena Ultra Sheer\nDry-Touch Sunscreen SPF 45",
           COL_HIST, COL_HIST_EDGE, fs=8.6)
b_h2 = box(0.15, 5.05, 3.15, 0.7, "2. CeraVe Foaming\nFacial Cleanser",
           COL_HIST, COL_HIST_EDGE, fs=8.6)
b_h3 = box(0.15, 4.1, 3.15, 0.7, "3. The Ordinary Niacinamide\n10% + Zinc 1% Serum",
           COL_HIST, COL_HIST_EDGE, fs=8.6)
arrow(b_h1, b_h2, side1="bottom", side2="top", lw=1.2)
arrow(b_h2, b_h3, side1="bottom", side2="top", lw=1.2)

b_query = box(0.15, 2.55, 3.15, 1.0,
              "Query text $Q$\n(titles 1+2+3 concatenated)",
              COL_QUERY, COL_QUERY_EDGE, lw=2.0, fs=9.8, weight="bold")
arrow(b_h3, b_query, side1="bottom", side2="top", lw=1.3)

ax.text(0.15, 1.9, "Same query $Q$, two possible\nretrieval outcomes at rank 1 →",
        fontsize=9.3, style="italic", color="#444444", linespacing=1.4)

# ---- Panel (a): desired outcome ------------------------------------------
ax.text(5.05, 6.85, "(a) What a well-trained retriever should return",
        fontsize=10, weight="bold", color="#2e6b31")
b_a1 = box(5.05, 6.0, 3.9, 0.7,
           "#1  CeraVe Moisturizing Cream", COL_GOOD, COL_GOOD_EDGE, fs=9.3, weight="bold")
ax.text(9.15, 6.35, "✓ not in history\n✓ same skincare routine\n(moisturizing step)",
        fontsize=8, va="center", color="#2e6b31", linespacing=1.3)
b_a2 = box(5.05, 5.15, 3.9, 0.6, "#2  Aveeno Daily Moisturizing Lotion",
           COL_NEUTRAL, COL_NEUTRAL_EDGE, fs=8.6)
b_a3 = box(5.05, 4.4, 3.9, 0.6, "#3  La Roche-Posay Toleriane Moisturizer",
           COL_NEUTRAL, COL_NEUTRAL_EDGE, fs=8.6)
arrow(b_query, b_a1, side1="right", side2="left", connectionstyle="arc3,rad=-0.35", lw=1.4)

# ---- Panel (b): contaminated outcome --------------------------------------
ax.text(5.05, 3.35, "(b) What actually happens in practice — history contamination",
        fontsize=10, weight="bold", color="#8f2f2f")
b_b1 = box(5.05, 2.5, 3.9, 0.7,
           "#1  CeraVe Foaming Facial Cleanser", COL_BAD, COL_BAD_EDGE, fs=9.3, weight="bold")
ax.text(9.15, 2.85, "✗ already item 2\nin the query itself\n(re-recommended)",
        fontsize=8, va="center", color="#8f2f2f", linespacing=1.3)
b_b2 = box(5.05, 1.65, 3.9, 0.6, "#2  CeraVe Moisturizing Cream  (correct answer, pushed down)",
           COL_NEUTRAL, COL_NEUTRAL_EDGE, fs=8.3)
b_b3 = box(5.05, 0.9, 3.9, 0.6, "#3  Neutrogena Hydro Boost Gel Cream",
           COL_NEUTRAL, COL_NEUTRAL_EDGE, fs=8.6)
arrow(b_query, b_b1, side1="right", side2="left", connectionstyle="arc3,rad=0.08", lw=1.4)

plt.tight_layout()
plt.savefig("/media/administrator/Data1/Projects/Python/repLLaMA/paper/figures/fade_example.pdf",
            bbox_inches="tight")
print("saved fade_example.pdf")
