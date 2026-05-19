"""
================================================================================
A Comparative Study of Chain-of-Thought Reasoning in Older and Newer
Open Source LLaMA Models Across Math, Factual, and Commonsense Domains
================================================================================

FIGURES GENERATED
-----------------
  Fig 1  — CoT Faithfulness Rate by Model and Domain (grouped bar + CI)
  Fig 2  — Faithfulness Outcome Distribution per Model (100% stacked bar)
  Fig 3  — Answer Change Rate by Model and Domain
  Fig 4  — Answer Change Direction by Model (horizontal stacked bar)
  Fig 5  — Faithfulness Rate by Difficulty Level
  Fig 6  — Condition × Domain Faithfulness Heatmap
  Fig 7  — Summary Panel (multi-panel composite)

  Hypothesis figures (separate files):
  Fig 8a — H1: Faithfulness by Domain with significance brackets (Mann-Whitney U)
  Fig 8b — H1/H2/H3: -log10(p) significance plot
  Fig 8c — Pairwise domain comparison matrix
  Fig 8d — Hypothesis decision summary table

OUTPUT
------
  ./figures/   PNG (300 dpi) + PDF for every figure
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

warnings.filterwarnings("ignore")

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset":      "dejavuserif",
    "font.size":             10,
    "axes.titlesize":        11,
    "axes.titleweight":      "bold",
    "axes.labelsize":        10,
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "legend.fontsize":       9,
    "legend.title_fontsize": 9,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.linewidth":        0.8,
    "axes.grid":             True,
    "grid.color":            "#e0e0e0",
    "grid.linewidth":        0.5,
    "grid.linestyle":        "-",
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "xtick.major.size":      3,
    "ytick.major.size":      3,
    "xtick.major.width":     0.8,
    "ytick.major.width":     0.8,
    "figure.facecolor":      "white",
    "axes.facecolor":        "white",
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.facecolor":     "white",
})

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_PATH = "./faithfulness_results_20260513_185950.csv"
OUT_DIR  = "./figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette (colorblind-safe, two-model) ──────────────────────────────────────
C_M1 = "#2166ac"   # LLaMA-3.1 8B  — blue
C_M2 = "#b2182b"   # LLaMA-4 Scout — red

MODEL_COLORS  = {"LLaMA-3.1 8B": C_M1, "LLaMA-4 Scout 17B": C_M2}
MODEL_HATCHES = {"LLaMA-3.1 8B": "",   "LLaMA-4 Scout 17B": "///"}

CODE_COLORS = {
    "faithful":           "#1a6e37",
    "neutral_integrated": "#5aae61",
    "neutral_ignored":    "#d9f0d3",
    "unfaithful":         "#762a83",
    "control":            "#c2c2c2",
}
CODE_LABELS = {
    "faithful":           "Faithful",
    "neutral_integrated": "Neutral Integrated",
    "neutral_ignored":    "Neutral Ignored",
    "unfaithful":         "Unfaithful",
    "control":            "Control",
}

# ── Load & prep ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["model_short"] = df["model"].map({
    "meta-llama/llama-4-scout-17b-16e-instruct": "LLaMA-4 Scout 17B",
    "llama-3.1-8b-instant":                      "LLaMA-3.1 8B",
})
df_inj = df[df["condition"] != "control"].copy()

MODELS        = ["LLaMA-3.1 8B", "LLaMA-4 Scout 17B"]
DOMAINS       = ["mathematics", "factual", "commonsense"]
DOMAIN_LABELS = {"mathematics": "Mathematics", "factual": "Factual", "commonsense": "Commonsense"}
DIFFS         = ["easy", "medium", "hard"]
CONDITIONS    = ["neutral", "contradictory"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_DIR}/{name}.{ext}")
    print(f"  ✓  {name}")


def _mean_ci(series):
    s  = series.dropna()
    m  = s.mean()
    ci = 1.96 * s.std() / np.sqrt(len(s)) if len(s) > 1 else 0.0
    return m, ci, len(s)


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def add_sig_bracket(ax, x1, x2, y, p, dy=0.04, fontsize=8.5):
    stars = sig_stars(p)
    col   = "#222222" if stars != "ns" else "#888888"
    ax.plot([x1, x1, x2, x2], [y, y + dy * 0.6, y + dy * 0.6, y],
            lw=0.9, color=col)
    ax.text((x1 + x2) / 2, y + dy * 0.65, stars,
            ha="center", va="bottom", fontsize=fontsize,
            color=col, fontstyle="italic" if stars == "ns" else "normal")


def format_pct(ax, axis="y"):
    fmt = mticker.PercentFormatter(xmax=1)
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def note(ax, txt, x=0.0, y=-0.14, fontsize=8):
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=fontsize, va="top", color="#444444", fontstyle="italic")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1  — Faithfulness Rate by Model x Domain
# ─────────────────────────────────────────────────────────────────────────────
def fig1_faithfulness_domain():
    rates = (df_inj.groupby(["model_short", "domain"])["faithfulness_binary"]
             .agg(["mean", "std", "count"]).reset_index())
    rates["ci"] = 1.96 * rates["std"] / np.sqrt(rates["count"])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x, w = np.arange(len(DOMAINS)), 0.32

    for i, model in enumerate(MODELS):
        sub  = rates[rates["model_short"] == model].set_index("domain")
        vals = [sub.loc[d, "mean"] for d in DOMAINS]
        cis  = [sub.loc[d, "ci"]   for d in DOMAINS]
        xpos = x + (i - 0.5) * w
        bars = ax.bar(xpos, vals, width=w, color=MODEL_COLORS[model],
                      hatch=MODEL_HATCHES[model], edgecolor="white",
                      linewidth=0.6, label=model, zorder=3)
        ax.errorbar(xpos, vals, yerr=cis, fmt="none", color="#333333",
                    capsize=3, capthick=0.8, linewidth=0.8, zorder=4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.055,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS])
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Mean Faithfulness Rate (\u00b195% CI)")
    ax.set_xlabel("Domain")
    format_pct(ax)
    ax.axhline(0.5, color="#aaaaaa", lw=0.8, ls="--", zorder=1)
    ax.text(len(DOMAINS) - 0.5, 0.515, "chance level",
            fontsize=7.5, color="#888888", ha="right", va="bottom")
    ax.legend(frameon=True, framealpha=1, edgecolor="#cccccc",
              loc="upper left", handlelength=1.5)
    ax.set_title("Figure 1.  CoT Faithfulness Rate by Model and Domain")
    note(ax, "Note. Bars show mean faithfulness rate; error bars = 95% CI. "
             "Dashed line = chance level (0.50).")
    fig.tight_layout()
    save(fig, "fig1_faithfulness_by_domain")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2  — Faithfulness Outcome Distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig2_code_distribution():
    codes_order = ["faithful", "neutral_integrated", "neutral_ignored",
                   "unfaithful", "control"]
    counts = (df.groupby(["model_short", "faithfulness_code"])
              .size().unstack(fill_value=0)
              .reindex(columns=codes_order, fill_value=0))
    pcts = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    bottom  = np.zeros(len(MODELS))
    mlabels = [m.replace(" ", "\n") for m in MODELS]

    for code in codes_order:
        vals = [pcts.loc[m, code] for m in MODELS]
        bars = ax.bar(mlabels, vals, bottom=bottom, color=CODE_COLORS[code],
                      label=CODE_LABELS[code], edgecolor="white", linewidth=0.5, zorder=3)
        for bar, v, bot in zip(bars, vals, bottom):
            if v > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bot + v / 2,
                        f"{v:.0%}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottom += np.array(vals)

    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Proportion of Responses")
    ax.set_xlabel("Model")
    format_pct(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True,
              framealpha=1, edgecolor="#cccccc", title="Outcome", title_fontsize=8.5)
    ax.set_title("Figure 2.  Faithfulness Outcome Distribution per Model")
    note(ax, "Note. Proportions computed over all responses including control condition.")
    fig.tight_layout()
    save(fig, "fig2_code_distribution")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3  — Answer Change Rate by Model x Domain
# ─────────────────────────────────────────────────────────────────────────────
def fig3_answer_change():
    rates = (df_inj.groupby(["model_short", "domain"])["answer_changed"]
             .agg(["mean", "std", "count"]).reset_index())
    rates["ci"] = 1.96 * rates["std"] / np.sqrt(rates["count"])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x, w = np.arange(len(DOMAINS)), 0.32

    for i, model in enumerate(MODELS):
        sub  = rates[rates["model_short"] == model].set_index("domain")
        vals = [sub.loc[d, "mean"] for d in DOMAINS]
        cis  = [sub.loc[d, "ci"]   for d in DOMAINS]
        xpos = x + (i - 0.5) * w
        bars = ax.bar(xpos, vals, width=w, color=MODEL_COLORS[model],
                      hatch=MODEL_HATCHES[model], edgecolor="white",
                      linewidth=0.6, label=model, zorder=3)
        ax.errorbar(xpos, vals, yerr=cis, fmt="none", color="#333333",
                    capsize=3, capthick=0.8, linewidth=0.8, zorder=4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS])
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Proportion of Responses with Answer Change")
    ax.set_xlabel("Domain")
    format_pct(ax)
    ax.legend(frameon=True, framealpha=1, edgecolor="#cccccc", loc="upper right")
    ax.set_title("Figure 3.  Answer Change Rate After Step Injection by Domain")
    note(ax, "Note. Proportion of trials in which the model's final answer changed "
             "following the injected reasoning step.")
    fig.tight_layout()
    save(fig, "fig3_answer_change_rate")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4  — Answer Change Direction (horizontal stacked bar)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_change_direction():
    dir_colors = {"no_change": "#d1e5f0", "toward_injection": "#2166ac"}
    dir_labels = {"no_change": "No Change", "toward_injection": "Toward Injection"}
    dir_order  = ["toward_injection", "no_change"]

    counts = (df.groupby(["model_short", "change_direction"])
              .size().unstack(fill_value=0)
              .reindex(columns=dir_order, fill_value=0))
    pcts = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    fig.subplots_adjust(bottom=0.28)
    model_labels = MODELS[::-1]
    y    = np.arange(len(model_labels))
    left = np.zeros(len(model_labels))

    for direction in dir_order:
        vals = [pcts.loc[m, direction] for m in model_labels]
        bars = ax.barh(y, vals, left=left, height=0.45,
                       color=dir_colors[direction], label=dir_labels[direction],
                       edgecolor="white", linewidth=0.5)
        for bar, v, lft in zip(bars, vals, left):
            if v > 0.06:
                ax.text(lft + v / 2, bar.get_y() + bar.get_height() / 2,
                        f"{v:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if direction == "toward_injection" else "#444444",
                        fontweight="bold")
        left += np.array(vals)

    ax.set_yticks(y)
    ax.set_yticklabels(model_labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Proportion of Responses", labelpad=10)
    format_pct(ax, axis="x")
    ax.axvline(0.5, color="#aaaaaa", lw=0.8, ls="--")
    ax.legend(frameon=True, framealpha=1, edgecolor="#cccccc",
              loc="lower right", title="Change Direction")
    ax.set_title("Figure 4.  Distribution of Answer Change Direction by Model")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.text(0.12, 0.04,
             "Note. 'Toward Injection' = final answer shifted in the direction "
             "of the injected reasoning step.",
             fontsize=8, color="#444444", fontstyle="italic", va="bottom")
    save(fig, "fig4_change_direction")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5  — Faithfulness by Difficulty Level
# ─────────────────────────────────────────────────────────────────────────────
def fig5_difficulty():
    rates = (df_inj.groupby(["model_short", "difficulty"])["faithfulness_binary"]
             .agg(["mean", "std", "count"]).reset_index())
    rates["ci"] = 1.96 * rates["std"] / np.sqrt(rates["count"])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    x, w = np.arange(len(DIFFS)), 0.32

    for i, model in enumerate(MODELS):
        sub  = rates[rates["model_short"] == model].set_index("difficulty")
        vals = [sub.loc[d, "mean"] for d in DIFFS]
        cis  = [sub.loc[d, "ci"]   for d in DIFFS]
        xpos = x + (i - 0.5) * w
        bars = ax.bar(xpos, vals, width=w, color=MODEL_COLORS[model],
                      hatch=MODEL_HATCHES[model], edgecolor="white",
                      linewidth=0.6, label=model, zorder=3)
        ax.errorbar(xpos, vals, yerr=cis, fmt="none", color="#333333",
                    capsize=3, capthick=0.8, linewidth=0.8, zorder=4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Medium", "Hard"])
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Mean Faithfulness Rate (\u00b195% CI)")
    ax.set_xlabel("Difficulty Level")
    format_pct(ax)
    ax.axhline(0.5, color="#aaaaaa", lw=0.8, ls="--")
    ax.legend(frameon=True, framealpha=1, edgecolor="#cccccc", loc="upper left")
    ax.set_title("Figure 5.  Faithfulness Rate by Question Difficulty Level")
    note(ax, "Note. Error bars = 95% CI. Dashed line = chance level (0.50).")
    fig.tight_layout()
    save(fig, "fig5_faithfulness_by_difficulty")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6  — Condition x Domain Heatmap
# NOTE: The 'neutral' condition has no faithfulness_binary scores (injected
# steps were either integrated or ignored without a binary outcome). This
# figure therefore shows two separate metrics:
#   Row 1 — Faithfulness rate under the Contradictory condition
#   Row 2 — Answer change rate under the Neutral condition
# ─────────────────────────────────────────────────────────────────────────────
def fig6_heatmap():
    cmap = LinearSegmentedColormap.from_list(
        "blues_seq", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#084594"], N=256)

    # Build two-row pivots: contradictory faithfulness + neutral answer_changed
    row_labels = ["Contradictory\n(Faithfulness Rate)", "Neutral\n(Answer Change Rate)"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0),
                             gridspec_kw={"wspace": 0.35})

    ims = []
    for ax, model in zip(axes, MODELS):
        sub = df_inj[df_inj["model_short"] == model]

        row0 = (sub[sub["condition"] == "contradictory"]
                .groupby("domain")["faithfulness_binary"]
                .mean()
                .reindex(DOMAINS))

        row1 = (sub[sub["condition"] == "neutral"]
                .groupby("domain")["answer_changed"]
                .mean()
                .reindex(DOMAINS))

        data = np.array([row0.values, row1.values], dtype=float)

        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ims.append(im)

        ax.set_xticks(range(len(DOMAINS)))
        ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS],
                           rotation=15, ha="right", fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(row_labels, fontsize=8.5)
        ax.set_title(model, fontsize=10, fontweight="bold", pad=8)

        for r in range(2):
            for c in range(len(DOMAINS)):
                val = data[r, c]
                if np.isnan(val):
                    ax.text(c, r, "N/A", ha="center", va="center",
                            fontsize=9, color="#888888")
                else:
                    tc = "white" if val > 0.55 else "#222222"
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                            fontsize=10, color=tc, fontweight="bold")

        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.6)
            sp.set_color("#bbbbbb")

    # Single shared colorbar to the right of both axes, well clear of titles
    cbar = fig.colorbar(ims[1], ax=axes[1], fraction=0.055, pad=0.18,
                        shrink=0.85)
    cbar.set_label("Rate (0 – 1)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Figure 6.  Response Rates by Condition and Domain",
                 fontsize=11, fontweight="bold", y=1.04)

    fig.text(0.5, -0.08,
             "Note. Row 1 (Contradictory): proportion of trials where the model's "
             "reasoning faithfully followed the injected step.\n"
             "Row 2 (Neutral): proportion of trials where the final answer changed "
             "after a consistency-preserving injected step.",
             ha="center", fontsize=8, color="#444444", fontstyle="italic",
             va="top")

    fig.tight_layout()
    save(fig, "fig6_heatmap")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7  — Summary Multi-Panel
# ─────────────────────────────────────────────────────────────────────────────
def fig7_summary_panel():
    fig = plt.figure(figsize=(13, 9.5))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.44,
                            left=0.07, right=0.97, top=0.89, bottom=0.09)
    fig.suptitle(
        "A Comparative Study of CoT Faithfulness in LLaMA Models\n"
        "LLaMA-3.1 8B vs. LLaMA-4 Scout 17B  |  "
        "Mathematics \u00b7 Factual \u00b7 Commonsense  |  Step-Intervention Paradigm",
        fontsize=11, fontweight="bold", y=0.97, linespacing=1.5)

    x, w = np.arange(len(DOMAINS)), 0.32

    # Panel A
    ax_a = fig.add_subplot(gs[0, 0])
    rates = (df_inj.groupby(["model_short", "domain"])["faithfulness_binary"]
             .agg(["mean", "std", "count"]).reset_index())
    rates["ci"] = 1.96 * rates["std"] / np.sqrt(rates["count"])
    for i, model in enumerate(MODELS):
        sub  = rates[rates["model_short"] == model].set_index("domain")
        vals = [sub.loc[d, "mean"] for d in DOMAINS]
        cis  = [sub.loc[d, "ci"]   for d in DOMAINS]
        xpos = x + (i - 0.5) * w
        ax_a.bar(xpos, vals, width=w, color=MODEL_COLORS[model],
                 hatch=MODEL_HATCHES[model], edgecolor="white",
                 linewidth=0.5, label=model, zorder=3)
        ax_a.errorbar(xpos, vals, yerr=cis, fmt="none",
                      color="#333333", capsize=2.5, linewidth=0.7, zorder=4)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS], fontsize=8.5)
    ax_a.set_ylim(0, 1.18)
    ax_a.set_ylabel("Faithfulness Rate")
    format_pct(ax_a)
    ax_a.axhline(0.5, color="#aaaaaa", lw=0.8, ls="--")
    ax_a.legend(fontsize=7.5, frameon=True, edgecolor="#cccccc", loc="upper left")
    ax_a.set_title("(A)  Faithfulness by Domain", fontsize=10)

    # Panel B
    ax_b = fig.add_subplot(gs[0, 1])
    codes_order = ["faithful", "neutral_integrated", "neutral_ignored", "unfaithful", "control"]
    cnt = (df.groupby(["model_short", "faithfulness_code"])
           .size().unstack(fill_value=0)
           .reindex(columns=codes_order, fill_value=0))
    pct = cnt.div(cnt.sum(axis=1), axis=0)
    bot = np.zeros(len(MODELS))
    mlabels = [m.replace(" ", "\n") for m in MODELS]
    for code in codes_order:
        vals = [pct.loc[m, code] for m in MODELS]
        bars = ax_b.bar(mlabels, vals, bottom=bot, color=CODE_COLORS[code],
                        label=CODE_LABELS[code], edgecolor="white", linewidth=0.4)
        for bar, v, b in zip(bars, vals, bot):
            if v > 0.07:
                ax_b.text(bar.get_x() + bar.get_width() / 2, b + v / 2,
                          f"{v:.0%}", ha="center", va="center",
                          fontsize=7.5, color="white", fontweight="bold")
        bot += np.array(vals)
    ax_b.set_ylim(0, 1.02)
    ax_b.set_ylabel("Proportion")
    format_pct(ax_b)
    ax_b.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1),
                frameon=True, edgecolor="#cccccc", title="Outcome", title_fontsize=7)
    ax_b.set_title("(B)  Outcome Distribution", fontsize=10)

    # Panel C
    ax_c = fig.add_subplot(gs[0, 2])
    ar = (df_inj.groupby(["model_short", "domain"])["answer_changed"]
          .agg(["mean", "std", "count"]).reset_index())
    ar["ci"] = 1.96 * ar["std"] / np.sqrt(ar["count"])
    for i, model in enumerate(MODELS):
        sub  = ar[ar["model_short"] == model].set_index("domain")
        vals = [sub.loc[d, "mean"] for d in DOMAINS]
        cis  = [sub.loc[d, "ci"]   for d in DOMAINS]
        xpos = x + (i - 0.5) * w
        ax_c.bar(xpos, vals, width=w, color=MODEL_COLORS[model],
                 hatch=MODEL_HATCHES[model], edgecolor="white",
                 linewidth=0.5, label=model, zorder=3)
        ax_c.errorbar(xpos, vals, yerr=cis, fmt="none",
                      color="#333333", capsize=2.5, linewidth=0.7, zorder=4)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS], fontsize=8.5)
    ax_c.set_ylim(0, 0.92)
    ax_c.set_ylabel("Answer Change Rate")
    format_pct(ax_c)
    ax_c.legend(fontsize=7.5, frameon=True, edgecolor="#cccccc", loc="upper right")
    ax_c.set_title("(C)  Answer Change Rate", fontsize=10)

    # Panel D
    ax_d = fig.add_subplot(gs[1, 0])
    dr = (df_inj.groupby(["model_short", "difficulty"])["faithfulness_binary"]
          .agg(["mean", "std", "count"]).reset_index())
    dr["ci"] = 1.96 * dr["std"] / np.sqrt(dr["count"])
    xd = np.arange(len(DIFFS))
    for i, model in enumerate(MODELS):
        sub  = dr[dr["model_short"] == model].set_index("difficulty")
        vals = [sub.loc[d, "mean"] for d in DIFFS]
        cis  = [sub.loc[d, "ci"]   for d in DIFFS]
        xpos = xd + (i - 0.5) * w
        ax_d.bar(xpos, vals, width=w, color=MODEL_COLORS[model],
                 hatch=MODEL_HATCHES[model], edgecolor="white",
                 linewidth=0.5, label=model, zorder=3)
        ax_d.errorbar(xpos, vals, yerr=cis, fmt="none",
                      color="#333333", capsize=2.5, linewidth=0.7, zorder=4)
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels(["Easy", "Medium", "Hard"])
    ax_d.set_ylim(0, 1.18)
    ax_d.set_ylabel("Faithfulness Rate")
    format_pct(ax_d)
    ax_d.axhline(0.5, color="#aaaaaa", lw=0.8, ls="--")
    ax_d.legend(fontsize=7.5, frameon=True, edgecolor="#cccccc", loc="upper left")
    ax_d.set_title("(D)  Faithfulness by Difficulty", fontsize=10)

    # Panels E & F — heatmaps (contradictory faithfulness + neutral answer_changed)
    cmap_h = LinearSegmentedColormap.from_list(
        "blues_seq", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#084594"], N=256)
    row_labels_short = ["Contradictory\n(Faithfulness)", "Neutral\n(Ans. Change)"]
    for col_idx, model in enumerate(MODELS):
        ax_h = fig.add_subplot(gs[1, col_idx + 1])
        sub  = df_inj[df_inj["model_short"] == model]
        row0 = (sub[sub["condition"] == "contradictory"]
                .groupby("domain")["faithfulness_binary"].mean().reindex(DOMAINS))
        row1 = (sub[sub["condition"] == "neutral"]
                .groupby("domain")["answer_changed"].mean().reindex(DOMAINS))
        data = np.array([row0.values, row1.values], dtype=float)
        im   = ax_h.imshow(data, cmap=cmap_h, vmin=0, vmax=1, aspect="auto")
        ax_h.set_xticks(range(len(DOMAINS)))
        ax_h.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS],
                              rotation=20, ha="right", fontsize=8.5)
        ax_h.set_yticks([0, 1])
        ax_h.set_yticklabels(row_labels_short, fontsize=8)
        for r in range(2):
            for c in range(len(DOMAINS)):
                val = data[r, c]
                tc  = "white" if val > 0.55 else "#222222"
                ax_h.text(c, r, f"{val:.2f}", ha="center", va="center",
                          fontsize=9, color=tc, fontweight="bold")
        letter = "E" if col_idx == 0 else "F"
        ax_h.set_title(f"({letter})  Condition Heatmap — {model}", fontsize=9.5)
        for sp in ax_h.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color("#bbbbbb")
        plt.colorbar(im, ax=ax_h, fraction=0.048, pad=0.04).ax.tick_params(labelsize=7)

    fig.tight_layout()
    save(fig, "fig7_summary_panel")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Shared: compute all hypothesis statistics once
# ─────────────────────────────────────────────────────────────────────────────
def _compute_hyp_stats():
    res, pv = {}, {}
    for domain in DOMAINS:
        g1 = df_inj[(df_inj["model_short"] == "LLaMA-3.1 8B")      & (df_inj["domain"] == domain)]["faithfulness_binary"].dropna()
        g2 = df_inj[(df_inj["model_short"] == "LLaMA-4 Scout 17B") & (df_inj["domain"] == domain)]["faithfulness_binary"].dropna()
        res[domain] = {
            "LLaMA-3.1 8B":      _mean_ci(g1),
            "LLaMA-4 Scout 17B": _mean_ci(g2),
        }
        _, p = stats.mannwhitneyu(g1, g2, alternative="less")
        pv[domain] = p

    ct_ov = pd.crosstab(df_inj["model_short"], df_inj["faithfulness_binary"])
    _, p_ov, _, _ = stats.chi2_contingency(ct_ov)

    sub_c = df_inj[df_inj["condition"] == "contradictory"]
    ct_ch = pd.crosstab(sub_c["model_short"], sub_c["answer_changed"])
    _, p_ch, _, _ = stats.chi2_contingency(ct_ch)

    pw = {}
    for model in MODELS:
        pw[model] = {}
        for d1, d2 in combinations(DOMAINS, 2):
            g1 = df_inj[(df_inj["model_short"] == model) & (df_inj["domain"] == d1)]["faithfulness_binary"].dropna()
            g2 = df_inj[(df_inj["model_short"] == model) & (df_inj["domain"] == d2)]["faithfulness_binary"].dropna()
            _, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            pw[model][(d1, d2)] = p

    return res, pv, p_ov, p_ch, pw


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8a — H1: Faithfulness by Domain + significance brackets
# ─────────────────────────────────────────────────────────────────────────────
def fig8a_h1_brackets():
    res, pv, _, _, _ = _compute_hyp_stats()

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    x, w = np.arange(len(DOMAINS)), 0.32

    for i, model in enumerate(MODELS):
        vals = [res[d][model][0] for d in DOMAINS]
        cis  = [res[d][model][1] for d in DOMAINS]
        xpos = x + (i - 0.5) * w
        bars = ax.bar(xpos, vals, width=w, color=MODEL_COLORS[model],
                      hatch=MODEL_HATCHES[model], edgecolor="white",
                      linewidth=0.6, label=model, zorder=3)
        ax.errorbar(xpos, vals, yerr=cis, fmt="none", color="#333333",
                    capsize=3, capthick=0.8, linewidth=0.8, zorder=4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    for i, domain in enumerate(DOMAINS):
        p   = pv[domain]
        top = max(res[domain]["LLaMA-3.1 8B"][0], res[domain]["LLaMA-4 Scout 17B"][0])
        ci_max = max(res[domain]["LLaMA-3.1 8B"][1], res[domain]["LLaMA-4 Scout 17B"][1])
        y   = top + ci_max + 0.11
        add_sig_bracket(ax, x[i] - w / 2, x[i] + w / 2, y, p, dy=0.045)

    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS])
    ax.set_ylim(0, 1.42)
    ax.set_ylabel("Mean Faithfulness Rate (\u00b195% CI)")
    ax.set_xlabel("Domain")
    format_pct(ax)
    ax.axhline(0.5, color="#aaaaaa", lw=0.8, ls="--")
    ax.text(len(DOMAINS) - 0.5, 0.515, "chance level",
            fontsize=7.5, color="#888888", ha="right", va="bottom")
    ax.legend(frameon=True, framealpha=1, edgecolor="#cccccc", loc="upper left")
    ax.set_title(
        "Figure 8a.  H\u2081: LLaMA-4 Scout 17B Exhibits Higher CoT Faithfulness\n"
        "Than LLaMA-3.1 8B Across All Domains  (Mann-Whitney U, one-tailed, \u03b1 = .05)")
    note(ax, "Note. Brackets show pairwise model comparisons per domain. "
             "* p < .05   ** p < .01   *** p < .001   ns = not significant.",
         y=-0.13)
    fig.tight_layout()
    save(fig, "fig8a_h1_significance_brackets")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8b — -log10(p) significance overview
# ─────────────────────────────────────────────────────────────────────────────
def fig8b_log_p_plot():
    _, pv, p_ov, p_ch, _ = _compute_hyp_stats()

    tests = [
        ("H\u2081a:  Mathematics  (newer > older)",      pv["mathematics"]),
        ("H\u2081b:  Factual  (newer > older)",          pv["factual"]),
        ("H\u2081c:  Commonsense  (newer > older)",      pv["commonsense"]),
        ("H\u2082:  Overall faithfulness differs",       p_ov),
        ("H\u2083:  Answer change rate differs",         p_ch),
    ]
    labels = [t[0] for t in tests]
    pvals  = [t[1] for t in tests]
    log_p  = [-np.log10(p) for p in pvals]

    def bar_color(p):
        if p < 0.001: return "#1a3a5c"
        if p < 0.01:  return "#2166ac"
        if p < 0.05:  return "#74add1"
        return "#cccccc"

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y    = np.arange(len(tests))
    bars = ax.barh(y, log_p, color=[bar_color(p) for p in pvals],
                   edgecolor="white", linewidth=0.5, height=0.55, zorder=3)

    t05  = -np.log10(0.05)
    t01  = -np.log10(0.01)
    t001 = -np.log10(0.001)
    ax.axvline(t05,  color="#333333", lw=1.1, ls="--", zorder=4, label="p = .05")
    ax.axvline(t01,  color="#777777", lw=0.9, ls=":",  zorder=4, label="p = .01")
    ax.axvline(t001, color="#aaaaaa", lw=0.7, ls=":",  zorder=4, label="p = .001")

    for bar, p, lv in zip(bars, pvals, log_p):
        st  = sig_stars(p)
        col = "#222222" if st != "ns" else "#888888"
        ax.text(lv + 0.06, bar.get_y() + bar.get_height() / 2,
                f"{st}   p = {p:.4f}", va="center", fontsize=8.5,
                color=col, fontstyle="normal")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(r"$-\log_{10}(p\text{-value})$  —  larger = more significant")
    ax.set_xlim(0, max(log_p) * 1.65)
    ax.legend(frameon=True, framealpha=1, edgecolor="#cccccc",
              fontsize=8.5, loc="lower right")
    ax.set_title("Figure 8b.  Statistical Significance Overview\n"
                 r"$-\log_{10}(p)$ Plot Across All Hypothesis Tests  ($\alpha = .05$)")
    ax.grid(axis="x", zorder=0)
    ax.grid(axis="y", visible=False)
    ax.axvspan(0, t05, alpha=0.035, color="#d62728", zorder=0)
    ax.axvspan(t05, max(log_p) * 1.65, alpha=0.03, color="#2166ac", zorder=0)
    note(ax, "Note. Red shaded region = p > .05 (not significant). "
             "Blue shaded region = p < .05 (significant). "
             "All five tests exceed the alpha threshold.",
         y=-0.13)
    fig.tight_layout()
    save(fig, "fig8b_log_p_plot")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8c — Pairwise domain comparison matrix (both models)
# ─────────────────────────────────────────────────────────────────────────────
def fig8c_pairwise_matrix():
    _, _, _, _, pw = _compute_hyp_stats()

    n_d = len(DOMAINS)
    dlabels = [DOMAIN_LABELS[d] for d in DOMAINS]
    cmap_p  = LinearSegmentedColormap.from_list(
        "pmat", ["#2166ac", "#d1e5f0", "#f7f7f7"], N=256)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    for ax, model in zip(axes, MODELS):
        mat = np.full((n_d, n_d), np.nan)
        for i, d1 in enumerate(DOMAINS):
            for j, d2 in enumerate(DOMAINS):
                if i < j:
                    p = pw[model].get((d1, d2), pw[model].get((d2, d1), np.nan))
                    mat[i, j] = p
                    mat[j, i] = p

        display = np.where(np.isnan(mat), 1.0, mat)
        masked  = np.ma.array(display, mask=np.eye(n_d, dtype=bool))
        im = ax.imshow(masked, cmap=cmap_p, vmin=0, vmax=0.15, aspect="equal")

        for i in range(n_d):
            for j in range(n_d):
                if i == j:
                    ax.text(j, i, "\u2014", ha="center", va="center",
                            fontsize=13, color="#aaaaaa")
                else:
                    p  = mat[i, j]
                    st = sig_stars(p)
                    fc = "white" if p < 0.05 else "#333333"
                    ax.text(j, i, f"{st}\np = {p:.3f}",
                            ha="center", va="center",
                            fontsize=8.5, color=fc, linespacing=1.5)

        ax.set_xticks(range(n_d))
        ax.set_xticklabels(dlabels, rotation=15, ha="right")
        ax.set_yticks(range(n_d))
        ax.set_yticklabels(dlabels)
        ax.set_title(model, fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color("#bbbbbb")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="p-value").ax.tick_params(labelsize=8)

    fig.suptitle("Figure 8c.  Pairwise Domain Significance Matrix\n"
                 "Mann-Whitney U Test (two-tailed, \u03b1 = .05)",
                 fontsize=11, fontweight="bold", y=1.04)
    note(axes[0],
         "Note. Darker blue = smaller p-value (stronger evidence). "
         "* p < .05   ** p < .01   *** p < .001   ns = not significant.",
         y=-0.25, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig8c_pairwise_matrix")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8d — Hypothesis Summary Table
# ─────────────────────────────────────────────────────────────────────────────
def fig8d_hypothesis_table():
    _, pv, p_ov, p_ch, _ = _compute_hyp_stats()

    rows = [
        ("H\u2081a", "LLaMA-4 Scout 17B > LLaMA-3.1 8B on Mathematics",
         "Mann-Whitney U (one-tailed)", pv["mathematics"]),
        ("H\u2081b", "LLaMA-4 Scout 17B > LLaMA-3.1 8B on Factual",
         "Mann-Whitney U (one-tailed)", pv["factual"]),
        ("H\u2081c", "LLaMA-4 Scout 17B > LLaMA-3.1 8B on Commonsense",
         "Mann-Whitney U (one-tailed)", pv["commonsense"]),
        ("H\u2082",  "Overall faithfulness distributions differ by model",
         "Pearson \u03c7\u00b2 (two-tailed)", p_ov),
        ("H\u2083",  "Answer change rate differs significantly by model",
         "Pearson \u03c7\u00b2 (two-tailed)", p_ch),
    ]

    col_headers = ["Hypothesis", "Statement", "Statistical Test",
                   "p-value", "Sig.", "Decision"]
    col_x = [0.0, 0.09, 0.52, 0.76, 0.855, 0.915]

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    row_h    = 0.72 / n_rows
    header_y = 0.96

    # Header background
    ax.add_patch(mpatches.Rectangle(
        (0, header_y - row_h * 0.95), 1.0, row_h * 0.95,
        facecolor="#2c3e50", edgecolor="none",
        transform=ax.transAxes, clip_on=False, zorder=2))
    for cx, ch in zip(col_x, col_headers):
        ax.text(cx + 0.008, header_y - row_h * 0.95 / 2, ch,
                transform=ax.transAxes, va="center",
                fontsize=9.5, fontweight="bold", color="white", zorder=3)

    # Data rows
    for r_i, (hyp, stmt, test, p) in enumerate(rows):
        y_top = header_y - row_h * 0.95 - r_i * row_h
        bg    = "#f5f8ff" if r_i % 2 == 0 else "white"
        ax.add_patch(mpatches.Rectangle(
            (0, y_top - row_h), 1.0, row_h,
            facecolor=bg, edgecolor="#e0e0e0", linewidth=0.5,
            transform=ax.transAxes, clip_on=False, zorder=1))

        st   = sig_stars(p)
        supp = p < 0.05
        vals_row = [hyp, stmt, test,
                    f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}",
                    st,
                    "Supported" if supp else "Not Supported"]
        cy = y_top - row_h / 2

        for cx, val in zip(col_x, vals_row):
            color = "#222222"; fw = "normal"
            if val == "Supported":       color = "#1a5c1a"; fw = "bold"
            elif val == "Not Supported": color = "#8b0000"; fw = "bold"
            elif val in ("***", "**", "*"): color = "#1a3a5c"; fw = "bold"
            elif val == "ns":            color = "#888888"
            ax.text(cx + 0.008, cy, val,
                    transform=ax.transAxes, va="center",
                    fontsize=9, color=color, fontweight=fw, zorder=3)

    # Outer border
    ax.add_patch(mpatches.Rectangle(
        (0, header_y - row_h * 0.95 - n_rows * row_h), 1.0,
        row_h * 0.95 + n_rows * row_h,
        facecolor="none", edgecolor="#666666", linewidth=0.8,
        transform=ax.transAxes, clip_on=False, zorder=4))

    # Note below table
    note_y = header_y - row_h * 0.95 - n_rows * row_h - 0.04
    ax.text(0, note_y,
            "Significance codes:  *** p < .001   ** p < .01   * p < .05   "
            "ns p \u2265 .05   |   \u03b1 = .05",
            transform=ax.transAxes, fontsize=8,
            color="#444444", va="top", fontstyle="italic")

    ax.set_title("Figure 8d.  Summary of Hypothesis Tests",
                 fontsize=11, fontweight="bold", pad=14)
    fig.tight_layout()
    save(fig, "fig8d_hypothesis_table")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Generating publication-quality figures...\n")
    fig1_faithfulness_domain()
    fig2_code_distribution()
    fig3_answer_change()
    fig4_change_direction()
    fig5_difficulty()
    fig6_heatmap()
    fig7_summary_panel()
    fig8a_h1_brackets()
    fig8b_log_p_plot()
    fig8c_pairwise_matrix()
    fig8d_hypothesis_table()
    print(f"\n  All figures saved to: {OUT_DIR}/\n")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"   {f}")
