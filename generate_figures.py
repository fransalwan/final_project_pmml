#!/usr/bin/env python3
"""
Generates high-resolution, publication-grade figures (300 DPI) from
experiment_outputs/experimental_results.json for the Notebook, IEEE/ACL Paper, and Poster.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.9,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.dpi": 300,
})

METHOD_LABELS = {
    "BM25_Base": "BM25 (Base)",
    "BM25_Context": "BM25 (+Step / +Hist)",
    "Dense_BiEncoder": "Dense Bi-Encoder",
    "Ablation_NoHardNegs": "w/o Hard Negatives",
    "Ablation_NoTempConvFeats": "w/o Temp/Turn Feats",
    "Ablation_NoAdaptivePrior": "w/o Adaptive Prior",
    "Ablation_NoGating": "w/o Contextual Gate",
    "Proposed_TempConv_GatedRanker": "TempConv-GatedRanker (Ours)",
}

COLORS = {
    "BM25_Base": "#94a3b8",
    "BM25_Context": "#64748b",
    "Dense_BiEncoder": "#3b82f6",
    "Ablation_NoHardNegs": "#f59e0b",
    "Ablation_NoTempConvFeats": "#f97316",
    "Ablation_NoAdaptivePrior": "#ec4899",
    "Ablation_NoGating": "#8b5cf6",
    "Proposed_TempConv_GatedRanker": "#10b981",
}


def main():
    res_path = os.path.join("experiment_outputs", "experimental_results.json")
    with open(res_path, encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs("experiment_outputs/figures", exist_ok=True)
    macro = data["macro_average"]
    per_dom = data["per_domain"]

    # -------------------------------------------------------------------------
    # Figure 1: Macro-Average nDCG@10 & MRR@10 on Held-Out Dev (Track 1 & Track 2)
    # -------------------------------------------------------------------------
    core_methods = [
        "BM25_Base",
        "BM25_Context",
        "Dense_BiEncoder",
        "Ablation_NoAdaptivePrior",
        "Proposed_TempConv_GatedRanker",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for idx, (track_key, title) in enumerate([
        ("track1_tempo", "Track 1: Temporal Grounded Retrieval (Dev)"),
        ("track2_recor", "Track 2: Conversational Retrieval (Dev)"),
    ]):
        ax = axes[idx]
        x = np.arange(len(core_methods))
        width = 0.26
        ndcg_vals = [macro[track_key]["dev"][m]["nDCG@10"] for m in core_methods]
        mrr_vals = [macro[track_key]["dev"][m]["MRR@10"] for m in core_methods]
        rec_vals = [macro[track_key]["dev"][m]["Recall@10"] for m in core_methods]

        b1 = ax.bar(x - width, ndcg_vals, width, label="nDCG@10 (Official)", color="#1d4ed8", edgecolor="white")
        b2 = ax.bar(x, mrr_vals, width, label="MRR@10", color="#0ea5e9", edgecolor="white")
        b3 = ax.bar(x + width, rec_vals, width, label="Recall@10", color="#10b981", edgecolor="white")

        for bars in (b1, b2, b3):
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1e293b"
                )

        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=12, color="#0f172a")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in core_methods], rotation=18, ha="right", fontsize=8.5)
        ax.set_ylim(0, max(max(ndcg_vals), max(mrr_vals), max(rec_vals)) * 1.18)
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.set_ylabel("Metric Score", fontsize=10, fontweight="bold")
            ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5, loc="upper left")

    plt.tight_layout()
    plt.savefig("experiment_outputs/figures/fig1_overall_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 2: Track 1 Official Temporal Diagnostics (nDCG@10, TC@10, TP@10, NDCG|FC@10)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    t1_metrics = ["nDCG@10", "TC@10", "TP@10", "TR@10", "NDCG|FC@10"]
    t1_methods = ["BM25_Base", "BM25_Context", "Dense_BiEncoder", "Ablation_NoTempConvFeats", "Proposed_TempConv_GatedRanker"]
    x = np.arange(len(t1_metrics))
    width = 0.15

    for i, m in enumerate(t1_methods):
        vals = [macro["track1_tempo"]["dev"][m][mk] for mk in t1_metrics]
        offset = (i - len(t1_methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=METHOD_LABELS[m], color=COLORS[m], edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    ax.set_title("Track 1 (TEMPO): Official Temporal Grounding Metrics on Held-Out Dev", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([
        "nDCG@10\n(Official Rank)",
        "Temporal Coverage\n(TC@10)",
        "Temporal Precision\n(TP@10)",
        "Temporal Relevance\n(TR@10)",
        "Full-Cov nDCG\n(NDCG|FC@10)"
    ], fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Score", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.90)
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5, ncol=3, loc="upper left")
    plt.tight_layout()
    plt.savefig("experiment_outputs/figures/fig2_track1_temporal_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 3: Track 2 Conversational Turn-Depth Degradation Analysis (T1 to T5+)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    turns = ["T1", "T2", "T3", "T4", "T5+"]
    t2_plot_methods = [
        "BM25_Base",
        "BM25_Context",
        "Dense_BiEncoder",
        "Ablation_NoTempConvFeats",
        "Proposed_TempConv_GatedRanker",
    ]
    markers = ["o", "s", "^", "d", "P"]

    for s_idx, (split_key, s_title) in enumerate([("train", "Train Split (427 Turns)"), ("dev", "Held-Out Dev Split (158 Turns)")]):
        ax = axes[s_idx]
        for m_idx, m in enumerate(t2_plot_methods):
            turn_vals = []
            for t_bucket in turns:
                dom_scores = [
                    per_dom["track2_recor"][d][split_key][m]["by_turn"].get(t_bucket, 0.0)
                    for d in per_dom["track2_recor"]
                    if t_bucket in per_dom["track2_recor"][d][split_key][m]["by_turn"]
                ]
                turn_vals.append(float(np.mean(dom_scores)) if dom_scores else 0.0)
            lw = 2.8 if m == "Proposed_TempConv_GatedRanker" else 1.8
            ls = "-" if m in ("Proposed_TempConv_GatedRanker", "BM25_Context", "Dense_BiEncoder") else "--"
            ax.plot(turns, turn_vals, marker=markers[m_idx], linewidth=lw, linestyle=ls,
                    markersize=7, color=COLORS[m], label=METHOD_LABELS[m])

        ax.set_title(f"Track 2 nDCG@10 by Turn Depth — {s_title}", fontsize=10.5, fontweight="bold", pad=10)
        ax.set_xlabel("Conversation Turn Position", fontsize=10, fontweight="bold")
        if s_idx == 0:
            ax.set_ylabel("Macro-Averaged nDCG@10", fontsize=10, fontweight="bold")
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8, loc="best")

    plt.subplots_adjust(wspace=0.22)
    plt.savefig("experiment_outputs/figures/fig3_track2_turn_degradation.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 4: Comprehensive Ablation Study Across Track 1 & Track 2
    # -------------------------------------------------------------------------
    ablation_order = [
        "Proposed_TempConv_GatedRanker",
        "Ablation_NoGating",
        "Ablation_NoAdaptivePrior",
        "Ablation_NoHardNegs",
        "Ablation_NoTempConvFeats",
        "BM25_Context",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.6), sharey=True)
    for idx, (track_key, t_name) in enumerate([
        ("track1_tempo", "Track 1: Temporal Retrieval (Dev nDCG@10)"),
        ("track2_recor", "Track 2: Conversational Retrieval (Dev nDCG@10)"),
    ]):
        ax = axes[idx]
        vals = [macro[track_key]["dev"][m]["nDCG@10"] for m in ablation_order]
        y_pos = np.arange(len(ablation_order))
        bar_colors = [COLORS[m] for m in ablation_order]
        bars = ax.barh(y_pos, vals, color=bar_colors, edgecolor="white", height=0.62)
        ax.set_yticks(y_pos)
        if idx == 0:
            ax.set_yticklabels([METHOD_LABELS[m] for m in ablation_order], fontsize=9.5, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Held-Out Dev nDCG@10", fontsize=9.5, fontweight="bold")
        ax.set_title(t_name, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlim(0, max(vals) * 1.38)
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        full_val = vals[0]
        for i, bar in enumerate(bars):
            w = bar.get_width()
            diff = w - full_val
            diff_str = " (Full)" if i == 0 else f" ({diff:+.4f})"
            ax.annotate(f"{w:.4f}{diff_str}", xy=(w, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=8.2, fontweight="bold")

    plt.subplots_adjust(wspace=0.12)
    plt.savefig("experiment_outputs/figures/fig4_ablation_study.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 5: Training Convergence Curves & Learned Contextual Gate Weights
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax0 = axes[0]
    epochs = np.arange(1, 21)
    for dom_label, (trk, dm, col, ls) in [
        ("Track 1: workplace", ("track1_tempo", "workplace", "#2563eb", "-")),
        ("Track 1: quant", ("track1_tempo", "quant", "#0ea5e9", "--")),
        ("Track 2: drones", ("track2_recor", "drones", "#10b981", "-")),
        ("Track 2: law", ("track2_recor", "law", "#8b5cf6", "-.")),
        ("Track 2: politics", ("track2_recor", "politics", "#f59e0b", ":")),
    ]:
        curve = per_dom[trk][dm]["loss_curves"]["Proposed_TempConv_GatedRanker"]
        ax0.plot(epochs, curve, label=dom_label, color=col, linestyle=ls, linewidth=2.2, marker="o", markersize=4)

    ax0.set_title("Pairwise Margin + BCE Training Loss Convergence", fontsize=11, fontweight="bold", pad=10)
    ax0.set_xlabel("Training Epoch", fontsize=9.5, fontweight="bold")
    ax0.set_ylabel("Composite Ranking Loss", fontsize=9.5, fontweight="bold")
    ax0.grid(True)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5)

    # Gate attention distribution across domains
    ax1 = axes[1]
    dom_list = [
        ("track1_tempo", "iota", "T1: iota"),
        ("track1_tempo", "quant", "T1: quant"),
        ("track1_tempo", "workplace", "T1: workplace"),
        ("track2_recor", "drones", "T2: drones"),
        ("track2_recor", "law", "T2: law"),
        ("track2_recor", "politics", "T2: politics"),
    ]
    lex_g = [per_dom[t][d]["gate_stats"]["lexical_gate_mean"] for t, d, _ in dom_list]
    sem_g = [per_dom[t][d]["gate_stats"]["semantic_gate_mean"] for t, d, _ in dom_list]
    tmp_g = [per_dom[t][d]["gate_stats"]["temporal_conv_gate_mean"] for t, d, _ in dom_list]
    x_d = np.arange(len(dom_list))

    ax1.bar(x_d, lex_g, label="Lexical Gate ($\\alpha_{lex}$)", color="#64748b", width=0.55)
    ax1.bar(x_d, sem_g, bottom=lex_g, label="Dense Semantic Gate ($\\alpha_{sem}$)", color="#3b82f6", width=0.55)
    ax1.bar(x_d, tmp_g, bottom=np.array(lex_g) + np.array(sem_g), label="Temporal/Conv Gate ($\\alpha_{temp}$)", color="#10b981", width=0.55)

    ax1.set_title("Learned Contextual Gate Allocation Across Domains (Dev)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xticks(x_d)
    ax1.set_xticklabels([lbl for _, _, lbl in dom_list], rotation=15, fontsize=9, fontweight="bold")
    ax1.set_ylabel("Mean Softmax Gate Weight", fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5, loc="upper right")

    plt.tight_layout()
    plt.savefig("experiment_outputs/figures/fig5_training_loss_and_gates.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 6: Per-Domain Performance Heatmap (nDCG@10 across all 6 Domains)
    # -------------------------------------------------------------------------
    hm_methods = [
        "BM25_Base",
        "BM25_Context",
        "Dense_BiEncoder",
        "Ablation_NoTempConvFeats",
        "Ablation_NoHardNegs",
        "Ablation_NoAdaptivePrior",
        "Ablation_NoGating",
        "Proposed_TempConv_GatedRanker",
    ]
    mat = np.zeros((len(hm_methods), len(dom_list)))
    for r_i, m in enumerate(hm_methods):
        for c_i, (t, d, _) in enumerate(dom_list):
            mat[r_i, c_i] = per_dom[t][d]["dev"][m]["nDCG@10"]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(dom_list)))
    ax.set_xticklabels([lbl for _, _, lbl in dom_list], fontsize=9.5, fontweight="bold")
    ax.set_yticks(np.arange(len(hm_methods)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in hm_methods], fontsize=9.5, fontweight="bold")
    ax.set_title("Per-Domain Held-Out Dev nDCG@10 Heatmap Across RETECO Sub-Tracks", fontsize=11.5, fontweight="bold", pad=12)

    for i in range(len(hm_methods)):
        for j in range(len(dom_list)):
            val = mat[i, j]
            text_color = "white" if val > 0.45 else "#0f172a"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=text_color, fontsize=8.5, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="nDCG@10")
    plt.tight_layout()
    plt.savefig("experiment_outputs/figures/fig6_per_domain_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 0: Proposed Architecture Block Diagram (TempConv-GatedRanker)
    # -------------------------------------------------------------------------
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle, bg="#f8fafc", border="#334155", title_col="#0f172a"):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.15",
                             facecolor=bg, edgecolor=border, linewidth=1.6)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.65, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=title_col)
        if subtitle:
            ax.text(x + w/2, y + h*0.30, subtitle, ha="center", va="center", fontsize=7.8, color="#334155")

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.4", color="#475569", lw=1.6))

    ax.text(6.5, 4.92, "Proposed Architecture: TempConv-GatedRanker (Two-Stage Reasoning-Oriented Retrieval)",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#0f172a")

    # Stage 1 inputs
    draw_box(0.3, 2.9, 2.3, 1.4, "Track 1 / Track 2 Input",
             "Temporal Query + Steps\nOR Turn + Dialogue History", bg="#eff6ff", border="#2563eb", title_col="#1e40af")
    draw_box(0.3, 0.7, 2.3, 1.4, "Query Reformulation",
             "Temporal Anchor Boost (T1)\nTurn-Decay Distillation (T2)", bg="#ecfdf5", border="#059669", title_col="#065f46")
    draw_arrow(1.45, 2.9, 1.45, 2.22)

    # Candidate Generation
    draw_box(3.1, 1.7, 2.2, 1.6, "Stage 1: Hybrid Pool",
             "BM25 (k1=0.9, b=0.4) +\nExpanded Query Top-100\n+ Hard Negative Mining", bg="#f1f5f9", border="#475569")
    draw_arrow(2.72, 3.5, 3.05, 2.8)
    draw_arrow(2.72, 1.4, 3.05, 2.2)

    # 3 Parallel Streams in Stage 2
    draw_box(5.9, 3.5, 2.8, 1.05, "Stream 1: Dense Bi-Encoder",
             "Shared Residual GELU-MLP\n[u * v ; |u - v| ; cos(u,v)]", bg="#eff6ff", border="#2563eb", title_col="#1d4ed8")
    draw_box(5.9, 2.0, 2.8, 1.05, "Stream 2: Temp/Conv Network",
             "Year/Ver Overlap, Signal Prod,\nTurn-Decay & Coverage Feats", bg="#ecfdf5", border="#10b981", title_col="#047857")
    draw_box(5.9, 0.5, 2.8, 1.05, "Stream 3: Lexical Skip",
             "Z-Normalized BM25 Base,\nExpanded BM25 & TF-IDF Sim", bg="#f8fafc", border="#64748b", title_col="#334155")

    draw_arrow(5.42, 2.8, 5.82, 4.0)
    draw_arrow(5.42, 2.5, 5.82, 2.5)
    draw_arrow(5.42, 2.2, 5.82, 1.0)

    # Contextual Gate & Adaptive Prior Fusion
    draw_box(9.3, 1.65, 2.0, 1.75, "Contextual Gate\n& Adaptive Prior",
             "Softmax Gate a(q,d)\n+ Bayesian Shrinkage\nN_train / (N_train + tau)", bg="#f5f3ff", border="#7c3aed", title_col="#5b21b6")
    draw_arrow(8.82, 4.0, 9.25, 2.9)
    draw_arrow(8.82, 2.5, 9.25, 2.5)
    draw_arrow(8.82, 1.0, 9.25, 2.1)

    # Output Ranking
    draw_box(11.65, 1.75, 1.15, 1.55, "Final Top-K\nRanking",
             "Pairwise Margin\n+ BCE Loss\n(nDCG@10)", bg="#fef3c7", border="#d97706", title_col="#92400e")
    draw_arrow(11.42, 2.5, 11.60, 2.5)

    plt.tight_layout()
    plt.savefig("experiment_outputs/figures/fig0_architecture_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Successfully generated all 7 figures in experiment_outputs/figures/!")


if __name__ == "__main__":
    main()
