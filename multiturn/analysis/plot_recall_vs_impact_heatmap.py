"""Decile-by-decile heatmap of recall-side best-match LACER vs. fixed-horizon citations (from compute_recall_vs_impact).

Both variables are converted to deciles by rank with seeded random tie-breaking (citations and best-match LACER are both
heavily tied). Each cell shows the number of papers (white-to-blue color scale).
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import utils


def rank_deciles(v, rng, q=10):
    order = np.lexsort((rng.random(len(v)), v))
    ranks = np.empty(len(v), dtype=int); ranks[order] = np.arange(len(v))
    return (ranks * q) // len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--citation_months", type=int, default=8)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()
    rows = utils.load_json(args.input_path)[0]
    lacer = np.array([r["max_lacer"] for r in rows], dtype=float)
    cites = np.array([r["citations"] for r in rows], dtype=float)
    rng = np.random.default_rng(args.seed)
    rho = spearmanr(cites, lacer).statistic
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(rows), len(rows)); boots.append(spearmanr(cites[i], lacer[i]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    q = args.bins
    cx, ly = rank_deciles(cites, rng, q), rank_deciles(lacer, rng, q)
    counts = np.zeros((q, q))
    for a, b in zip(ly, cx):
        counts[a, b] += 1
    ratio = counts / (len(rows) / (q * q))
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    vmin, vmax = counts.min(), counts.max()
    im = ax.imshow(counts, origin="lower", cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
    if q <= 6:
        for a_ in range(q):
            for b_ in range(q):
                c = counts[a_, b_]
                ax.text(b_, a_, f"{int(c)}", ha="center", va="center", fontsize=8, color="white" if c > vmin + 0.6 * (vmax - vmin) else "black")
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("Papers per cell", fontsize=8)
    ax.set_xticks(range(q)); ax.set_xticklabels([str(i + 1) for i in range(q)], fontsize=8)
    ax.set_yticks(range(q)); ax.set_yticklabels([str(i + 1) for i in range(q)], fontsize=8)
    unit = {10: "decile", 5: "quintile"}.get(q, "bin")
    ax.set_xlabel(f"{args.citation_months}-month citation {unit} (low $\\rightarrow$ high)", fontsize=9)
    ax.set_ylabel(f"Recall best-match LACER {unit}\n(low $\\rightarrow$ high coverage)", fontsize=9)
    ax.set_title(f"Spearman $\\rho$ = {rho:+.2f} [{lo:+.2f}, {hi:+.2f}], n = {len(rows)}", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output_path, dpi=300, bbox_inches="tight")
    utils.log(f"rho={rho:+.3f} [{lo:+.3f}, {hi:+.3f}] n={len(rows)}; saved {args.output_path}")


if __name__ == "__main__":
    main()
