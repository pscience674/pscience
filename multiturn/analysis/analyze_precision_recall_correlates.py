"""Correlate LACER distance (10 - best-match LACER) from compute_lacer_precision_recall with paper properties.

Recall side (real query papers): citations at a fixed horizon (--citation_months; papers with enough months of data), number of influential references,
max author h-index, and number of authors. Precision side (synthetic query papers): number of influential references,
number of authors, and max author publication-history length (synthetic authors carry no h-index).
Reports Spearman correlations with bootstrap 95% CIs and a binned plot per property.
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import utils


def feats_real(p, months=12):
    a = p.get("authors") or []
    traj = p.get("citation_trajectory") or []
    return {f"{months}-month citations": traj[months - 1] if len(traj) >= months else None, "# influential refs": len(p.get("key_references") or []),
            "max author h-index": max((x.get("h_index") or 0 for x in a), default=None), "# authors": len(a)}


def feats_syn(p):
    a = p.get("authors") or []
    return {"# influential refs": len(p.get("key_references") or []), "# authors": len(a),
            "max author history length": max((len(x.get("publication_history") or []) for x in a), default=None)}


def spearman_ci(x, y, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(x); vals = []
    for _ in range(B):
        i = rng.integers(0, n, n); vals.append(spearmanr(x[i], y[i]).statistic)
    return float(spearmanr(x, y).statistic), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_path", required=True)
    ap.add_argument("--data_dir", default="data/prescience-ai", help="Dataset directory (with train/ and test/ subdirectories)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--synthetic_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--citation_months", type=int, default=12, help="Citation horizon (months) for the recall-side citation correlate")
    args = ap.parse_args()
    res = utils.load_json(args.results_path)[0][0]
    real_papers, _, _ = utils.load_corpus(data_dir=args.data_dir, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=False)
    real = {p["corpus_id"]: p for p in real_papers}
    syn = {p["corpus_id"]: p for p in utils.load_json(os.path.join(args.synthetic_dir, "all_papers.json"))[0]}
    out = {}
    os.makedirs(args.output_dir, exist_ok=True)
    for direction, src, fn in (("recall", real, lambda p: feats_real(p, args.citation_months)), ("precision", syn, feats_syn)):
        recs = [r for r in res["records"] if r["direction"] == direction and r["max_lacer"] is not None]
        dist = np.array([10 - r["max_lacer"] for r in recs], dtype=float)
        F = [fn(src[r["query_id"]]) for r in recs]
        out[direction] = {}
        names = list(F[0].keys())
        fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 2.8), sharey=True)
        for ax, name in zip(axes, names):
            v = np.array([f[name] if f[name] is not None else np.nan for f in F], dtype=float); m = ~np.isnan(v)
            x, y = v[m], dist[m]
            rho, lo, hi = spearman_ci(x, y)
            out[direction][name] = {"spearman": rho, "ci95": [lo, hi], "n": int(m.sum())}
            utils.log(f"{direction:9s} {name:26s} rho={rho:+.2f} [{lo:+.2f}, {hi:+.2f}] n={m.sum()}")
            edges = np.unique(np.percentile(x, [0, 25, 50, 75, 100]))
            idx = np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges) - 2)
            mids = [np.mean(y[idx == b]) for b in range(len(edges) - 1)]
            ses = [np.std(y[idx == b]) / np.sqrt(max((idx == b).sum(), 1)) for b in range(len(edges) - 1)]
            labels = [f"{edges[b]:.0f}-{edges[b + 1]:.0f}" for b in range(len(edges) - 1)]
            ax.errorbar(range(len(mids)), mids, yerr=ses, marker="o", capsize=2)
            ax.set_xticks(range(len(mids))); ax.set_xticklabels(labels, fontsize=7, rotation=30)
            ax.set_title(f"{name}\n$\\rho$={rho:+.2f} [{lo:+.2f}, {hi:+.2f}], n={m.sum()}", fontsize=8)
        axes[0].set_ylabel("LACER distance (10 - best match)", fontsize=8)
        fig.suptitle(f"{direction.capitalize()} side ({'real' if direction == 'recall' else 'synthetic'} query papers)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"lacer_distance_correlates_{direction}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    utils.save_json([out], os.path.join(args.output_dir, "lacer_distance_correlates.json"), metadata=utils.update_metadata([], args), overwrite=True)


if __name__ == "__main__":
    main()
