"""Combined three-panel prior work selection analysis figure with a shared legend.

Panels: (a) nDCG vs. mean prior-publication count of the target's authors, (b) nDCG vs. number of influential references,
(c) R-precision vs. team size. Computed from per-instance scored predictions and the corpus; error bars are standard errors.
"""
import os
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import utils

PANELS = [
    ("exp", "ndcg", "Mean author experience (# prior papers)", "nDCG", [0, 4, 7, 11, 21, 51, np.inf], ["1-3", "4-6", "7-10", "11-20", "21-50", "51+"]),
    ("refs", "ndcg", "# influential references", "nDCG", [1, 2, 3, 4, 5, 6, 11], ["1", "2", "3", "4", "5", "6-10"]),
    ("team", "precision", "Team size (# authors)", "R-precision", [1, 2, 3, 4, 5, 7, np.inf], ["1", "2", "3", "4", "5-6", "7+"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/prescience-ai", help="Dataset directory (with train/ and test/ subdirectories)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--baseline", nargs=3, action="append", metavar=("LABEL", "EVAL_PATH", "COLOR"), required=True)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()

    papers, _, _ = utils.load_corpus(data_dir=args.data_dir, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=False)
    by_id = {p["corpus_id"]: p for p in papers}

    def feats(cid):
        p = by_id[cid]; a = p.get("authors") or []
        hist = [len(x.get("publication_history") or []) for x in a]
        return {"exp": float(np.mean(hist)) if hist else 0.0, "refs": len(p.get("key_references") or []), "team": len(a)}

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    handles = []
    for label, path, color in args.baseline:
        per = utils.load_json(path)[0]["per_instance"]
        F = {r["corpus_id"]: feats(r["corpus_id"]) for r in per if r["corpus_id"] in by_id}
        for ax, (fk, mk, xlabel, ylabel, edges, blabels) in zip(axes, PANELS):
            buckets = defaultdict(list)
            for r in per:
                if r["corpus_id"] not in F:
                    continue
                v = F[r["corpus_id"]][fk]
                b = np.searchsorted(edges, v, side="right") - 1
                if 0 <= b < len(blabels):
                    buckets[b].append(r[mk])
            xs = [b for b in range(len(blabels)) if len(buckets[b]) >= 20]
            ys = [np.mean(buckets[b]) for b in xs]
            es = [np.std(buckets[b]) / np.sqrt(len(buckets[b])) for b in xs]
            h = ax.errorbar(xs, ys, yerr=es, marker="o", ms=3.5, lw=1.6, capsize=2, color=color, label=label)
            ax.set_xticks(range(len(blabels))); ax.set_xticklabels(blabels)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True, ls="--", alpha=0.3)
        handles.append(h)
    for ax, t in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.set_title(t, fontsize=9, loc="left")
    fig.legend(handles, [b[0] for b in args.baseline], loc="upper center", ncol=len(args.baseline), frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fig.savefig(args.output_path, dpi=300, bbox_inches="tight")
    utils.log(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()
