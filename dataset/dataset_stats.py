"""Compute the paper's dataset-statistics table (tables/dataset.tex fields) for a corpus with train/ and test/ splits.

Usage: python -m dataset.dataset_stats --corpus_root data/prescience-ai --name ai
Writes data/iclr_results/dataset_stats_<name>.json.
"""
import os
import json
import argparse
import statistics as st

import utils


def split_stats(path):
    papers, _ = utils.load_json(os.path.join(path, "all_papers.json"))
    targets = [p for p in papers if "target" in p["roles"]]
    words = [len(((p.get("title") or "") + " " + (p.get("abstract") or "")).split()) for p in targets]
    refs = [len(p.get("key_references") or []) for p in targets]
    nauth = [len(p.get("authors") or []) for p in targets]
    hist = [len(a.get("publication_history") or []) for p in targets for a in p.get("authors") or []]
    c12 = [p["citation_trajectory"][11] for p in targets if len(p.get("citation_trajectory") or []) >= 12]
    authors = {a["author_id"] for p in targets for a in p.get("authors") or []}
    return {"target_papers": len(targets), "all_papers": len(papers), "avg_words": st.mean(words), "avg_infl_refs": st.mean(refs),
            "median_infl_refs": st.median(refs), "unique_authors": len(authors), "avg_authors": st.mean(nauth), "avg_author_hist": st.mean(hist),
            "median_author_hist": st.median(hist), "avg_citations_12m": st.mean(c12) if c12 else None, "n_with_12m_citations": len(c12),
            "date_range": [min(p["date"] for p in targets), max(p["date"] for p in targets)],
            "_ids": [p["corpus_id"] for p in papers], "_authors": sorted(authors)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_root", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    res = {s: split_stats(os.path.join(args.corpus_root, s)) for s in ("train", "test")}
    res["all"] = {"target_papers": res["train"]["target_papers"] + res["test"]["target_papers"],
                  "all_papers": len(set(res["train"]["_ids"]) | set(res["test"]["_ids"])),
                  "unique_authors": len(set(res["train"]["_authors"]) | set(res["test"]["_authors"]))}
    for s in ("train", "test"):
        res[s].pop("_ids"); res[s].pop("_authors")
    os.makedirs("data/iclr_results", exist_ok=True)
    out = f"data/iclr_results/dataset_stats_{args.name}.json"
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
