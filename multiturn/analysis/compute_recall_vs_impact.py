"""Recall-side best-match LACER for a large sample of real test papers with fixed-horizon citation counts.

Samples real target papers that have at least --citation_months months of citation data, reuses recall-side scores already
computed by compute_lacer_precision_recall for any sampled paper, retrieves each new paper's k nearest synthetic papers
(GRIT cosine), LACER-scores each pair (real paper as reference), and saves per-paper best-match LACER with citation counts.
"""
import os
import json
import random
import argparse
import concurrent.futures as cf
from collections import defaultdict

import numpy as np
import openai
from tqdm import tqdm

import utils
from multiturn.analysis.utils import extract_key_embeddings
from multiturn.analysis.compute_lacer_novelty import score_pair
from multiturn.analysis.compute_lacer_precision_recall import nearest, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/prescience-ai", help="Dataset directory (with train/ and test/ subdirectories)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--embeddings_dir", required=True)
    ap.add_argument("--synthetic_dir", required=True)
    ap.add_argument("--existing_results", default=None, help="compute_lacer_precision_recall output to reuse recall-side scores from")
    ap.add_argument("--n_total", type=int, default=1000)
    ap.add_argument("--citation_months", type=int, default=8)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt_path", default="task_followup_prediction/evaluate/templates/lacer_scoring_prompt_percentile50.110.txt")
    ap.add_argument("--model", default="gpt-5-2025-08-07")
    ap.add_argument("--max_workers", type=int, default=200)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()

    papers, _, _ = utils.load_corpus(data_dir=args.data_dir, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=False)
    m = args.citation_months
    real = {p["corpus_id"]: p for p in papers if "target" in p["roles"] and len(p.get("citation_trajectory") or []) >= m}
    real_emb = extract_key_embeddings(utils.load_pkl(os.path.join(args.embeddings_dir, "all_papers.grit_embeddings.pkl"))[0])
    syn_papers, _ = utils.load_json(os.path.join(args.synthetic_dir, "all_papers.json"))
    syn = {p["corpus_id"]: p for p in syn_papers if "synthetic" in p["roles"]}
    syn_emb = extract_key_embeddings(utils.load_pkl(os.path.join(args.synthetic_dir, "all_papers.grit_embeddings.pkl"))[0])
    syn_ids = [c for c in syn if c in syn_emb]

    reuse = {}
    if args.existing_results:
        res = utils.load_json(args.existing_results)[0][0]
        reuse = {r["query_id"]: r["max_lacer"] for r in res["records"] if r["direction"] == "recall" and r["query_id"] in real and r["max_lacer"] is not None}
    eligible = sorted(c for c in real if c in real_emb and c not in reuse)
    rng = random.Random(args.seed)
    new_ids = rng.sample(eligible, max(0, args.n_total - len(reuse)))
    utils.log(f"{len(real)} eligible papers; reusing {len(reuse)}, scoring {len(new_ids)} new")

    nb = nearest(new_ids, real_emb, syn_ids, pool(syn_ids, syn_emb), args.k)
    with open(args.prompt_path, encoding="utf-8") as f:
        template = f.read().strip()
    client = openai.OpenAI()
    jobs = [{"client": client, "prompt_template": template, "model": args.model, "query_paper": real[q], "neighbor_paper": syn[n],
             "neighbor_info": {"query_id": q, "neighbor_id": n, "cosine": s}} for q, neigh in nb.items() for n, s in neigh]
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        scored = list(tqdm(ex.map(score_pair, jobs), total=len(jobs), desc="LACER recall vs impact"))
    best = defaultdict(list)
    for r in scored:
        if r.get("lacer_score") is not None:
            best[r["query_id"]].append(r["lacer_score"])
    rows = [{"corpus_id": c, "max_lacer": v, "citations": real[c]["citation_trajectory"][m - 1], "reused": True} for c, v in reuse.items()]
    rows += [{"corpus_id": c, "max_lacer": max(v), "citations": real[c]["citation_trajectory"][m - 1], "reused": False} for c, v in best.items()]
    utils.log(f"Saved {len(rows)} papers")
    utils.save_json(rows, args.output_path, metadata=utils.update_metadata([], args), overwrite=True)


if __name__ == "__main__":
    main()
