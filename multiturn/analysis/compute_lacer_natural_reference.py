"""Natural-vs-natural reference for compute_lacer_precision_recall: best-match LACER of real papers against other real papers.

Takes the real query papers from the recall side of a precision/recall results file, retrieves each one's k nearest other
real test-period target papers (GRIT cosine, excluding itself), LACER-scores each pair with the query as the reference, and
reports the mean max-LACER. This gives the similarity level real literature reaches against itself.
"""
import os
import argparse
import concurrent.futures as cf
from collections import defaultdict

import numpy as np
import openai
from tqdm import tqdm

import utils
from multiturn.analysis.utils import extract_key_embeddings
from multiturn.analysis.compute_lacer_novelty import score_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_path", required=True, help="Output of compute_lacer_precision_recall (recall-side query ids are reused)")
    ap.add_argument("--data_dir", default="data/prescience-ai", help="Dataset directory (with train/ and test/ subdirectories)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--embeddings_dir", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--prompt_path", default="task_followup_prediction/evaluate/templates/lacer_scoring_prompt_percentile50.110.txt")
    ap.add_argument("--model", default="gpt-5-2025-08-07")
    ap.add_argument("--max_workers", type=int, default=200)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()

    res = utils.load_json(args.results_path)[0][0]
    query_ids = [r["query_id"] for r in res["records"] if r["direction"] == "recall"]
    papers, _, _ = utils.load_corpus(data_dir=args.data_dir, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=False)
    real = {p["corpus_id"]: p for p in papers if "target" in p["roles"]}
    emb = extract_key_embeddings(utils.load_pkl(os.path.join(args.embeddings_dir, "all_papers.grit_embeddings.pkl"))[0])
    pool_ids = [c for c in real if c in emb]
    M = np.stack([emb[c] for c in pool_ids]).astype(np.float32); M /= np.linalg.norm(M, axis=1, keepdims=True)
    Q = np.stack([emb[c] for c in query_ids]).astype(np.float32); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    sims = Q @ M.T
    pos = {c: i for i, c in enumerate(pool_ids)}
    for i, c in enumerate(query_ids):
        sims[i, pos[c]] = -np.inf  # exclude the query itself
    top = np.argsort(-sims, axis=1)[:, :args.k]

    with open(args.prompt_path, encoding="utf-8") as f:
        template = f.read().strip()
    client = openai.OpenAI()
    jobs = [{"client": client, "prompt_template": template, "model": args.model, "query_paper": real[q], "neighbor_paper": real[pool_ids[j]],
             "neighbor_info": {"query_id": q, "neighbor_id": pool_ids[j], "cosine": float(sims[i, j])}}
            for i, q in enumerate(query_ids) for j in top[i]]
    utils.log(f"Scoring {len(jobs)} natural-natural pairs")
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        scored = list(tqdm(ex.map(score_pair, jobs), total=len(jobs), desc="LACER natural reference"))
    per = defaultdict(list)
    for r in scored:
        if r.get("lacer_score") is not None:
            per[r["query_id"]].append(r["lacer_score"])
    v = np.array([max(s) for s in per.values()])
    b = np.random.default_rng(0).choice(v, (2000, len(v))).mean(axis=1)
    summary = {"mean_max_lacer": float(v.mean()), "ci95": float((np.percentile(b, 97.5) - np.percentile(b, 2.5)) / 2), "n": int(len(v))}
    utils.log(f"natural reference: mean max-LACER {v.mean():.2f} ± {summary['ci95']:.2f} (n={len(v)})")
    utils.save_json([{"summary": summary, "pairs": scored}], args.output_path, metadata=utils.update_metadata([], args), overwrite=True)


if __name__ == "__main__":
    main()
