"""LACER-based precision and recall of a synthetic corpus against the real test-period corpus.

Precision (synthetic -> real): for n sampled synthetic papers per month, retrieve their k nearest real test-period target
papers (GRIT cosine, whole test period) and LACER-score each pair; the paper's precision score is the max over the k.
Recall (real -> synthetic): for n sampled real target papers per month, retrieve their k nearest synthetic papers and take
the max LACER score. In both directions the real paper is the LACER reference and the synthetic paper is the generation.
All pairs from both directions are scored in a single worker pool.
"""
import os
import random
import argparse
import concurrent.futures as cf
from collections import defaultdict

import numpy as np
import openai
from tqdm import tqdm

import utils
from multiturn.analysis.utils import extract_key_embeddings, get_bucket_start
from multiturn.analysis.compute_lacer_novelty import score_pair


def sample_per_month(papers, n, rng):
    buckets = defaultdict(list)
    for p in papers:
        buckets[get_bucket_start(p["date"], 30)].append(p)
    return {b: (rng.sample(ps, n) if len(ps) > n else ps) for b, ps in sorted(buckets.items())}


def nearest(query_ids, query_emb, pool_ids, pool_matrix, k):
    q = np.stack([query_emb[c] for c in query_ids]).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sims = q @ pool_matrix.T
    top = np.argsort(-sims, axis=1)[:, :k]
    return {cid: [(pool_ids[j], float(sims[i, j])) for j in top[i]] for i, cid in enumerate(query_ids)}


def pool(ids, emb):
    m = np.stack([emb[c] for c in ids]).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description="LACER precision/recall of a synthetic corpus vs. the real test corpus")
    ap.add_argument("--data_dir", default="data/prescience-ai", help="Dataset directory (with train/ and test/ subdirectories)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--embeddings_dir", required=True, help="Real corpus embeddings dir (all_papers.grit_embeddings.pkl)")
    ap.add_argument("--synthetic_dir", required=True, help="Rollout dir with all_papers.json and all_papers.grit_embeddings.pkl")
    ap.add_argument("--n", type=int, default=25, help="Query papers sampled per month in each direction")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt_path", default="task_followup_prediction/evaluate/templates/lacer_scoring_prompt_percentile50.110.txt")
    ap.add_argument("--model", default="gpt-5-2025-08-07")
    ap.add_argument("--max_workers", type=int, default=200)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    real_papers, _, _ = utils.load_corpus(data_dir=args.data_dir, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=False)
    real = {p["corpus_id"]: p for p in real_papers if "target" in p["roles"]}
    real_emb = extract_key_embeddings(utils.load_pkl(os.path.join(args.embeddings_dir, "all_papers.grit_embeddings.pkl"))[0])
    syn_papers, _ = utils.load_json(os.path.join(args.synthetic_dir, "all_papers.json"))
    syn = {p["corpus_id"]: p for p in syn_papers if "synthetic" in p["roles"]}
    syn_emb = extract_key_embeddings(utils.load_pkl(os.path.join(args.synthetic_dir, "all_papers.grit_embeddings.pkl"))[0])
    real_ids = [c for c in real if c in real_emb]
    syn_ids = [c for c in syn if c in syn_emb]
    utils.log(f"{len(real_ids)} real targets, {len(syn_ids)} synthetic papers with embeddings")

    syn_q = sample_per_month([syn[c] for c in syn_ids], args.n, rng)
    real_q = sample_per_month([real[c] for c in real_ids], args.n, rng)
    prec_nb = nearest([p["corpus_id"] for ps in syn_q.values() for p in ps], syn_emb, real_ids, pool(real_ids, real_emb), args.k)
    rec_nb = nearest([p["corpus_id"] for ps in real_q.values() for p in ps], real_emb, syn_ids, pool(syn_ids, syn_emb), args.k)

    with open(args.prompt_path, encoding="utf-8") as f:
        template = f.read().strip()
    client = openai.OpenAI()
    jobs = []
    for direction, nb, qsrc, nsrc in (("precision", prec_nb, syn, real), ("recall", rec_nb, real, syn)):
        for qid, neigh in nb.items():
            for nid, sim in neigh:
                ref, gen = (nsrc[nid], qsrc[qid]) if direction == "precision" else (qsrc[qid], nsrc[nid])
                jobs.append({"client": client, "prompt_template": template, "model": args.model, "query_paper": ref, "neighbor_paper": gen,
                             "neighbor_info": {"direction": direction, "query_id": qid, "neighbor_id": nid, "cosine": sim}})
    utils.log(f"Scoring {len(jobs)} pairs with {args.max_workers} workers")
    scored = []
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for r in tqdm(ex.map(score_pair, jobs), total=len(jobs), desc="LACER precision/recall"):
            scored.append(r)

    per_query = defaultdict(list)
    for r in scored:
        per_query[(r["direction"], r["query_id"])].append(r)
    records = []
    for (direction, qid), rs in per_query.items():
        vals = [r["lacer_score"] for r in rs if r.get("lacer_score") is not None]
        q = (syn if direction == "precision" else real)[qid]
        records.append({"direction": direction, "query_id": qid, "date": q["date"], "bucket": get_bucket_start(q["date"], 30),
                        "max_lacer": max(vals) if vals else None, "n_scored": len(vals), "neighbors": rs})
    summary = {}
    for d in ("precision", "recall"):
        v = np.array([r["max_lacer"] for r in records if r["direction"] == d and r["max_lacer"] is not None])
        b = np.random.default_rng(0).choice(v, (2000, len(v))).mean(axis=1)
        summary[d] = {"mean_max_lacer": float(v.mean()), "ci95": float((np.percentile(b, 97.5) - np.percentile(b, 2.5)) / 2), "n": int(len(v))}
        utils.log(f"{d}: mean max-LACER {v.mean():.2f} ± {summary[d]['ci95']:.2f} (n={len(v)})")
    utils.save_json([{"summary": summary, "records": records}], args.output_path, metadata=utils.update_metadata([], args), overwrite=True)


if __name__ == "__main__":
    main()
