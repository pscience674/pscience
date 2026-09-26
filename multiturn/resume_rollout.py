"""Rebuild a consistent rollout state from a (possibly partially written) checkpoint so simulate.py can continue.

A preempted rollout can leave all_papers.json complete but the embeddings pickle truncated (it is written after the
JSON). This script takes the checkpoint's all_papers.json, reuses the base corpus embeddings for pre-rollout papers,
re-embeds the remaining (synthetic) papers with the same GRIT settings, rebuilds sd2publications from the paper records,
and writes a state directory that simulate.py loads in local-directory mode (it continues from the latest paper date).

Usage: python -m multiturn.resume_rollout --checkpoint_dir CKPT --base_embeddings BASE.pkl --output_dir STATE
Then:  python -m multiturn.simulate --data_dir STATE --embeddings_dir STATE --depth <remaining days> ...
"""
import os
import argparse
from collections import defaultdict

import utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--base_embeddings", required=True, help="Pre-rollout corpus embeddings pickle (e.g. train all_papers.grit_embeddings.pkl)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--embedding_type", default="grit")
    args = ap.parse_args()

    papers, meta = utils.load_json(os.path.join(args.checkpoint_dir, "all_papers.json"))
    base, _ = utils.load_pkl(args.base_embeddings)
    missing = [p for p in papers if p["corpus_id"] not in base]
    utils.log(f"{len(papers)} papers; {len(missing)} need embeddings (synthetic: {sum('synthetic' in p['roles'] for p in missing)})")
    embs = utils.embed_on_gpus([utils.get_title_abstract_string(p) for p in missing], args.embedding_type, cache_model=True, return_dict=True)
    emb = {p["corpus_id"]: base[p["corpus_id"]] for p in papers if p["corpus_id"] in base}
    emb.update({p["corpus_id"]: e for p, e in zip(missing, embs)})
    assert len(emb) == len(papers), "embedding coverage mismatch"

    sd2publications = defaultdict(list)
    for p in sorted(papers, key=lambda x: x["date"]):
        for a in p.get("authors") or []:
            sd2publications[a["author_id"]].append(p["corpus_id"])

    os.makedirs(args.output_dir, exist_ok=True)
    utils.save_json(papers, os.path.join(args.output_dir, "all_papers.json"), metadata=meta, overwrite=True)
    utils.save_pkl(emb, os.path.join(args.output_dir, f"all_papers.{args.embedding_type}_embeddings.pkl"), overwrite=True)
    utils.save_json(dict(sd2publications), os.path.join(args.output_dir, "sd2publications.json"), overwrite=True)
    syn = [p["date"] for p in papers if "synthetic" in p["roles"]]
    utils.log(f"Resume state written to {args.output_dir}; last synthetic date {max(syn) if syn else None}")


if __name__ == "__main__":
    main()
