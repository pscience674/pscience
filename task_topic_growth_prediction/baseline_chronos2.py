"""Chronos-2 pretrained time-series baseline for topic growth prediction (causally safe).

Zero-shot forecast of the next |F| monthly counts per topic with amazon/chronos-2;
predicted growth = sum of the median forecast path. With --covariates, conditions on
two past-covariate series mirroring the momentum baselines' signals: monthly citation
influx and monthly author strength (h-index sum of publishing authors), both computed
strictly from information observable at history_end.
"""

import os
import argparse

import numpy as np
import torch
from chronos import Chronos2Pipeline

import utils
from task_topic_growth_prediction.dataset import load_topics_benchmark

torch.manual_seed(42)


def months_between(date_start, date_end):
    """Return integer month count date_end - date_start for 'YYYY-MM-DD' strings."""
    ys, ms = int(date_start[:4]), int(date_start[5:7])
    ye, me = int(date_end[:4]), int(date_end[5:7])
    return (ye - ys) * 12 + (me - ms)


def build_covariates(membership, papers_by_id, history_start, num_months):
    """Per-cluster monthly citation-influx and author-strength series over the history period.

    Only papers published inside the history window contribute, and only citation-trajectory
    entries up to the end of each history month are read (age = months since publication),
    so every value is observable at history_end.
    """
    citations = {c: np.zeros(num_months, dtype=np.float32) for c in membership}
    authors = {c: np.zeros(num_months, dtype=np.float32) for c in membership}
    for cluster_id, corpus_ids in membership.items():
        for cid in corpus_ids:
            paper = papers_by_id.get(cid)
            if paper is None:
                continue
            pub_month = months_between(history_start, paper["date"])
            if pub_month < 0 or pub_month >= num_months:
                continue
            traj = paper.get("citation_trajectory") or []
            for m in range(pub_month, min(pub_month + len(traj), num_months)):
                age = m - pub_month + 1
                influx = traj[age - 1] - (traj[age - 2] if age >= 2 else 0)
                citations[cluster_id][m] += max(influx, 0)
            authors[cluster_id][pub_month] += sum(a["h_index"] for a in paper["authors"])
    return citations, authors


def main():
    parser = argparse.ArgumentParser(description="Chronos-2 pretrained forecaster baseline")
    parser.add_argument("--topics_path", type=str, required=True, help="Path to topics.vX.json")
    parser.add_argument("--model", type=str, default="amazon/chronos-2", help="Chronos-2 model to use")
    parser.add_argument("--covariates", action="store_true", help="Condition on citation-influx and author-strength past covariates")
    parser.add_argument("--cross_learning", action="store_true", help="Enable Chronos-2 in-context cross-learning across series (history-period series only, causally safe)")
    parser.add_argument("--data_dir", type=str, default="data/prescience-ai", help="Dataset directory (with train/ and test/ subdirectories)")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for inference")
    parser.add_argument("--batch_size", type=int, default=256, help="Series per forward pass")
    parser.add_argument("--output_dir", type=str, default="data/task_topic_growth_prediction/predictions", help="Output directory")
    args = parser.parse_args()

    utils.log(f"Loading benchmark from {args.topics_path}")
    config, instances, membership, _, metadata = load_topics_benchmark(args.topics_path)
    F, H = config["forecast_months"], config["history_months"]
    history_start = config["history_start"]

    if args.covariates:
        utils.log(f"Loading corpus from {args.data_dir} to build covariates")
        train_papers, _, _ = utils.load_corpus(data_dir=args.data_dir, split="train", embedding_type=None, load_sd2publications=False)
        test_papers, _, _ = utils.load_corpus(data_dir=args.data_dir, split="test", embedding_type=None, load_sd2publications=False)
        papers_by_id = {}
        for p in train_papers + test_papers:
            if p["corpus_id"] not in papers_by_id:
                papers_by_id[p["corpus_id"]] = p
        utils.log(f"Building covariate series for {len(membership)} clusters over {H} months")
        citations, authors = build_covariates(membership, papers_by_id, history_start, H)

    utils.log(f"Loading Chronos-2 model {args.model} on {args.device}")
    pipeline = Chronos2Pipeline.from_pretrained(args.model, device_map=args.device, torch_dtype=torch.float32)

    inputs = []
    for cid, inst in instances:
        target = np.asarray(inst["history_monthly_counts"], dtype=np.float32)
        if args.covariates:
            inputs.append({"target": target, "past_covariates": {"citations": citations[cid], "author_strength": authors[cid]}})
        else:
            inputs.append(target)

    utils.log(f"Forecasting {F} months for {len(inputs)} topics (covariates={args.covariates}, cross_learning={args.cross_learning})")
    quantiles, _ = pipeline.predict_quantiles(inputs, prediction_length=F, quantile_levels=[0.5], batch_size=args.batch_size, cross_learning=args.cross_learning)

    predictions = []
    for (cid, inst), q in zip(instances, quantiles):
        median_path = q[0, :, 0].cpu().numpy()
        predicted = float(np.sum(np.maximum(median_path, 0.0)))
        predictions.append({"cluster_id": cid, "predicted_growth": predicted, "gt_growth": inst["gt_growth"]})

    suffix = "chronos2_covariates" if args.covariates else "chronos2"
    if args.cross_learning:
        suffix += "_crosslearn"
    output_path = os.path.join(args.output_dir, f"predictions.{suffix}.json")
    utils.log(f"Saving {len(predictions)} predictions to {output_path}")
    utils.save_json(predictions, output_path, metadata=utils.update_metadata(metadata, args), overwrite=True)


if __name__ == "__main__":
    main()
