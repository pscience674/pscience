"""Independence baseline for topic-pair growth prediction.

The substantive "no interaction" floor: predicts pair shares as the product of marginal
topic shares (computed from history-period single-topic counts). Beating this on Δshare
captures real co-occurrence dynamics beyond the size prior at the topic level.

Steps:
  1. From the labels JSONL, count history-period papers per topic (single-topic counts,
     each paper contributes once per topic in its label set).
  2. topic_history_share[t] = topic_history_count[t] / Σ_t topic_history_count[t].
  3. unnormalized_score(a, b) = topic_history_share[a] × topic_history_share[b].
  4. Z = Σ_pairs unnormalized_score(a, b) over the filtered pair set.
  5. pred_pair_share(a, b) = unnormalized_score(a, b) / Z.
  6. Project total predicted forecast pair-mass via the inner-split scaling used by
     proportional/mean: total = inner_forecast_pair_total × (|F| / |F_inner|).
  7. predicted_growth(a, b) = pred_pair_share(a, b) × total.
"""

import os
import json
import argparse

import utils
from dataset.corpus.assign_topics import load_topics
from task_topic_pair_growth_prediction.dataset import load_topics_benchmark


def load_topic_labels(path):
    labels = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            labels[r["corpus_id"]] = r["topics"]
    return labels


def main():
    parser = argparse.ArgumentParser(description="Independence baseline (marginal-product, inner-split projected)")
    parser.add_argument("--topics_path", type=str, required=True, help="Path to topic_pairs.vX.json (the pair benchmark)")
    parser.add_argument("--labels_path", type=str, required=True, help="JSONL of per-paper topic labels (for marginal topic counts)")
    parser.add_argument("--single_topics_path", type=str, default="dataset/corpus/topics_list_v5.txt", help="Text file with one topic per line (the topic vocabulary)")
    parser.add_argument("--data_dir", type=str, default="data/prescience-ai", help="Dataset directory (with train/ and test/ subdirectories)")
    parser.add_argument("--inner_history_months", type=int, default=6, help="First N months of history used as inner-history")
    parser.add_argument("--output_dir", type=str, default="data/task_topic_pair_growth_prediction/predictions", help="Output directory")
    args = parser.parse_args()

    utils.log(f"Loading benchmark from {args.topics_path}")
    config, instances, _, _, metadata = load_topics_benchmark(args.topics_path)
    H, F, N = config["history_months"], config["forecast_months"], args.inner_history_months
    assert 1 <= N < H, f"--inner_history_months must be in [1, {H-1}]"
    F_inner = H - N
    history_start, history_end = config["history_start"], config["history_end"]

    topic_vocab = load_topics(args.single_topics_path)
    topic_set = set(topic_vocab)
    utils.log(f"Loaded {len(topic_vocab)} topics from {args.single_topics_path}")

    utils.log(f"Loading topic labels from {args.labels_path}")
    labels = load_topic_labels(args.labels_path)

    utils.log(f"Loading corpus from {args.data_dir}")
    train_papers, _, _ = utils.load_corpus(data_dir=args.data_dir, split="train", embedding_type=None, load_sd2publications=False)
    test_papers, _, _ = utils.load_corpus(data_dir=args.data_dir, split="test", embedding_type=None, load_sd2publications=False)
    papers_by_id = {}
    for p in train_papers + test_papers:
        if p.get("corpus_id") in labels and p["corpus_id"] not in papers_by_id and "target" in (p.get("roles") or []):
            papers_by_id[p["corpus_id"]] = p

    topic_history_count = {t: 0 for t in topic_vocab}
    for cid, paper in papers_by_id.items():
        if not (history_start <= paper["date"] < history_end):
            continue
        paper_topics = set(t for t in labels.get(cid, []) if t in topic_set)
        for t in paper_topics:
            topic_history_count[t] += 1

    total_topic_count = sum(topic_history_count.values())
    if total_topic_count <= 0:
        topic_history_share = {t: 0.0 for t in topic_vocab}
    else:
        topic_history_share = {t: c / total_topic_count for t, c in topic_history_count.items()}
    utils.log(f"Σ topic_history_count = {total_topic_count}; nonzero topics = {sum(1 for v in topic_history_count.values() if v > 0)}")

    unnormalized = []
    for cid, inst in instances:
        a, b = inst["topic_a"], inst["topic_b"]
        unnormalized.append(topic_history_share.get(a, 0.0) * topic_history_share.get(b, 0.0))
    Z = sum(unnormalized)
    utils.log(f"Z (sum of marginal-products over pair set) = {Z:.6g}")

    inner_forecast_pair_total = sum(sum(inst["history_monthly_counts"][N:H]) for _, inst in instances)
    total_predicted = inner_forecast_pair_total * (F / F_inner) if F_inner > 0 else 0.0
    utils.log(f"inner_forecast_pair_total = {inner_forecast_pair_total}; total_predicted_forecast_pairs = {total_predicted:.4f}")

    predictions = []
    for (cid, inst), u in zip(instances, unnormalized):
        share = (u / Z) if Z > 0 else 0.0
        predicted = share * total_predicted
        predictions.append({"cluster_id": cid, "predicted_growth": float(predicted), "gt_growth": inst["gt_growth"]})

    output_path = os.path.join(args.output_dir, "predictions.independence.json")
    utils.log(f"Saving {len(predictions)} predictions to {output_path}")
    utils.save_json(predictions, output_path, metadata=utils.update_metadata(metadata, args), overwrite=True)


if __name__ == "__main__":
    main()
