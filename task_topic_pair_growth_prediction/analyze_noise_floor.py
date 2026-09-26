"""Three diagnostic analyses for the topic-pair growth task.

(1) Bootstrap reproducibility ceiling on Δshare:
    Random split of labeled papers into halves A and B, K times. Compute Δshare on each half
    independently (over the same filtered pair set), report R² / Pearson / Spearman of
    Δshare_A vs Δshare_B. Median ± IQR over K splits is the irreducible-noise upper bound on
    what any model can achieve.

(2) Within-history-vs-across-window trend continuation:
    ΔshareH (late_history_share − early_history_share, intra-history shift) vs
    ΔshareF (gt_share − history_share, what models predict). Shared variance estimates the
    fraction of forecast-Δshare attributable to the inner-history trend continuing forward
    — the suspected indexing-lag-confounded portion.

(3) Cross-model agreement on pairs (GPT-5.4 vs Claude on 530-paper overlap):
    Build pair counts twice from the two labelers on the same papers; report pair-set Jaccard
    plus Pearson on log(history_pair_count) and Pearson on Δshare over shared pairs.

All three operate on the existing topic_pairs.v5.stable.json filtered pair set so results are
directly comparable to the headline baseline table.
"""

import argparse
import json
import random
import numpy as np

from scipy.stats import spearmanr
from sklearn.metrics import r2_score

import utils
from task_topic_pair_growth_prediction.dataset import load_topics_benchmark
from task_topic_pair_growth_prediction.compute_topic_pairs import SEP, make_pair_id, load_topic_labels


def safe_corrcoef(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def safe_spearman(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(spearmanr(a, b).correlation)


def normalize_to_share(values, fallback_n):
    clipped = np.maximum(values, 0.0)
    s = clipped.sum()
    if s <= 0:
        return np.full_like(clipped, 1.0 / fallback_n)
    return clipped / s


def aggregate_pair_counts(papers_with_topics, period_start, period_end, pair_id_set, topic_set):
    """For papers in [period_start, period_end), accumulate {pair_id: count} restricted to pair_id_set."""
    counts = {p: 0 for p in pair_id_set}
    for cid, date, topics in papers_with_topics:
        if not (period_start <= date < period_end):
            continue
        ts = sorted(set(t for t in topics if t in topic_set))
        if len(ts) < 2:
            continue
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                pid = make_pair_id(ts[i], ts[j])
                if pid in pair_id_set:
                    counts[pid] += 1
    return counts


def vectorize(counts, ordered_pair_ids):
    return np.array([counts[p] for p in ordered_pair_ids], dtype=np.float64)


def bootstrap_ceiling(papers_with_topics, ordered_pair_ids, pair_id_set, topic_set, history_start, history_end, forecast_end, n_pairs, K, seed):
    """K random splits of the paper population into halves A/B. For each, compute pair-Δshare on each half independently (over the same filtered pair set) and correlate. Returns list of dicts."""
    rng = random.Random(seed)
    results = []
    for k in range(K):
        idxs = list(range(len(papers_with_topics)))
        rng.shuffle(idxs)
        half = len(idxs) // 2
        a_idx, b_idx = set(idxs[:half]), set(idxs[half:])
        papers_a = [papers_with_topics[i] for i in a_idx]
        papers_b = [papers_with_topics[i] for i in b_idx]

        history_a = aggregate_pair_counts(papers_a, history_start, history_end, pair_id_set, topic_set)
        gt_a      = aggregate_pair_counts(papers_a, history_end, forecast_end, pair_id_set, topic_set)
        history_b = aggregate_pair_counts(papers_b, history_start, history_end, pair_id_set, topic_set)
        gt_b      = aggregate_pair_counts(papers_b, history_end, forecast_end, pair_id_set, topic_set)

        h_a = normalize_to_share(vectorize(history_a, ordered_pair_ids), n_pairs)
        g_a = normalize_to_share(vectorize(gt_a,      ordered_pair_ids), n_pairs)
        h_b = normalize_to_share(vectorize(history_b, ordered_pair_ids), n_pairs)
        g_b = normalize_to_share(vectorize(gt_b,      ordered_pair_ids), n_pairs)

        delta_a = g_a - h_a
        delta_b = g_b - h_b

        results.append({"split": k, "r2": float(r2_score(delta_b, delta_a)), "pearson": safe_corrcoef(delta_a, delta_b), "spearman": safe_spearman(delta_a, delta_b)})
    return results


def trend_continuation(papers_with_topics, ordered_pair_ids, pair_id_set, topic_set, history_start, history_end, forecast_end, n_pairs):
    """Compare intra-history Δshare (late − early halves of history) against forecast Δshare (gt − history)."""
    ys, ms = int(history_start[:4]), int(history_start[5:7])
    ye, me = int(history_end[:4]), int(history_end[5:7])
    H_months = (ye - ys) * 12 + (me - ms)
    half = H_months // 2
    mid_total = ms - 1 + half
    history_mid = f"{ys + mid_total // 12:04d}-{mid_total % 12 + 1:02d}-01"
    utils.log(f"[trend] history_mid = {history_mid}; H_early=[{history_start},{history_mid}); H_late=[{history_mid},{history_end})")

    early   = aggregate_pair_counts(papers_with_topics, history_start, history_mid, pair_id_set, topic_set)
    late    = aggregate_pair_counts(papers_with_topics, history_mid,   history_end, pair_id_set, topic_set)
    history = aggregate_pair_counts(papers_with_topics, history_start, history_end, pair_id_set, topic_set)
    forecast = aggregate_pair_counts(papers_with_topics, history_end,  forecast_end, pair_id_set, topic_set)

    e = normalize_to_share(vectorize(early,    ordered_pair_ids), n_pairs)
    l = normalize_to_share(vectorize(late,     ordered_pair_ids), n_pairs)
    h = normalize_to_share(vectorize(history,  ordered_pair_ids), n_pairs)
    f = normalize_to_share(vectorize(forecast, ordered_pair_ids), n_pairs)

    delta_H = l - e
    delta_F = f - h
    return {"r2_predict_F_from_H": float(r2_score(delta_F, delta_H)), "pearson": safe_corrcoef(delta_H, delta_F), "spearman": safe_spearman(delta_H, delta_F), "frac_variance_explained": (safe_corrcoef(delta_H, delta_F) or 0.0) ** 2}


def cross_labeler_agreement(gpt_labels, claude_labels, papers_by_id, topic_set, history_start, history_end, forecast_end, min_pair_count_filter):
    """Restrict to papers labeled by BOTH GPT and Claude. Build pair counts per labeler, report agreement metrics."""
    overlap_ids = [cid for cid in claude_labels if cid in gpt_labels and cid in papers_by_id]
    utils.log(f"[xlabel] overlap papers: {len(overlap_ids)}")

    def papers_with_topics_for(label_dict):
        return [(cid, papers_by_id[cid]["date"], label_dict.get(cid, [])) for cid in overlap_ids]

    pwt_g = papers_with_topics_for(gpt_labels)
    pwt_c = papers_with_topics_for(claude_labels)

    # Universal pair set: any pair appearing in either labeler in either period (after a small filter, since n=530 is tiny)
    def all_pairs_with_min(pwt, k):
        h, f_ = {}, {}
        for cid, date, topics in pwt:
            ts = sorted(set(t for t in topics if t in topic_set))
            if len(ts) < 2:
                continue
            in_h = history_start <= date < history_end
            in_f = history_end <= date < forecast_end
            if not (in_h or in_f):
                continue
            for i in range(len(ts)):
                for j in range(i + 1, len(ts)):
                    pid = make_pair_id(ts[i], ts[j])
                    if in_h:
                        h[pid] = h.get(pid, 0) + 1
                    else:
                        f_[pid] = f_.get(pid, 0) + 1
        keep = {p for p in (set(h) | set(f_)) if h.get(p, 0) >= k or f_.get(p, 0) >= k}
        return h, f_, keep

    h_g, f_g, keep_g = all_pairs_with_min(pwt_g, min_pair_count_filter)
    h_c, f_c, keep_c = all_pairs_with_min(pwt_c, min_pair_count_filter)
    utils.log(f"[xlabel] GPT pairs ≥{min_pair_count_filter}: {len(keep_g)}; Claude: {len(keep_c)}")
    inter = keep_g & keep_c
    union = keep_g | keep_c
    jaccard = len(inter) / max(len(union), 1)
    utils.log(f"[xlabel] pair-set Jaccard (≥{min_pair_count_filter}-filter): {jaccard:.4f}; |inter|={len(inter)}, |union|={len(union)}")

    if len(inter) < 2:
        return {"jaccard": jaccard, "n_shared_pairs": len(inter), "log_history_count_pearson": None, "delta_share_pearson": None, "delta_share_r2": None}

    shared = sorted(inter)
    n = len(shared)
    g_h = np.array([h_g.get(p, 0) for p in shared], dtype=np.float64)
    g_f = np.array([f_g.get(p, 0) for p in shared], dtype=np.float64)
    c_h = np.array([h_c.get(p, 0) for p in shared], dtype=np.float64)
    c_f = np.array([f_c.get(p, 0) for p in shared], dtype=np.float64)

    g_h_share = normalize_to_share(g_h, n)
    g_f_share = normalize_to_share(g_f, n)
    c_h_share = normalize_to_share(c_h, n)
    c_f_share = normalize_to_share(c_f, n)
    delta_g = g_f_share - g_h_share
    delta_c = c_f_share - c_h_share

    return {"jaccard": jaccard, "n_shared_pairs": len(inter), "log_history_count_pearson": safe_corrcoef(np.log1p(g_h), np.log1p(c_h)), "delta_share_pearson": safe_corrcoef(delta_g, delta_c), "delta_share_r2": float(r2_score(delta_c, delta_g))}


def main():
    parser = argparse.ArgumentParser(description="Pair-task noise floor + trend continuation + cross-labeler agreement")
    parser.add_argument("--topics_path", type=str, default="data/task_topic_pair_growth_prediction/topic_pairs.v5.stable.json")
    parser.add_argument("--gpt_labels_path", type=str, default="data/task_topic_pair_growth_prediction/topic_labels.v5.gpt-5.4.full.jsonl")
    parser.add_argument("--claude_labels_path", type=str, default="data/task_topic_growth_prediction/topic_labels.v5.claude-opus-4-7.500.jsonl")
    parser.add_argument("--single_topics_path", type=str, default="dataset/corpus/topics_list_v5.txt")
    parser.add_argument("--data_dir", type=str, default="data/prescience-ai", help="Dataset directory (with train/ and test/ subdirectories)")
    parser.add_argument("--bootstrap_K", type=int, default=10)
    parser.add_argument("--xlabel_min_count", type=int, default=2, help="Min pair count in either period within the 530-paper overlap (smaller than the headline 10 because n=530 is tiny)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="data/task_topic_pair_growth_prediction/scored/noise_floor.json")
    args = parser.parse_args()

    config, instances, _, _, metadata = load_topics_benchmark(args.topics_path)
    ordered_pair_ids = [cid for cid, _ in instances]
    pair_id_set = set(ordered_pair_ids)
    history_start, history_end, forecast_end = config["history_start"], config["history_end"], config["forecast_end"]
    n_pairs = len(ordered_pair_ids)
    utils.log(f"Loaded benchmark: {n_pairs} pairs; history=[{history_start},{history_end}); forecast=[{history_end},{forecast_end})")

    from dataset.corpus.assign_topics import load_topics
    topic_vocab = set(load_topics(args.single_topics_path))

    utils.log(f"Loading GPT labels from {args.gpt_labels_path}")
    gpt_labels = load_topic_labels(args.gpt_labels_path)
    utils.log(f"Loading Claude labels from {args.claude_labels_path}")
    claude_labels = load_topic_labels(args.claude_labels_path)

    utils.log(f"Loading corpus from {args.data_dir}")
    train_papers, _, _ = utils.load_corpus(data_dir=args.data_dir, split="train", embedding_type=None, load_sd2publications=False)
    test_papers, _, _ = utils.load_corpus(data_dir=args.data_dir, split="test", embedding_type=None, load_sd2publications=False)
    papers_by_id = {}
    for p in train_papers + test_papers:
        if p.get("corpus_id") in gpt_labels and p["corpus_id"] not in papers_by_id and "target" in (p.get("roles") or []):
            papers_by_id[p["corpus_id"]] = p
    utils.log(f"Resolved {len(papers_by_id)} target papers with GPT labels")

    papers_with_topics = [(cid, p["date"], gpt_labels.get(cid, [])) for cid, p in papers_by_id.items()]
    utils.log(f"Built (cid, date, topics) tuples for {len(papers_with_topics)} papers")

    utils.log(f"=== (1) Bootstrap reproducibility ceiling, K={args.bootstrap_K} ===")
    boots = bootstrap_ceiling(papers_with_topics, ordered_pair_ids, pair_id_set, topic_vocab, history_start, history_end, forecast_end, n_pairs, args.bootstrap_K, args.seed)
    r2s = [b["r2"] for b in boots]
    pearsons = [b["pearson"] for b in boots if b["pearson"] is not None]
    boot_summary = {"K": args.bootstrap_K, "r2_median": float(np.median(r2s)), "r2_q25": float(np.quantile(r2s, 0.25)), "r2_q75": float(np.quantile(r2s, 0.75)), "r2_min": float(min(r2s)), "r2_max": float(max(r2s)), "pearson_median": float(np.median(pearsons)) if pearsons else None}
    utils.log(f"[bootstrap] Δshare R² median={boot_summary['r2_median']:.4f}  IQR=[{boot_summary['r2_q25']:.4f}, {boot_summary['r2_q75']:.4f}]  range=[{boot_summary['r2_min']:.4f}, {boot_summary['r2_max']:.4f}]")
    utils.log(f"[bootstrap] Pearson median={boot_summary['pearson_median']:.4f}")

    utils.log("=== (2) Within-history vs across-window trend continuation ===")
    trend = trend_continuation(papers_with_topics, ordered_pair_ids, pair_id_set, topic_vocab, history_start, history_end, forecast_end, n_pairs)
    utils.log(f"[trend] R²(predict ΔshareF from ΔshareH) = {trend['r2_predict_F_from_H']:.4f}")
    utils.log(f"[trend] Pearson = {trend['pearson']:.4f}; fraction of forecast-Δshare variance explained by inner-history trend = {trend['frac_variance_explained']:.4f}")

    utils.log("=== (3) Cross-labeler agreement on pairs (GPT vs Claude on 530-paper overlap) ===")
    xlabel = cross_labeler_agreement(gpt_labels, claude_labels, papers_by_id, topic_vocab, history_start, history_end, forecast_end, args.xlabel_min_count)
    for k, v in xlabel.items():
        utils.log(f"[xlabel] {k}: {v}")

    output = {"config": {"topics_path": args.topics_path, "history_start": history_start, "history_end": history_end, "forecast_end": forecast_end, "bootstrap_K": args.bootstrap_K, "xlabel_min_count": args.xlabel_min_count, "seed": args.seed, "n_pairs": n_pairs}, "bootstrap_ceiling": {"summary": boot_summary, "per_split": boots}, "trend_continuation": trend, "cross_labeler_agreement": xlabel}
    utils.log(f"Saving analysis output to {args.output_path}")
    utils.save_json(output, args.output_path, metadata=utils.update_metadata(metadata, args), overwrite=True)


if __name__ == "__main__":
    main()
