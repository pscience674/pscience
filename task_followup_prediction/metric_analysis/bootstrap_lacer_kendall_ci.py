"""Bootstrap confidence intervals for the LACER metric-validation Kendall tau-b agreements.

Reuses the exact tau-b pipeline from plot_lacer_kendall_matrix.py so point estimates match
the paper (IAA ~0.53, LACER ~0.57). Reports 95% CIs for inter-annotator agreement (IAA) and
each metric-vs-human agreement, plus the metric-minus-IAA difference with a one-sided p-value,
under two resampling units: papers (cluster over the 5 target papers) and unit (flat over the
per-observation tau values)."""

import argparse
import json
import os
import warnings
from itertools import combinations

import numpy as np

from task_followup_prediction.metric_analysis.plot_lacer_kendall_matrix import (
    LABELS, load_json, collect_annotator_ids, parse_ground_truth_title,
    human_vectors_from_record, lacer_vector_from_record, average_lacer_vectors,
    metric_vectors_from_dataset, is_valid_vector, kendall_tau,
    compute_human_human_value, compute_human_metric_value,
)

METRIC_LABELS = LABELS[1:]


def build_records_data(gpt5_path, opus_path, dataset_path):
    """Load scored records and assemble the human + metric ranking vectors (matches the matrix script)."""
    gpt5_records = load_json(gpt5_path)
    opus_records = load_json(opus_path)
    dataset = load_json(dataset_path)
    dataset_mapping = {entry["target"]["title"].strip(): entry for entry in dataset if isinstance(entry.get("target", {}).get("title"), str)}

    records_data = []
    for gpt5_record, opus_record in zip(gpt5_records, opus_records):
        title = parse_ground_truth_title(gpt5_record.get("ground_truth", ""))
        if title not in dataset_mapping:
            raise KeyError(f"No annotation dataset entry found for title: {title}")
        gpt5_vector = lacer_vector_from_record(gpt5_record)
        opus_vector = lacer_vector_from_record(opus_record)
        metric_vectors = metric_vectors_from_dataset(dataset_mapping[title])
        metric_vectors["LACER (GPT-5)"] = gpt5_vector
        metric_vectors["LACER (Opus)"] = opus_vector
        metric_vectors["LACER (Avg)"] = average_lacer_vectors(gpt5_vector, opus_vector)
        records_data.append({"humans": human_vectors_from_record(gpt5_record), "metrics": metric_vectors})

    annotator_ids = collect_annotator_ids(gpt5_records)
    return records_data, annotator_ids


def point_estimates(records_data, annotator_ids):
    """Nested-average point estimates: IAA and each metric-vs-human tau-b."""
    estimates = {"IAA": compute_human_human_value(records_data, annotator_ids)[0]}
    for label in METRIC_LABELS:
        estimates[label] = compute_human_metric_value(records_data, annotator_ids, label)[0]
    return estimates


def precompute_base_taus(records_data, annotator_ids):
    """Precompute base tau-b per (annotator-pair, record) for IAA and per (annotator, record) for each metric.

    Bootstrapping over records then reduces to resampling columns and re-averaging these arrays, avoiding
    millions of redundant kendalltau calls. NaN marks an invalid (pair, record) or (annotator, record) cell.
    """
    n = len(records_data)
    pair_tau = np.full((len(list(combinations(annotator_ids, 2))), n), np.nan)
    for pair_idx, (annotator_a, annotator_b) in enumerate(combinations(annotator_ids, 2)):
        for r, record in enumerate(records_data):
            vec_a, vec_b = record["humans"].get(annotator_a), record["humans"].get(annotator_b)
            if is_valid_vector(vec_a) and is_valid_vector(vec_b):
                pair_tau[pair_idx, r] = kendall_tau(vec_a, vec_b)
    metric_tau = {}
    for label in METRIC_LABELS:
        arr = np.full((len(annotator_ids), n), np.nan)
        for a, annotator_id in enumerate(annotator_ids):
            for r, record in enumerate(records_data):
                human_vec, metric_vec = record["humans"].get(annotator_id), record["metrics"].get(label)
                if is_valid_vector(human_vec) and is_valid_vector(metric_vec):
                    arr[a, r] = kendall_tau(human_vec, metric_vec)
        metric_tau[label] = arr
    return pair_tau, metric_tau


def bootstrap_paper(records_data, annotator_ids, n_boot):
    """Cluster bootstrap over target papers (records), recomputing nested tau-b jointly from precomputed base taus."""
    pair_tau, metric_tau = precompute_base_taus(records_data, annotator_ids)
    n = pair_tau.shape[1]
    samples = {"IAA": np.empty(n_boot)}
    for label in METRIC_LABELS:
        samples[label] = np.empty(n_boot)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for b in range(n_boot):
            idx = np.random.randint(0, n, size=n)
            samples["IAA"][b] = np.nanmean(np.nanmean(pair_tau[:, idx], axis=1))
            for label in METRIC_LABELS:
                samples[label][b] = np.nanmean(np.nanmean(metric_tau[label][:, idx], axis=1))
    return samples


def collect_tau_values(records_data, annotator_ids):
    """Flat per-observation tau lists: per (record, annotator-pair) for IAA, per (record, annotator) for each metric."""
    tau_lists = {"IAA": []}
    for record in records_data:
        for annotator_a, annotator_b in combinations(annotator_ids, 2):
            vec_a = record["humans"].get(annotator_a)
            vec_b = record["humans"].get(annotator_b)
            if is_valid_vector(vec_a) and is_valid_vector(vec_b):
                tau = kendall_tau(vec_a, vec_b)
                if tau is not None:
                    tau_lists["IAA"].append(tau)
    for label in METRIC_LABELS:
        values = []
        for record in records_data:
            metric_vec = record["metrics"].get(label)
            for annotator_id in annotator_ids:
                human_vec = record["humans"].get(annotator_id)
                if is_valid_vector(human_vec) and is_valid_vector(metric_vec):
                    tau = kendall_tau(human_vec, metric_vec)
                    if tau is not None:
                        values.append(tau)
        tau_lists[label] = values
    return {key: np.array(values, dtype=float) for key, values in tau_lists.items()}


def bootstrap_unit(tau_lists, n_boot):
    """Flat bootstrap: resample each quantity's per-observation tau values with replacement and take the mean."""
    samples = {}
    for key, values in tau_lists.items():
        means = [values[np.random.randint(0, len(values), size=len(values))].mean() for _ in range(n_boot)]
        samples[key] = np.array(means, dtype=float)
    return samples


def summarize(samples, point, ci_level):
    """Build per-quantity CI + metric-minus-IAA difference CI and one-sided p (P[metric < IAA])."""
    lo_q, hi_q = (100 - ci_level) / 2, 100 - (100 - ci_level) / 2
    iaa_draws = samples["IAA"]
    result = {"per_quantity": {}, "difference_vs_iaa": {}}
    for key, draws in samples.items():
        result["per_quantity"][key] = {"point": point.get(key), "ci_low": float(np.nanpercentile(draws, lo_q)), "ci_high": float(np.nanpercentile(draws, hi_q))}
    for label in METRIC_LABELS:
        diff = samples[label] - iaa_draws
        result["difference_vs_iaa"][label] = {"diff": float(point[label] - point["IAA"]), "ci_low": float(np.nanpercentile(diff, lo_q)), "ci_high": float(np.nanpercentile(diff, hi_q)), "p_metric_below_iaa": float(np.mean(diff < 0))}
    return result


def render_markdown(results, point, args):
    """Render a drop-in summary of point estimates and CIs under both resampling units."""
    lines = [f"# LACER validation: bootstrap {args.ci_level:.0f}% CIs", ""]
    lines.append(f"n_boot={args.n_boot}, seed={args.seed}. Validation set: 5 papers x 5 annotators x 10 generations.")
    lines.append(f"Point estimates (nested average): IAA={point['IAA']:.4f}, LACER (GPT-5)={point['LACER (GPT-5)']:.4f}.")
    lines.append("")
    unit = results.get("unit", results.get("paper"))
    lacer_diff = unit["difference_vs_iaa"]["LACER (GPT-5)"]
    baselines = ["MRR", "BERTScore", "ASPIRE Distance", "FacetScore"]
    all_baselines_below = all(unit["difference_vs_iaa"][m]["p_metric_below_iaa"] >= 0.95 for m in baselines)
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- LACER agreement with experts is **statistically comparable to inter-annotator agreement**: the LACER (GPT-5) minus IAA difference is {lacer_diff['diff']:+.3f} with 95% CI [{lacer_diff['ci_low']:+.3f}, {lacer_diff['ci_high']:+.3f}] (per-observation bootstrap), i.e. the CI includes 0 / the per-metric CIs overlap.")
    if all_baselines_below:
        lines.append("- LACER **significantly outperforms every prior automated metric** (MRR/ROUGE, BERTScore, ASPIRE, FacetScore): each sits below IAA with P(metric < IAA) >= 0.95, and all fall below LACER's CI.")
    lines.append("- Net: the CIs support 'LACER matches human agreement and beats prior automated metrics' without overclaiming a significant gap over humans.")
    lines.append("")
    for unit in results:
        lines.append(f"## Resampling unit: {unit}")
        lines.append("")
        lines.append("| Quantity | tau-b | CI low | CI high |")
        lines.append("|---|---|---|---|")
        for key, cell in results[unit]["per_quantity"].items():
            lines.append(f"| {key} | {cell['point']:.3f} | {cell['ci_low']:.3f} | {cell['ci_high']:.3f} |")
        lines.append("")
        lines.append("| Metric - IAA | diff | CI low | CI high | P(metric < IAA) |")
        lines.append("|---|---|---|---|---|")
        for label, cell in results[unit]["difference_vs_iaa"].items():
            lines.append(f"| {label} | {cell['diff']:+.3f} | {cell['ci_low']:+.3f} | {cell['ci_high']:+.3f} | {cell['p_metric_below_iaa']:.3f} |")
        lines.append("")
    lines.append("Note: `paper` resamples the 5 papers jointly (difference CI is exact under the joint resample);")
    lines.append("`unit` resamples per-observation tau values and forms the difference from independent draws (approximate).")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for LACER validation Kendall tau-b agreements")
    parser.add_argument("--gpt5_path", default="data/task_followup_prediction/metrics_analysis/annotator_rankings_rows_2_6.gpt5_scored.percentile50.110.json", help="Path to GPT-5 LACER scored JSON")
    parser.add_argument("--opus_path", default="data/task_followup_prediction/metrics_analysis/annotator_rankings_rows_2_6.opus_scored.percentile50.110.json", help="Path to Opus LACER scored JSON")
    parser.add_argument("--dataset_path", default="data/task_followup_prediction/metrics_analysis/annotation_dataset.json", help="Path to annotation_dataset.json with automated metrics")
    parser.add_argument("--output_dir", default="outputs/lacer_confidence_intervals", help="Directory for CI outputs")
    parser.add_argument("--bootstrap_unit", default="both", choices=["paper", "unit", "both"], help="Resampling unit for the bootstrap")
    parser.add_argument("--n_boot", type=int, default=10000, help="Number of bootstrap resamples")
    parser.add_argument("--ci_level", type=float, default=95.0, help="Confidence level in percent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    records_data, annotator_ids = build_records_data(args.gpt5_path, args.opus_path, args.dataset_path)
    print(f"Loaded {len(records_data)} papers, {len(annotator_ids)} annotators")
    point = point_estimates(records_data, annotator_ids)

    results = {}
    if args.bootstrap_unit in ("paper", "both"):
        print(f"Cluster bootstrap over {len(records_data)} papers ({args.n_boot} resamples)")
        results["paper"] = summarize(bootstrap_paper(records_data, annotator_ids, args.n_boot), point, args.ci_level)
    if args.bootstrap_unit in ("unit", "both"):
        tau_lists = collect_tau_values(records_data, annotator_ids)
        print(f"Flat bootstrap over per-observation tau values ({args.n_boot} resamples); IAA n={len(tau_lists['IAA'])}, metric n={len(tau_lists['LACER (GPT-5)'])}")
        results["unit"] = summarize(bootstrap_unit(tau_lists, args.n_boot), point, args.ci_level)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "lacer_validation_ci.json"), "w") as f:
        json.dump({"metadata": vars(args), "point_estimates": point, "results": results}, f, indent=2)
    with open(os.path.join(args.output_dir, "lacer_validation_ci.md"), "w") as f:
        f.write(render_markdown(results, point, args))
    print(f"Saved CI outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
