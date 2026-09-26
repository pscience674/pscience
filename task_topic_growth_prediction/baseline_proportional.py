"""Size-proportional baseline for topic growth prediction (causally safe, inner-split calibrated).

Calibration only uses information within the history period:
  α = (Σ_t N_t(F_inner) / |F_inner|) / (Σ_t N_t(H_inner) / |H_inner|)
  predicted(t) = (N_t(H) / |H|) × |F| × α

α captures the field-wide rate ratio between inner-forecast and inner-history months. Under
stationary rates α ≈ 1 and this reduces to the naive "same per-topic rate continues" prediction.
"""

import os
import argparse

import utils
from task_topic_growth_prediction.dataset import load_topics_benchmark


def main():
    parser = argparse.ArgumentParser(description="Proportional baseline (inner-split calibrated)")
    parser.add_argument("--topics_path", type=str, required=True, help="Path to topics.vX.json")
    parser.add_argument("--inner_history_months", type=int, default=6, help="First N months of history used as inner-history; remaining months are inner-forecast")
    parser.add_argument("--output_dir", type=str, default="data/task_topic_growth_prediction/predictions", help="Output directory")
    args = parser.parse_args()

    utils.log(f"Loading benchmark from {args.topics_path}")
    config, instances, _, _, metadata = load_topics_benchmark(args.topics_path)
    H = config["history_months"]
    F = config["forecast_months"]
    N = args.inner_history_months
    assert 1 <= N < H, f"--inner_history_months must be in [1, {H-1}]"
    F_inner = H - N
    utils.log(f"|H|={H}, |F|={F}, inner_history_months={N}, |F_inner|={F_inner}")

    inner_history_total = sum(sum(inst["history_monthly_counts"][:N]) for _, inst in instances)
    inner_forecast_total = sum(sum(inst["history_monthly_counts"][N:H]) for _, inst in instances)
    inner_history_rate = inner_history_total / N if N > 0 else 0.0
    inner_forecast_rate = inner_forecast_total / F_inner if F_inner > 0 else 0.0
    alpha = (inner_forecast_rate / inner_history_rate) if inner_history_rate > 0 else 0.0
    utils.log(f"Σ N_t(H_inner)={inner_history_total}, Σ N_t(F_inner)={inner_forecast_total}")
    utils.log(f"inner_history_rate_per_month={inner_history_rate:.4f}, inner_forecast_rate_per_month={inner_forecast_rate:.4f}, α={alpha:.4f}")

    predictions = []
    for cid, inst in instances:
        history_rate_per_month = inst["history_size"] / H if H > 0 else 0.0
        predicted = history_rate_per_month * F * alpha
        predictions.append({"cluster_id": cid, "predicted_growth": float(predicted), "gt_growth": inst["gt_growth"]})

    output_path = os.path.join(args.output_dir, "predictions.proportional.json")
    utils.log(f"Saving {len(predictions)} predictions to {output_path}")
    utils.save_json(predictions, output_path, metadata=utils.update_metadata(metadata, args), overwrite=True)


if __name__ == "__main__":
    main()
