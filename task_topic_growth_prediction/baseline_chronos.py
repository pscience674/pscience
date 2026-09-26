"""Chronos pretrained time-series baseline for topic growth prediction (causally safe).

Zero-shot forecast of the next |F| monthly counts per topic with a pretrained
Chronos-Bolt model; predicted growth = sum of the median forecast path.
Uses only history-period data.
"""

import os
import argparse

import numpy as np
import torch
from chronos import BaseChronosPipeline

import utils
from task_topic_growth_prediction.dataset import load_topics_benchmark

torch.manual_seed(42)


def main():
    parser = argparse.ArgumentParser(description="Chronos pretrained forecaster baseline")
    parser.add_argument("--topics_path", type=str, required=True, help="Path to topics.vX.json")
    parser.add_argument("--model", type=str, default="amazon/chronos-bolt-base", help="Chronos model to use")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for inference")
    parser.add_argument("--batch_size", type=int, default=256, help="Series per forward pass")
    parser.add_argument("--output_dir", type=str, default="data/task_topic_growth_prediction/predictions", help="Output directory")
    args = parser.parse_args()

    utils.log(f"Loading benchmark from {args.topics_path}")
    config, instances, _, _, metadata = load_topics_benchmark(args.topics_path)
    F = config["forecast_months"]

    utils.log(f"Loading Chronos model {args.model} on {args.device}")
    pipeline = BaseChronosPipeline.from_pretrained(args.model, device_map=args.device, torch_dtype=torch.float32)

    utils.log(f"Forecasting {F} months for {len(instances)} topics")
    predictions = []
    for start in range(0, len(instances), args.batch_size):
        batch = instances[start:start + args.batch_size]
        contexts = [torch.tensor(inst["history_monthly_counts"], dtype=torch.float32) for _, inst in batch]
        quantiles, _ = pipeline.predict_quantiles(contexts, prediction_length=F, quantile_levels=[0.5])
        medians = quantiles[:, :, 0].cpu().numpy()
        for (cid, inst), median_path in zip(batch, medians):
            predicted = float(np.sum(np.maximum(median_path, 0.0)))
            predictions.append({"cluster_id": cid, "predicted_growth": predicted, "gt_growth": inst["gt_growth"]})

    suffix = args.model.split("/")[-1].replace("-", "_").replace(".", "_")
    output_path = os.path.join(args.output_dir, f"predictions.{suffix}.json")
    utils.log(f"Saving {len(predictions)} predictions to {output_path}")
    utils.save_json(predictions, output_path, metadata=utils.update_metadata(metadata, args), overwrite=True)


if __name__ == "__main__":
    main()
