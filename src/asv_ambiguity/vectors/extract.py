from __future__ import annotations

from typing import Any

import torch


def extract_mean_difference_vector(
    *,
    activation_bundle: dict[str, Any],
    layer: int,
    positive_label: str,
    negative_labels: list[str],
    split: str = "train",
    normalize: str | None = "l2",
) -> tuple[torch.Tensor, dict[str, Any]]:
    positives: list[torch.Tensor] = []
    negatives: list[torch.Tensor] = []

    for record in activation_bundle["records"]:
        if record["split"] != split:
            continue
        vector = record["layers"][layer]
        if record["label"] == positive_label:
            positives.append(vector)
        elif record["label"] in negative_labels:
            negatives.append(vector)

    if not positives:
        raise ValueError("No positive examples found for vector extraction.")
    if not negatives:
        raise ValueError("No negative examples found for vector extraction.")

    pos_mean = torch.stack(positives, dim=0).mean(dim=0)
    neg_mean = torch.stack(negatives, dim=0).mean(dim=0)
    vector = pos_mean - neg_mean

    norm_before = float(torch.norm(vector).item())
    if normalize == "l2":
        if norm_before == 0.0:
            raise ValueError("Vector norm is zero; cannot normalize.")
        vector = vector / norm_before

    metadata = {
        "layer": layer,
        "split": split,
        "positive_label": positive_label,
        "negative_labels": negative_labels,
        "num_positive": len(positives),
        "num_negative": len(negatives),
        "norm_before_normalization": norm_before,
        "normalize": normalize,
    }
    return vector, metadata
