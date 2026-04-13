from __future__ import annotations

from typing import Any

import torch


def _get_record_vector(
    *,
    activation_bundle: dict[str, Any],
    record: dict[str, Any],
    position: str,
    layer: int,
) -> torch.Tensor:
    if "positions" in record:
        return record["positions"][position][layer]

    bundle_position = activation_bundle.get("position")
    if bundle_position != position:
        raise ValueError(
            f"Old-style activation bundle has position={bundle_position}, "
            f"but vector extraction requested position={position}."
        )
    return record["layers"][layer]


def extract_mean_difference_vector(
    *,
    activation_bundle: dict[str, Any],
    layer: int,
    position: str,
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

        vector = _get_record_vector(
            activation_bundle=activation_bundle,
            record=record,
            position=position,
            layer=layer,
        )

        if record["label"] == positive_label:
            positives.append(vector)
        elif record["label"] in negative_labels:
            negatives.append(vector)

    if not positives:
        raise ValueError(f"No positive examples found for layer={layer}, position={position}.")
    if not negatives:
        raise ValueError(f"No negative examples found for layer={layer}, position={position}.")

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
        "position": position,
        "split": split,
        "positive_label": positive_label,
        "negative_labels": negative_labels,
        "num_positive": len(positives),
        "num_negative": len(negatives),
        "norm_before_normalization": norm_before,
        "normalize": normalize,
    }
    return vector, metadata