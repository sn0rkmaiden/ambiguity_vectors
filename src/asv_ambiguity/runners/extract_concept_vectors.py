from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from asv_ambiguity.config import load_yaml


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def slugify(text: str) -> str:
    text = text.strip().replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_")


def resolve_project_path(config_path: str | Path, maybe_relative_path: str | Path) -> Path:
    base = Path(config_path).resolve().parents[2]
    path = Path(maybe_relative_path)
    return path if path.is_absolute() else (base / path)


def extract_one_vector(
    *,
    activation_bundle: dict,
    concept: str,
    layer: int,
    split: str,
    normalize: str | None,
) -> tuple[torch.Tensor, dict]:
    positives = []
    negatives = []

    for record in activation_bundle["records"]:
        if record["split"] != split:
            continue
        vec = record["layers"][layer]
        if record["concept"] == concept:
            positives.append(vec)
        else:
            negatives.append(vec)

    if not positives:
        raise ValueError(f"No positives found for concept={concept}, layer={layer}, split={split}")
    if not negatives:
        raise ValueError(f"No negatives found for concept={concept}, layer={layer}, split={split}")

    pos_mean = torch.stack(positives, dim=0).mean(dim=0)
    neg_mean = torch.stack(negatives, dim=0).mean(dim=0)
    vector = pos_mean - neg_mean

    norm_before = float(torch.norm(vector).item())
    if normalize == "l2":
        if norm_before == 0.0:
            raise ValueError("Vector norm is zero; cannot normalize.")
        vector = vector / norm_before

    meta = {
        "concept": concept,
        "layer": layer,
        "split": split,
        "num_positive": len(positives),
        "num_negative": len(negatives),
        "burn_in_tokens": activation_bundle.get("burn_in_tokens"),
        "kind": activation_bundle.get("kind"),
        "normalize": normalize,
        "norm_before_normalization": norm_before,
        "contrast": "mean_of_concept_minus_mean_of_other_concepts",
    }
    return vector, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-config", required=True)
    parser.add_argument("--activations", required=True)
    args = parser.parse_args()

    vector_config = load_yaml(args.vector_config)
    bundle = torch.load(args.activations, map_location="cpu")

    concepts = [str(x) for x in vector_config["vector"]["concepts"]]
    layers = [int(x) for x in vector_config["vector"]["layers"]]
    split = str(vector_config["vector"].get("split", "train"))
    normalize = vector_config["vector"].get("normalize", "l2")

    output_dir = resolve_project_path(args.vector_config, vector_config["output"]["output_dir"])
    ensure_parent(output_dir / "dummy.txt")

    for concept in concepts:
        for layer in layers:
            vector, metadata = extract_one_vector(
                activation_bundle=bundle,
                concept=concept,
                layer=layer,
                split=split,
                normalize=normalize,
            )

            stem = f"{slugify(concept)}__layer{layer}"
            vector_path = output_dir / f"{stem}.pt"
            metadata_path = output_dir / f"{stem}.json"

            torch.save(vector, vector_path)
            write_json(metadata_path, metadata)

            print(f"Saved vector to {vector_path}")
            print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()