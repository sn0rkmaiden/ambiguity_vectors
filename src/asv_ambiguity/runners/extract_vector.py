from __future__ import annotations

import argparse

import torch

from asv_ambiguity.config import load_yaml
from asv_ambiguity.generation.dataset_builder import resolve_project_path
from asv_ambiguity.utils.io import write_json, ensure_parent
from asv_ambiguity.vectors.extract import extract_mean_difference_vector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-config", required=True)
    parser.add_argument("--activations", required=True)
    args = parser.parse_args()

    vector_config = load_yaml(args.vector_config)
    activation_bundle = torch.load(args.activations, map_location="cpu")

    vector, metadata = extract_mean_difference_vector(
        activation_bundle=activation_bundle,
        layer=int(vector_config["vector"]["layer"]),
        positive_label=str(vector_config["vector"]["positive_label"]),
        negative_labels=[str(x) for x in vector_config["vector"]["negative_labels"]],
        split="train",
        normalize=vector_config["vector"].get("normalize", "l2"),
    )

    vector_path = resolve_project_path(args.vector_config, vector_config["output"]["vector_pt"])
    metadata_path = resolve_project_path(args.vector_config, vector_config["output"]["metadata_json"])
    ensure_parent(vector_path)
    torch.save(vector, vector_path)
    write_json(metadata_path, metadata)

    print(f"Saved vector to {vector_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
