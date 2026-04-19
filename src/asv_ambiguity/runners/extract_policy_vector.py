from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from asv_ambiguity.config import load_yaml, resolve_project_path
from asv_ambiguity.utils.io import write_json, ensure_parent
from asv_ambiguity.vectors.extract import extract_mean_difference_vector


def slugify(text: str) -> str:
    text = text.strip().replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-config", required=True)
    parser.add_argument("--activations", required=True)
    args = parser.parse_args()

    vector_config = load_yaml(args.vector_config)
    activation_bundle = torch.load(args.activations, map_location="cpu")

    vector_section = vector_config["vector"]

    layers = vector_section.get("layers")
    if layers is None:
        layers = [int(vector_section["layer"])]
    else:
        layers = [int(x) for x in layers]

    positions = vector_section.get("positions")
    if positions is None:
        if "position" in vector_section:
            positions = [str(vector_section["position"])]
        else:
            positions = [str(x) for x in activation_bundle.get("positions", [activation_bundle.get("position")])]
    else:
        positions = [str(x) for x in positions]

    output_cfg = vector_config["output"]
    if "output_dir" in output_cfg:
        output_dir = resolve_project_path(args.vector_config, output_cfg["output_dir"])
        prefix = str(output_cfg.get("prefix", vector_section.get("concept_name", "vector")))
    else:
        base_vector_path = resolve_project_path(args.vector_config, output_cfg["vector_pt"])
        output_dir = base_vector_path.parent
        prefix = base_vector_path.stem

    ensure_parent(output_dir / "dummy.txt")

    for position in positions:
        for layer in layers:
            vector, metadata = extract_mean_difference_vector(
                activation_bundle=activation_bundle,
                layer=layer,
                position=position,
                positive_label=str(vector_section["positive_label"]),
                negative_labels=[str(x) for x in vector_section["negative_labels"]],
                split="train",
                normalize=vector_section.get("normalize", "l2"),
            )

            stem = f"{slugify(prefix)}__{slugify(position)}__layer{layer}"
            vector_path = output_dir / f"{stem}.pt"
            metadata_path = output_dir / f"{stem}.json"

            torch.save(vector, vector_path)
            write_json(metadata_path, metadata)

            print(f"Saved vector to {vector_path}")
            print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()