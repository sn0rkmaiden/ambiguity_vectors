from __future__ import annotations

import argparse
from pathlib import Path

import torch

from asv_ambiguity.activations.collector import collect_hidden_state_vectors
from asv_ambiguity.config import load_yaml
from asv_ambiguity.generation.dataset_builder import resolve_project_path
from asv_ambiguity.models.hf import HFCausalModel
from asv_ambiguity.utils.io import read_jsonl, ensure_parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--extraction-config", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    model_config = load_yaml(args.model_config)
    extraction_config = load_yaml(args.extraction_config)
    model = HFCausalModel(model_config)

    dataset_rows = read_jsonl(args.dataset)
    layers = [int(x) for x in extraction_config["extraction"]["layers"]]
    position = str(extraction_config["extraction"]["position"])

    bundle = collect_hidden_state_vectors(
        model=model,
        dataset_rows=dataset_rows,
        layers=layers,
        position=position,
    )

    output_path = resolve_project_path(
        args.extraction_config,
        extraction_config["output"]["activations_pt"],
    )
    ensure_parent(output_path)
    torch.save(bundle, output_path)
    print(f"Saved activations to {output_path}")


if __name__ == "__main__":
    main()
