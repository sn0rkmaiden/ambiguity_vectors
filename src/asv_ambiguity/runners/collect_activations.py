from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from asv_ambiguity.activations.collector import collect_hidden_state_vectors
from asv_ambiguity.config import load_yaml
from asv_ambiguity.models.hf import HFCausalModel
from asv_ambiguity.utils.io import read_jsonl, ensure_parent


def slugify(text: str) -> str:
    text = text.strip().replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_")


def build_activation_output_path(
    *,
    extraction_config_path: str,
    configured_output_path: str,
    dataset_path: str,
    model_name: str,
    position: str,
) -> Path:
    base_output_path = Path(configured_output_path)
    if not base_output_path.is_absolute():
        repo_root = Path(extraction_config_path).resolve().parents[2]
        base_output_path = repo_root / configured_output_path

    output_dir = base_output_path.parent
    dataset_stem = Path(dataset_path).stem
    filename = f"{dataset_stem}__{slugify(model_name)}__{slugify(position)}.pt"
    return output_dir / filename


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

    output_path = build_activation_output_path(
        extraction_config_path=args.extraction_config,
        configured_output_path=extraction_config["output"]["activations_pt"],
        dataset_path=args.dataset,
        model_name=model.model_name,
        position=position,
    )
    ensure_parent(output_path)
    torch.save(bundle, output_path)
    print(f"Saved activations to {output_path}")


if __name__ == "__main__":
    main()