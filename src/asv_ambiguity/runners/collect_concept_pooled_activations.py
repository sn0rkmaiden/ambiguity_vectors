from __future__ import annotations

import argparse
from pathlib import Path
import re
import json

import torch
from tqdm.auto import tqdm

from asv_ambiguity.config import load_yaml
from asv_ambiguity.models.hf import HFCausalModel


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    text = text.strip().replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_")


def build_output_path(
    *,
    extraction_config_path: str,
    configured_output_path: str,
    dataset_path: str,
    model_name: str,
) -> Path:
    base_output_path = Path(configured_output_path)
    if not base_output_path.is_absolute():
        repo_root = Path(extraction_config_path).resolve().parents[2]
        base_output_path = repo_root / configured_output_path

    output_dir = base_output_path.parent
    dataset_stem = Path(dataset_path).stem
    filename = f"{dataset_stem}__{slugify(model_name)}__concept_pooled.pt"
    return output_dir / filename


def pooled_representation_for_layer(
    *,
    hidden: torch.Tensor,
    burn_in_tokens: int,
) -> tuple[torch.Tensor, int]:
    seq_len = int(hidden.shape[0])
    start = min(burn_in_tokens, max(seq_len - 1, 0))
    pooled = hidden[start:].mean(dim=0)
    return pooled, start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--extraction-config", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    model_config = load_yaml(args.model_config)
    extraction_config = load_yaml(args.extraction_config)

    model = HFCausalModel(model_config)
    model_name = getattr(model, "model_name", model_config["model"]["name"])

    dataset_rows = read_jsonl(Path(args.dataset))
    layers = [int(x) for x in extraction_config["extraction"]["layers"]]
    burn_in_tokens = int(extraction_config["extraction"]["burn_in_tokens"])

    records = []
    for row in tqdm(dataset_rows, desc="Collecting pooled concept activations", unit="text"):
        outputs = model.forward_hidden_states(row["text"])
        input_ids = outputs["input_ids"][0].detach().cpu()
        token_count = int(input_ids.shape[0])

        per_layer = {}
        used_start = None
        for layer_idx in layers:
            hidden = outputs["hidden_states"][layer_idx][0].detach().cpu().float()
            pooled, start = pooled_representation_for_layer(
                hidden=hidden,
                burn_in_tokens=burn_in_tokens,
            )
            per_layer[layer_idx] = pooled
            used_start = start

        records.append(
            {
                "record_id": row["record_id"],
                "concept": row["concept"],
                "split": row["split"],
                "topic": row.get("topic", ""),
                "ambiguity_type": row.get("ambiguity_type", ""),
                "token_count": token_count,
                "used_start_token": used_start,
                "layers": per_layer,
                "text": row["text"],
            }
        )

    bundle = {
        "kind": "concept_pooled",
        "burn_in_tokens": burn_in_tokens,
        "layers": layers,
        "records": records,
    }

    output_path = build_output_path(
        extraction_config_path=args.extraction_config,
        configured_output_path=extraction_config["output"]["activations_pt"],
        dataset_path=args.dataset,
        model_name=model_name,
    )
    ensure_parent(output_path)
    torch.save(bundle, output_path)
    print(f"Saved pooled activations to {output_path}")


if __name__ == "__main__":
    main()