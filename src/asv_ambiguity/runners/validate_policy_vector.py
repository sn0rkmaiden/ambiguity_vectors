from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from asv_ambiguity.activations.positions import select_hidden_representation
from asv_ambiguity.config import load_yaml
from asv_ambiguity.models.hf import HFCausalModel


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_output_paths(vector_path: str, dataset_path: str, splits: list[str]) -> tuple[Path, Path]:
    vector_stem = Path(vector_path).stem
    dataset_stem = Path(dataset_path).stem
    split_tag = "-".join(splits)
    out_dir = Path("outputs") / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{vector_stem}__{dataset_stem}__{split_tag}__summary.json"
    rows_path = out_dir / f"{vector_stem}__{dataset_stem}__{split_tag}__rows.csv"
    return summary_path, rows_path


def load_vector(vector_path: str) -> torch.Tensor:
    obj = torch.load(vector_path, map_location="cpu")
    if isinstance(obj, dict):
        if "vector" in obj:
            obj = obj["vector"]
        elif "direction" in obj:
            obj = obj["direction"]
        else:
            raise ValueError(f"Unsupported vector file structure in {vector_path}")
    if not isinstance(obj, torch.Tensor):
        raise ValueError(f"Expected tensor in {vector_path}, got {type(obj)}")
    return obj.float().cpu()


def build_scoring_user_prompt(row: dict) -> str:
    candidates = row.get("candidate_values", row.get("candidate_referents", []))
    candidates_block = "\n".join(f"- {x}" for x in candidates)

    ambiguity_type = row.get("ambiguity_type", "clarification")
    gold_missing_slot = row.get("gold_missing_slot", "unknown")

    return f"""You are an assistant responding to an ambiguous instruction.

Give a single natural response to the situation below.

Ambiguity type: {ambiguity_type}
Missing slot: {gold_missing_slot}

Context:
{row['context']}

Instruction:
{row['instruction']}

Candidate values:
{candidates_block}"""


def render_full_text_and_prompt_token_count(
    model: HFCausalModel,
    user_prompt: str,
    response_text: str,
) -> tuple[str, int]:
    prompt_text = model.render_prompt(user_prompt)
    prompt_ids = model.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    prompt_token_count = int(prompt_ids.shape[1])
    full_text = prompt_text + response_text
    return full_text, prompt_token_count


def score_response(
    *,
    model: HFCausalModel,
    vector: torch.Tensor,
    layer_idx: int,
    position: str,
    user_prompt: str,
    response_text: str,
) -> float:
    full_text, prompt_token_count = render_full_text_and_prompt_token_count(
        model=model,
        user_prompt=user_prompt,
        response_text=response_text,
    )

    outputs = model.forward_hidden_states(full_text)
    hidden_states = outputs["hidden_states"]

    if layer_idx >= len(hidden_states):
        raise IndexError(
            f"Requested layer {layer_idx}, but model returned only {len(hidden_states)} hidden-state tensors."
        )

    hidden = hidden_states[layer_idx][0].detach().float().cpu()
    rep = select_hidden_representation(
        hidden=hidden,
        input_ids=outputs["input_ids"].detach().cpu(),
        prompt_token_count=prompt_token_count,
        tokenizer=model.tokenizer,
        mode=position,
    ).detach().float().cpu()

    return float(torch.dot(rep, vector))


def summarize_results(results: list[dict], positive_label: str, negative_labels: list[str]) -> dict:
    n = len(results)
    if n == 0:
        raise ValueError("No validation rows were scored.")

    pos_gt_each = {}
    mean_gap_each = {}

    for neg in negative_labels:
        wins = sum(r[f"score_{positive_label}"] > r[f"score_{neg}"] for r in results)
        pos_gt_each[neg] = wins / n
        mean_gap_each[neg] = sum(
            r[f"score_{positive_label}"] - r[f"score_{neg}"] for r in results
        ) / n

    pos_gt_all = sum(
        all(r[f"score_{positive_label}"] > r[f"score_{neg}"] for neg in negative_labels)
        for r in results
    ) / n

    argmax_positive = sum(
        r["best_label"] == positive_label
        for r in results
    ) / n

    return {
        "num_examples": n,
        "positive_label": positive_label,
        "negative_labels": negative_labels,
        "positive_beats_each_negative_accuracy": pos_gt_each,
        "positive_beats_all_negatives_accuracy": pos_gt_all,
        "argmax_is_positive_accuracy": argmax_positive,
        "mean_score_gaps": mean_gap_each,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--vector", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    args = parser.parse_args()

    model_config = load_yaml(args.model_config)
    model = HFCausalModel(model_config)

    dataset_rows = read_jsonl(Path(args.dataset))
    vector = load_vector(args.vector)

    with open(args.metadata, "r", encoding="utf-8") as f:
        meta = json.load(f)

    layer_idx = int(meta["layer"])
    position = str(meta["position"])
    positive_label = str(meta["positive_label"])
    negative_labels = [str(x) for x in meta["negative_labels"]]

    filtered_rows = [row for row in dataset_rows if row["split"] in set(args.splits)]
    if not filtered_rows:
        raise ValueError(f"No rows found for splits {args.splits}")

    results = []
    labels_to_score = [positive_label] + negative_labels

    for row in tqdm(filtered_rows, desc="Validating policy vector", unit="example"):
        user_prompt = build_scoring_user_prompt(row)

        scored = {}
        for label in labels_to_score:
            scored[f"score_{label}"] = score_response(
                model=model,
                vector=vector,
                layer_idx=layer_idx,
                position=position,
                user_prompt=user_prompt,
                response_text=row[label],
            )

        best_label = max(labels_to_score, key=lambda lbl: scored[f"score_{lbl}"])

        result_row = {
            "example_id": row["example_id"],
            "split": row["split"],
            "topic": row.get("topic", ""),
            "ambiguity_type": row.get("ambiguity_type", ""),
            "instruction": row["instruction"],
            "best_label": best_label,
        }
        result_row.update(scored)
        results.append(result_row)

    summary = summarize_results(
        results=results,
        positive_label=positive_label,
        negative_labels=negative_labels,
    )
    summary["vector_path"] = str(Path(args.vector).resolve())
    summary["metadata_path"] = str(Path(args.metadata).resolve())
    summary["dataset_path"] = str(Path(args.dataset).resolve())
    summary["model_name"] = model.model_name
    summary["layer"] = layer_idx
    summary["position"] = position
    summary["splits"] = args.splits

    summary_path, rows_path = build_output_paths(
        vector_path=args.vector,
        dataset_path=args.dataset,
        splits=args.splits,
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fieldnames = list(results[0].keys())
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved per-row scores to {rows_path}")


if __name__ == "__main__":
    main()