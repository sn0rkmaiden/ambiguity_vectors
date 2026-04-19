from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from asv_ambiguity.activations.positions import select_hidden_representation
from asv_ambiguity.config import load_yaml
from asv_ambiguity.models.hf import HFCausalModel



def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows



def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")



def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def load_vector(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "vector" in obj:
            obj = obj["vector"]
        elif "direction" in obj:
            obj = obj["direction"]
        else:
            raise ValueError(f"Unsupported vector file structure in {path}")
    if not isinstance(obj, torch.Tensor):
        raise ValueError(f"Expected tensor in {path}, got {type(obj)}")
    return obj.detach().float().cpu()



def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def score_prompt(model: HFCausalModel, vector: torch.Tensor, user_prompt: str, layer: int, position: str) -> float:
    prompt_text = model.render_prompt(user_prompt)
    outputs = model.forward_hidden_states(prompt_text)
    input_ids = outputs["input_ids"].detach().cpu()
    hidden = outputs["hidden_states"][layer][0].detach().float().cpu()
    rep = select_hidden_representation(
        hidden=hidden,
        input_ids=input_ids,
        prompt_token_count=int(input_ids.shape[1]),
        tokenizer=model.tokenizer,
        mode=position,
    ).detach().float().cpu()
    return float(torch.dot(rep, vector))



def render_html_report(summary: dict, aggregate_rows: list[dict], detail_rows: list[dict]) -> str:
    agg_table = ["<table border='1' cellpadding='6' cellspacing='0'><tr><th>family</th><th>ambiguity_type</th><th>strength</th><th>count</th><th>mean score</th></tr>"]
    for row in aggregate_rows:
        agg_table.append(
            f"<tr><td>{html.escape(row['family'])}</td><td>{html.escape(row['ambiguity_type'])}</td>"
            f"<td>{row['ambiguity_strength']}</td><td>{row['count']}</td><td>{row['mean_score']:.4f}</td></tr>"
        )
    agg_table.append("</table>")

    detail_cards = []
    for row in detail_rows:
        detail_cards.append(
            "<div class='card'>"
            f"<div><b>{html.escape(row['example_id'])}</b> | family={html.escape(row['family'])} | strength={row['ambiguity_strength']} | score={row['score']:.4f}</div>"
            f"<pre>{html.escape(row['prompt_text'])}</pre>"
            "</div>"
        )

    return "".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<style>body { font-family: sans-serif; margin: 24px; line-height: 1.5; } pre { background:#f7f7f7; padding:12px; white-space:pre-wrap;} .card { border-top:1px solid #ddd; padding-top:14px; margin-top:14px; }</style>",
            "</head><body>",
            "<h2>Controlled ambiguity sweep evaluation</h2>",
            f"<pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>",
            "<h3>Aggregate means</h3>",
            "".join(agg_table),
            "<h3>Per-example details</h3>",
            "".join(detail_cards),
            "</body></html>",
        ]
    )



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--vector", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--position", default="last_token")
    parser.add_argument("--output-dir", default="outputs/controlled_sweeps_eval")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = HFCausalModel(load_yaml(args.model_config))
    dataset_rows = read_jsonl(Path(args.dataset))
    vector = load_vector(Path(args.vector))
    metadata = load_metadata(Path(args.metadata))
    layer_idx = int(args.layer if args.layer is not None else metadata["layer"])
    position = str(args.position)

    detail_rows: list[dict] = []
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)

    for row in tqdm(dataset_rows, desc="Scoring controlled sweeps", unit="prompt"):
        score = score_prompt(model, vector, row["prompt_text"], layer_idx, position)
        detail = dict(row)
        detail["score"] = score
        detail_rows.append(detail)
        grouped[(row["family"], row["ambiguity_type"], int(row["ambiguity_strength"]))].append(score)

    aggregate_rows = []
    for (family, ambiguity_type, strength), vals in sorted(grouped.items()):
        aggregate_rows.append(
            {
                "family": family,
                "ambiguity_type": ambiguity_type,
                "ambiguity_strength": strength,
                "count": len(vals),
                "mean_score": sum(vals) / len(vals),
            }
        )

    family_means: dict[str, list[float]] = defaultdict(list)
    monotonic_families = 0
    for row in aggregate_rows:
        family_means[row["family"]].append(row["mean_score"])
    for means in family_means.values():
        if all(b >= a for a, b in zip(means[:-1], means[1:])):
            monotonic_families += 1

    summary = {
        "dataset": str(Path(args.dataset).resolve()),
        "vector": str(Path(args.vector).resolve()),
        "metadata": str(Path(args.metadata).resolve()),
        "model_name": model.model_name,
        "layer": layer_idx,
        "position": position,
        "num_examples": len(detail_rows),
        "num_families": len(family_means),
        "num_monotonic_families": monotonic_families,
    }

    stem = f"controlled_sweeps__{Path(args.vector).stem}__{position}__layer{layer_idx}"
    write_json(output_dir / f"{stem}__summary.json", summary)
    write_csv(output_dir / f"{stem}__aggregate.csv", aggregate_rows)
    write_csv(output_dir / f"{stem}__rows.csv", detail_rows)
    html_report = render_html_report(summary, aggregate_rows, detail_rows)
    (output_dir / f"{stem}__report.html").write_text(html_report, encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved summary to {output_dir / (stem + '__summary.json')}")
    print(f"Saved aggregate rows to {output_dir / (stem + '__aggregate.csv')}")
    print(f"Saved detail rows to {output_dir / (stem + '__rows.csv')}")
    print(f"Saved HTML report to {output_dir / (stem + '__report.html')}")


if __name__ == "__main__":
    main()
