
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm


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


def load_all_vectors(vector_dir: Path, layer: int) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
    vectors = {}
    metadata = {}
    for pt_path in sorted(vector_dir.glob(f"*__layer{layer}.pt")):
        concept = pt_path.stem.rsplit("__layer", 1)[0]
        vectors[concept] = load_vector(pt_path)
        meta_path = pt_path.with_suffix(".json")
        if meta_path.exists():
            metadata[concept] = json.loads(meta_path.read_text(encoding="utf-8"))
    if not vectors:
        raise ValueError(f"No vectors found in {vector_dir} for layer {layer}")
    return vectors, metadata


def render_matrix_html(labels: list[str], matrix: dict[str, dict[str, int]]) -> str:
    header = "".join(f"<th>{html.escape(x)}</th>" for x in ["true \\ pred"] + labels)
    rows = []
    for true_label in labels:
        cells = [f"<td><b>{html.escape(true_label)}</b></td>"]
        for pred_label in labels:
            cells.append(f"<td>{matrix.get(true_label, {}).get(pred_label, 0)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table border="1" cellpadding="6" cellspacing="0"><tr>{header}</tr>{"".join(rows)}</table>'


def render_html_report(
    *,
    title: str,
    summary: dict,
    confusion_labels: list[str] | None,
    confusion_matrix: dict[str, dict[str, int]] | None,
    top_examples_by_concept: dict[str, list[dict]],
) -> str:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        """<style>
        body { font-family: sans-serif; margin: 24px; line-height: 1.5; }
        pre { background: #f7f7f7; padding: 12px; overflow-x: auto; white-space: pre-wrap; }
        .card { border-top: 1px solid #ddd; padding-top: 16px; margin-top: 16px; }
        .small { color: #555; }
        table { border-collapse: collapse; margin-top: 12px; }
        td, th { text-align: left; }
        </style></head><body>""",
        f"<h2>{html.escape(title)}</h2>",
        "<h3>Summary</h3>",
        f"<pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>",
    ]

    if confusion_labels and confusion_matrix:
        parts.append("<h3>Confusion matrix</h3>")
        parts.append(render_matrix_html(confusion_labels, confusion_matrix))

    if top_examples_by_concept:
        parts.append("<h3>Top activating held-out examples by concept</h3>")
        for concept, items in top_examples_by_concept.items():
            parts.append(f"<div class='card'><h4>{html.escape(concept)}</h4>")
            for item in items:
                meta = item.get("meta", "")
                text = item.get("text", "")
                score = item.get("score")
                parts.append(
                    f"<div style='margin-bottom:14px;'>"
                    f"<div class='small'>score={score:.4f} | {html.escape(meta)}</div>"
                    f"<pre>{html.escape(text)}</pre>"
                    f"</div>"
                )
            parts.append("</div>")

    parts.append("</body></html>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", required=True, help="concept-pooled activations .pt")
    parser.add_argument("--vector-dir", required=True, help="directory containing concept vectors")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--split", nargs="+", default=["val", "test"])
    parser.add_argument("--output-dir", default="outputs/concept_eval")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = torch.load(args.activations, map_location="cpu")
    if bundle.get("kind") != "concept_pooled":
        raise ValueError("This script expects concept-pooled activations produced by collect_concept_pooled_activations.py")

    vectors, _vector_meta = load_all_vectors(Path(args.vector_dir), args.layer)
    concepts = sorted(vectors.keys())

    records = [r for r in bundle["records"] if r["split"] in set(args.split)]
    if not records:
        raise ValueError(f"No records found for splits {args.split}")

    rows = []
    confusion = defaultdict(lambda: defaultdict(int))
    per_concept_total = Counter()
    per_concept_correct = Counter()

    top_examples_by_concept: dict[str, list[dict]] = {c: [] for c in concepts}

    for record in tqdm(records, desc="Scoring held-out concept texts", unit="text"):
        h = record["layers"][args.layer].float().cpu()
        scores = {concept: float(torch.dot(h, vec)) for concept, vec in vectors.items()}
        pred = max(scores, key=scores.get)
        true = record["concept"]

        confusion[true][pred] += 1
        per_concept_total[true] += 1
        per_concept_correct[true] += int(pred == true)

        row = {
            "record_id": record["record_id"],
            "split": record["split"],
            "topic": record.get("topic", ""),
            "ambiguity_type": record.get("ambiguity_type", ""),
            "true_concept": true,
            "pred_concept": pred,
            **{f"score_{c}": scores[c] for c in concepts},
        }
        rows.append(row)

        for concept in concepts:
            top_examples_by_concept[concept].append(
                {
                    "score": scores[concept],
                    "text": record["text"],
                    "meta": f"record_id={record['record_id']} | true={true} | pred={pred} | split={record['split']}",
                }
            )

    for concept in concepts:
        top_examples_by_concept[concept].sort(key=lambda x: x["score"], reverse=True)
        top_examples_by_concept[concept] = top_examples_by_concept[concept][: args.top_k]

    overall_acc = sum(per_concept_correct.values()) / max(sum(per_concept_total.values()), 1)
    per_concept_acc = {
        c: (per_concept_correct[c] / per_concept_total[c]) if per_concept_total[c] else None
        for c in concepts
    }

    summary = {
        "mode": "held_out_classification",
        "layer": args.layer,
        "splits": args.split,
        "num_examples": len(records),
        "concepts": concepts,
        "overall_accuracy": overall_acc,
        "per_concept_accuracy": per_concept_acc,
        "vector_dir": str(Path(args.vector_dir).resolve()),
        "activations": str(Path(args.activations).resolve()),
    }

    stem = f"concept_eval__layer{args.layer}__{'-'.join(args.split)}"
    write_json(output_dir / f"{stem}__summary.json", summary)
    write_csv(output_dir / f"{stem}__rows.csv", rows)

    html_report = render_html_report(
        title=f"Concept vector evaluation | layer {args.layer}",
        summary=summary,
        confusion_labels=concepts,
        confusion_matrix=confusion,
        top_examples_by_concept=top_examples_by_concept,
    )
    (output_dir / f"{stem}__report.html").write_text(html_report, encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved summary to {output_dir / (stem + '__summary.json')}")
    print(f"Saved rows to {output_dir / (stem + '__rows.csv')}")
    print(f"Saved HTML report to {output_dir / (stem + '__report.html')}")


if __name__ == "__main__":
    main()
