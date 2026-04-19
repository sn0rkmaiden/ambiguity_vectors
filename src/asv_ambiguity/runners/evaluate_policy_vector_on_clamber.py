from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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



def score_prompt(
    *,
    model: HFCausalModel,
    vector: torch.Tensor,
    user_prompt: str,
    layer_idx: int,
    position: str,
) -> float:
    prompt_text = model.render_prompt(user_prompt)
    outputs = model.forward_hidden_states(prompt_text)
    input_ids = outputs["input_ids"].detach().cpu()
    hidden = outputs["hidden_states"][layer_idx][0].detach().float().cpu()
    prompt_token_count = int(input_ids.shape[1])
    rep = select_hidden_representation(
        hidden=hidden,
        input_ids=input_ids,
        prompt_token_count=prompt_token_count,
        tokenizer=model.tokenizer,
        mode=position,
    ).detach().float().cpu()
    return float(torch.dot(rep, vector))



def _auc_from_scores(scores: list[float], labels: list[int]) -> float | None:
    pos = [(s, y) for s, y in zip(scores, labels) if y == 1]
    neg = [(s, y) for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None

    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    ranks: list[float] = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    pos_rank_sum = sum(rank for rank, (_, y) in zip(ranks, pairs) if y == 1)
    n_pos = len(pos)
    n_neg = len(neg)
    u = pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)
    return float(u / (n_pos * n_neg))



def _threshold_candidates(scores: list[float]) -> list[float]:
    uniq = sorted(set(scores))
    if not uniq:
        return [0.0]
    if len(uniq) == 1:
        return uniq
    mids = [(a + b) / 2.0 for a, b in zip(uniq[:-1], uniq[1:])]
    return [uniq[0] - 1e-6] + mids + [uniq[-1] + 1e-6]



def _metrics_at_threshold(scores: list[float], labels: list[int], threshold: float) -> dict[str, float]:
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(p == 1 and y == 1 for p, y in zip(preds, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(preds, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(preds, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(preds, labels))

    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    balanced_accuracy = 0.5 * (recall + specificity)
    overclarification = fp / max(fp + tn, 1)
    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall_ambiguous": recall,
        "specificity_clear": specificity,
        "precision_ambiguous": precision,
        "f1_ambiguous": f1,
        "balanced_accuracy": balanced_accuracy,
        "overclarification_rate": overclarification,
    }



def _best_threshold_metrics(scores: list[float], labels: list[int]) -> dict[str, float]:
    candidates = _threshold_candidates(scores)
    all_metrics = [_metrics_at_threshold(scores, labels, thr) for thr in candidates]
    return max(all_metrics, key=lambda x: (x["balanced_accuracy"], x["f1_ambiguous"], -abs(x["threshold"])))



def render_html_report(
    *,
    title: str,
    summary: dict,
    category_rows: list[dict],
    top_high: list[dict],
    top_low: list[dict],
) -> str:
    def render_examples(items: list[dict]) -> str:
        parts = []
        for item in items:
            parts.append(
                "<div class='card'>"
                f"<div><b>{html.escape(item['example_id'])}</b> | score={item['score']:.4f} | label={html.escape(item['label'])}</div>"
                f"<div class='small'>category={html.escape(item['category'])} | subclass={html.escape(item['subclass'])}</div>"
                f"<pre>{html.escape(item['prompt_text'])}</pre>"
                "</div>"
            )
        return "".join(parts)

    category_table = [
        "<table border='1' cellpadding='6' cellspacing='0'><tr><th>category</th><th>subclass</th><th>count</th><th>mean score</th></tr>"
    ]
    for row in category_rows:
        category_table.append(
            f"<tr><td>{html.escape(row['category'])}</td><td>{html.escape(row['subclass'])}</td>"
            f"<td>{row['count']}</td><td>{row['mean_score']:.4f}</td></tr>"
        )
    category_table.append("</table>")

    return "".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'>",
            f"<title>{html.escape(title)}</title>",
            """<style>
            body { font-family: sans-serif; margin: 24px; line-height: 1.5; }
            pre { background: #f7f7f7; padding: 12px; white-space: pre-wrap; }
            .card { border-top: 1px solid #ddd; padding-top: 14px; margin-top: 14px; }
            .small { color: #555; }
            table { border-collapse: collapse; margin-top: 12px; }
            td, th { text-align: left; }
            </style></head><body>""",
            f"<h2>{html.escape(title)}</h2>",
            f"<pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>",
            "<h3>Ambiguous categories</h3>",
            "".join(category_table),
            "<h3>Top high-scoring prompts</h3>",
            render_examples(top_high),
            "<h3>Top low-scoring prompts</h3>",
            render_examples(top_low),
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
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/clamber_eval")
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = HFCausalModel(load_yaml(args.model_config))
    dataset_rows = read_jsonl(Path(args.dataset))
    if args.max_rows > 0:
        dataset_rows = dataset_rows[: args.max_rows]

    vector = load_vector(Path(args.vector))
    metadata = load_metadata(Path(args.metadata))
    layer_idx = int(args.layer if args.layer is not None else metadata["layer"])
    position = str(args.position)

    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    labels: list[int] = []

    category_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    label_counts = Counter()

    for row in tqdm(dataset_rows, desc="Scoring CLAMBER prompts", unit="prompt"):
        score = score_prompt(
            model=model,
            vector=vector,
            user_prompt=row["prompt_text"],
            layer_idx=layer_idx,
            position=position,
        )
        require_clarification = int(row["require_clarification"])
        label_counts[require_clarification] += 1
        if require_clarification == 1:
            category_scores[(row.get("category", ""), row.get("subclass", ""))].append(score)

        out_row = {
            "example_id": row["example_id"],
            "label": row.get("label", "ambiguous" if require_clarification else "clear"),
            "require_clarification": require_clarification,
            "category": row.get("category", ""),
            "subclass": row.get("subclass", ""),
            "score": score,
            "question": row.get("question", ""),
            "context": row.get("context", ""),
            "prompt_text": row["prompt_text"],
        }
        rows.append(out_row)
        scores.append(score)
        labels.append(require_clarification)

    ambiguous_scores = [s for s, y in zip(scores, labels) if y == 1]
    clear_scores = [s for s, y in zip(scores, labels) if y == 0]
    best = _best_threshold_metrics(scores, labels)
    auc = _auc_from_scores(scores, labels)

    category_rows = [
        {
            "category": category,
            "subclass": subclass,
            "count": len(vals),
            "mean_score": sum(vals) / len(vals),
        }
        for (category, subclass), vals in category_scores.items()
        if vals
    ]
    category_rows.sort(key=lambda x: x["mean_score"], reverse=True)

    summary = {
        "mode": "prompt_only_policy_probe",
        "dataset": str(Path(args.dataset).resolve()),
        "vector": str(Path(args.vector).resolve()),
        "metadata": str(Path(args.metadata).resolve()),
        "model_name": model.model_name,
        "layer": layer_idx,
        "position": position,
        "num_examples": len(rows),
        "num_ambiguous": label_counts[1],
        "num_clear": label_counts[0],
        "mean_score_ambiguous": (sum(ambiguous_scores) / len(ambiguous_scores)) if ambiguous_scores else None,
        "mean_score_clear": (sum(clear_scores) / len(clear_scores)) if clear_scores else None,
        "auc_ambiguous_vs_clear": auc,
        "best_threshold_metrics": best,
    }

    stem = f"clamber_policy_probe__{Path(args.vector).stem}__{position}__layer{layer_idx}"
    write_json(output_dir / f"{stem}__summary.json", summary)
    write_csv(output_dir / f"{stem}__rows.csv", rows)

    rows_by_score_desc = sorted(rows, key=lambda x: x["score"], reverse=True)
    rows_by_score_asc = list(reversed(rows_by_score_desc))
    top_high = rows_by_score_desc[: args.top_k]
    top_low = rows_by_score_asc[: args.top_k]
    html_report = render_html_report(
        title=f"CLAMBER policy probe | layer {layer_idx} | {position}",
        summary=summary,
        category_rows=category_rows,
        top_high=top_high,
        top_low=top_low,
    )
    (output_dir / f"{stem}__report.html").write_text(html_report, encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved summary to {output_dir / (stem + '__summary.json')}")
    print(f"Saved rows to {output_dir / (stem + '__rows.csv')}")
    print(f"Saved HTML report to {output_dir / (stem + '__report.html')}")


if __name__ == "__main__":
    main()
