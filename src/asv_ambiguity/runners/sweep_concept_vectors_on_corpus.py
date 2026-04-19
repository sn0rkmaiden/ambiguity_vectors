
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def load_all_vectors(vector_dir: Path, layer: int) -> dict[str, torch.Tensor]:
    vectors = {}
    for pt_path in sorted(vector_dir.glob(f"*__layer{layer}.pt")):
        concept = pt_path.stem.rsplit("__layer", 1)[0]
        vectors[concept] = load_vector(pt_path)
    if not vectors:
        raise ValueError(f"No vectors found in {vector_dir} for layer {layer}")
    return vectors


def pooled_representation(hidden: torch.Tensor, burn_in_tokens: int) -> tuple[torch.Tensor, int]:
    seq_len = int(hidden.shape[0])
    start = min(burn_in_tokens, max(seq_len - 1, 0))
    return hidden[start:].mean(dim=0), start


def render_html_report(title: str, summary: dict, top_examples_by_concept: dict[str, list[dict]]) -> str:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        """<style>
        body { font-family: sans-serif; margin: 24px; line-height: 1.5; }
        pre { background: #f7f7f7; padding: 12px; overflow-x: auto; white-space: pre-wrap; }
        .card { border-top: 1px solid #ddd; padding-top: 16px; margin-top: 16px; }
        .small { color: #555; }
        </style></head><body>""",
        f"<h2>{html.escape(title)}</h2>",
        f"<pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>",
    ]
    for concept, items in top_examples_by_concept.items():
        parts.append(f"<div class='card'><h3>{html.escape(concept)}</h3>")
        for item in items:
            parts.append(
                f"<div class='small'>score={item['score']:.4f} | row_index={item['row_index']}</div>"
                f"<pre>{html.escape(item['text'])}</pre>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--vector-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--external-jsonl", required=True)
    parser.add_argument("--external-text-key", default="text")
    parser.add_argument("--burn-in-tokens", type=int, default=30)
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--output-dir", default="outputs/concept_eval_external")
    args = parser.parse_args()

    model = HFCausalModel(load_yaml(args.model_config))
    vectors = load_all_vectors(Path(args.vector_dir), args.layer)
    concepts = sorted(vectors.keys())

    rows = read_jsonl(Path(args.external_jsonl))[: args.max_rows]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    top_examples_by_concept = {c: [] for c in concepts}

    for i, row in enumerate(tqdm(rows, desc="Sweeping external corpus", unit="doc")):
        text = row.get(args.external_text_key)
        if not isinstance(text, str) or not text.strip():
            continue

        outputs = model.forward_hidden_states(text)
        hidden = outputs["hidden_states"][args.layer][0].detach().cpu().float()
        pooled, used_start = pooled_representation(hidden, args.burn_in_tokens)

        scores = {c: float(torch.dot(pooled, v)) for c, v in vectors.items()}

        for concept in concepts:
            top_examples_by_concept[concept].append(
                {
                    "row_index": i,
                    "score": scores[concept],
                    "text": text,
                    "used_start_token": used_start,
                }
            )

    for concept in concepts:
        top_examples_by_concept[concept].sort(key=lambda x: x["score"], reverse=True)
        top_examples_by_concept[concept] = top_examples_by_concept[concept][: args.top_k]

    summary = {
        "mode": "external_corpus_sweep",
        "layer": args.layer,
        "num_rows_scanned": min(len(rows), args.max_rows),
        "concepts": concepts,
        "external_jsonl": str(Path(args.external_jsonl).resolve()),
        "vector_dir": str(Path(args.vector_dir).resolve()),
        "burn_in_tokens": args.burn_in_tokens,
    }

    stem = f"external_sweep__layer{args.layer}"
    write_json(out_dir / f"{stem}__summary.json", summary)
    write_json(out_dir / f"{stem}__top_examples.json", top_examples_by_concept)
    html_report = render_html_report(
        title=f"External corpus sweep | layer {args.layer}",
        summary=summary,
        top_examples_by_concept=top_examples_by_concept,
    )
    (out_dir / f"{stem}__report.html").write_text(html_report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved HTML report to {out_dir / (stem + '__report.html')}")


if __name__ == "__main__":
    main()
