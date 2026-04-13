from __future__ import annotations

import argparse
import html
import json
import math
import random
from pathlib import Path
from typing import Any

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


def load_vector(path: str) -> torch.Tensor:
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


def build_full_text_from_dataset_row(
    *,
    model: HFCausalModel,
    row: dict,
    label: str,
) -> tuple[str, int]:
    if label not in row:
        raise ValueError(f"Label '{label}' not found in dataset row.")

    user_prompt = build_scoring_user_prompt(row)
    prompt_text = model.render_prompt(user_prompt)
    prompt_ids = model.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    prompt_token_count = int(prompt_ids.shape[1])

    response_text = row[label]
    full_text = prompt_text + response_text
    return full_text, prompt_token_count


def zscore(xs: list[float]) -> list[float]:
    if not xs:
        return xs
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    std = math.sqrt(var)
    if std == 0.0:
        return [0.0 for _ in xs]
    return [(x - mean) / std for x in xs]


def minmax_scale(xs: list[float]) -> list[float]:
    if not xs:
        return xs
    lo = min(xs)
    hi = max(xs)
    if hi == lo:
        return [0.0 for _ in xs]
    mid = (hi + lo) / 2.0
    half_range = (hi - lo) / 2.0
    return [(x - mid) / half_range for x in xs]


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def score_to_rgba(score: float, max_abs: float = 2.5) -> str:
    s = clip(score / max_abs, -1.0, 1.0)
    alpha = abs(s) * 0.85

    if s >= 0:
        r, g, b = 220, 60, 60
    else:
        r, g, b = 70, 110, 220

    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


def render_token_spans(tokens: list[str], scores: list[float]) -> str:
    parts = []
    for token, score in zip(tokens, scores):
        bg = score_to_rgba(score)
        token_text = html.escape(token).replace("\n", "↵\n")
        parts.append(
            f'<span title="{score:.3f}" '
            f'style="background:{bg}; padding:2px 1px; border-radius:3px;">'
            f"{token_text}</span>"
        )
    return "".join(parts)


def render_html(
    *,
    title: str,
    subtitle: str,
    sections: list[dict[str, Any]],
) -> str:
    section_html = []
    for sec in sections:
        section_html.append(
            f"""
            <div class="section">
              <h3>{html.escape(sec["header"])}</h3>
              <div class="meta">{html.escape(sec["meta"])}</div>
              <div class="response"><b>Response:</b> {html.escape(sec["response_text"])}</div>
              <div class="tokens">{render_token_spans(sec["tokens"], sec["scores"])}</div>
            </div>
            """
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{
  font-family: sans-serif;
  margin: 24px;
  line-height: 1.8;
}}
.tokens {{
  font-family: monospace;
  white-space: pre-wrap;
  word-break: break-word;
  margin-top: 10px;
}}
.legend {{
  margin-bottom: 16px;
}}
.box {{
  display:inline-block;
  width:18px;
  height:18px;
  vertical-align:middle;
  margin-right:6px;
  border-radius:3px;
}}
.section {{
  border-top: 1px solid #ddd;
  padding-top: 18px;
  margin-top: 18px;
}}
.meta {{
  color: #444;
  margin-bottom: 8px;
}}
.response {{
  margin-bottom: 8px;
}}
</style>
</head>
<body>
<h2>{html.escape(title)}</h2>
<p>{html.escape(subtitle)}</p>
<div class="legend">
  <span class="box" style="background:rgba(220,60,60,0.75)"></span> positive activation
  &nbsp;&nbsp;
  <span class="box" style="background:rgba(70,110,220,0.75)"></span> negative activation
</div>
{''.join(section_html)}
</body>
</html>"""


def score_tokens(
    *,
    model: HFCausalModel,
    text: str,
    vector: torch.Tensor,
    layer_idx: int,
) -> tuple[list[str], list[float]]:
    outputs = model.forward_hidden_states(text)
    input_ids = outputs["input_ids"][0].detach().cpu()
    hidden = outputs["hidden_states"][layer_idx][0].detach().float().cpu()

    if hidden.shape[1] != vector.shape[0]:
        raise ValueError(
            f"Vector dim {vector.shape[0]} does not match hidden dim {hidden.shape[1]}"
        )

    scores = torch.matmul(hidden, vector).tolist()
    tokens = [
        model.tokenizer.decode([int(tok.item())], skip_special_tokens=False)
        for tok in input_ids
    ]
    return tokens, scores


def save_scores_json(
    *,
    output_json: Path,
    metadata: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    payload = {
        "metadata": metadata,
        "sections": sections,
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def find_rows_by_ids(rows: list[dict], example_ids: list[str]) -> list[dict]:
    wanted = set(example_ids)
    found = [row for row in rows if row["example_id"] in wanted]
    found_ids = {row["example_id"] for row in found}
    missing = sorted(wanted - found_ids)
    if missing:
        raise ValueError(f"Example ids not found: {missing}")
    order = {eid: i for i, eid in enumerate(example_ids)}
    found.sort(key=lambda r: order[r["example_id"]])
    return found


def maybe_filter_special_tokens(
    *,
    tokens: list[str],
    raw_scores: list[float],
    display_scores: list[float],
    drop_special_tokens: bool,
) -> tuple[list[str], list[float], list[float]]:
    if not drop_special_tokens:
        return tokens, raw_scores, display_scores

    keep_tokens = []
    keep_raw = []
    keep_display = []

    for tok, raw, disp in zip(tokens, raw_scores, display_scores):
        stripped = tok.strip()
        is_special = stripped.startswith("<|") and stripped.endswith("|>")
        if is_special:
            continue
        keep_tokens.append(tok)
        keep_raw.append(raw)
        keep_display.append(disp)

    return keep_tokens, keep_raw, keep_display


def normalize_scores(raw_scores: list[float], mode: str) -> list[float]:
    if mode == "zscore":
        return zscore(raw_scores)
    if mode == "minmax":
        return minmax_scale(raw_scores)
    return raw_scores


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--vector", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-html", required=True)

    parser.add_argument("--example-ids", nargs="*", default=None)
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["positive_response", "negative_wrong_question"],
        choices=["positive_response", "negative_direct_answer", "negative_wrong_question"],
    )
    parser.add_argument(
        "--normalize",
        default="zscore",
        choices=["zscore", "minmax", "none"],
    )
    parser.add_argument(
        "--span",
        default="assistant_only",
        choices=["assistant_only", "full_sequence"],
    )
    parser.add_argument("--drop-special-tokens", action="store_true")
    args = parser.parse_args()

    log("Loading model config...")
    model_config = load_yaml(args.model_config)

    log("Loading model...")
    model = HFCausalModel(model_config)

    log("Loading vector...")
    vector = load_vector(args.vector)

    log("Loading metadata...")
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    layer_idx = int(metadata["layer"])
    position = metadata.get("position", "unknown")

    log("Loading dataset...")
    rows = read_jsonl(Path(args.dataset))

    allowed_splits = set(args.splits)
    rows = [row for row in rows if row["split"] in allowed_splits]
    if not rows:
        raise ValueError(f"No rows left after filtering by splits {args.splits}")

    if args.example_ids:
        selected_rows = find_rows_by_ids(rows, args.example_ids)
    else:
        rng = random.Random(args.seed)
        shuffled = rows[:]
        rng.shuffle(shuffled)
        selected_rows = shuffled[: args.num_examples]

    if not selected_rows:
        raise ValueError("No examples selected.")

    log(f"Scoring {len(selected_rows)} examples x {len(args.labels)} labels...")
    sections = []
    section_json = []

    total = len(selected_rows) * len(args.labels)
    with tqdm(total=total, desc="Visualizing", unit="sequence") as pbar:
        for row in selected_rows:
            for label in args.labels:
                full_text, prompt_token_count = build_full_text_from_dataset_row(
                    model=model,
                    row=row,
                    label=label,
                )

                tokens, raw_scores = score_tokens(
                    model=model,
                    text=full_text,
                    vector=vector,
                    layer_idx=layer_idx,
                )

                if args.span == "assistant_only":
                    tokens = tokens[prompt_token_count:]
                    raw_scores = raw_scores[prompt_token_count:]

                display_scores = normalize_scores(raw_scores, args.normalize)

                tokens, raw_scores, display_scores = maybe_filter_special_tokens(
                    tokens=tokens,
                    raw_scores=raw_scores,
                    display_scores=display_scores,
                    drop_special_tokens=args.drop_special_tokens,
                )

                header = f"{row['example_id']} | {label}"
                meta = (
                    f"split={row['split']} | topic={row.get('topic', '')} | "
                    f"ambiguity_type={row.get('ambiguity_type', '')}"
                )

                sections.append(
                    {
                        "header": header,
                        "meta": meta,
                        "response_text": row[label],
                        "tokens": tokens,
                        "scores": display_scores,
                    }
                )

                section_json.append(
                    {
                        "example_id": row["example_id"],
                        "label": label,
                        "split": row["split"],
                        "topic": row.get("topic", ""),
                        "ambiguity_type": row.get("ambiguity_type", ""),
                        "response_text": row[label],
                        "tokens": tokens,
                        "raw_scores": raw_scores,
                        "display_scores": display_scores,
                    }
                )

                pbar.update(1)

    title = f"Vector activations | layer {layer_idx} | {position}"
    subtitle = (
        f"examples={len(selected_rows)} | labels={args.labels} | "
        f"span={args.span} | normalize={args.normalize}"
    )

    log("Rendering HTML...")
    html_text = render_html(
        title=title,
        subtitle=subtitle,
        sections=sections,
    )

    out_html = Path(args.output_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_text, encoding="utf-8")

    out_json = out_html.with_suffix(".json")
    save_scores_json(
        output_json=out_json,
        metadata={
            "layer": layer_idx,
            "position": position,
            "model_name": model.model_name,
            "normalize": args.normalize,
            "span": args.span,
            "drop_special_tokens": args.drop_special_tokens,
            "dataset": args.dataset,
            "splits": args.splits,
            "labels": args.labels,
            "selected_example_ids": [row["example_id"] for row in selected_rows],
        },
        sections=section_json,
    )

    log(f"Saved visualization to {out_html}")
    log(f"Saved token scores to {out_json}")
    log(f"Rendered {len(sections)} sections.")


if __name__ == "__main__":
    main()