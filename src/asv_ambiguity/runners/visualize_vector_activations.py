from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import torch

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


def render_html(
    *,
    tokens: list[str],
    scores: list[float],
    title: str,
    subtitle: str | None = None,
) -> str:
    parts = []
    for token, score in zip(tokens, scores):
        bg = score_to_rgba(score)
        token_text = html.escape(token).replace("\n", "↵\n")
        parts.append(
            f'<span title="{score:.3f}" '
            f'style="background:{bg}; padding:2px 1px; border-radius:3px;">'
            f"{token_text}</span>"
        )

    subtitle_html = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""

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
</style>
</head>
<body>
<h2>{html.escape(title)}</h2>
{subtitle_html}
<div class="legend">
  <span class="box" style="background:rgba(220,60,60,0.75)"></span> positive activation
  &nbsp;&nbsp;
  <span class="box" style="background:rgba(70,110,220,0.75)"></span> negative activation
</div>
<div class="tokens">
{''.join(parts)}
</div>
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
    tokens: list[str],
    raw_scores: list[float],
    display_scores: list[float],
    metadata: dict[str, Any],
) -> None:
    payload = {
        "metadata": metadata,
        "tokens": tokens,
        "raw_scores": raw_scores,
        "display_scores": display_scores,
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def find_row(rows: list[dict], example_id: str) -> dict:
    for row in rows:
        if row["example_id"] == example_id:
            return row
    raise ValueError(f"Example id '{example_id}' not found.")


def log(msg: str) -> None:
    print(msg, flush=True)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--vector", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-html", required=True)

    parser.add_argument("--text-file", default=None)

    parser.add_argument("--dataset", default=None)
    parser.add_argument("--example-id", default=None)
    parser.add_argument(
        "--label",
        default="positive_response",
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
    parser.add_argument(
        "--drop-special-tokens",
        action="store_true",
    )
    args = parser.parse_args()

    using_text_file = args.text_file is not None
    using_dataset = args.dataset is not None or args.example_id is not None

    if using_text_file and using_dataset:
        raise ValueError("Use either --text-file or (--dataset + --example-id), not both.")
    if not using_text_file and not (args.dataset and args.example_id):
        raise ValueError("Provide either --text-file OR both --dataset and --example-id.")

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

    log("Preparing text...")
    subtitle = None
    prompt_token_count = 0

    if args.text_file is not None:
        text = Path(args.text_file).read_text(encoding="utf-8")
        subtitle = f"Source: {args.text_file} | span={args.span}"
    else:
        rows = read_jsonl(Path(args.dataset))
        row = find_row(rows, args.example_id)
        text, prompt_token_count = build_full_text_from_dataset_row(
            model=model,
            row=row,
            label=args.label,
        )
        subtitle = (
            f"example_id={row['example_id']} | split={row['split']} | "
            f"ambiguity_type={row.get('ambiguity_type', '')} | label={args.label} | "
            f"span={args.span}"
        )

    log(f"Computing token activations at layer {layer_idx} ({position})...")
    tokens, raw_scores = score_tokens(
        model=model,
        text=text,
        vector=vector,
        layer_idx=layer_idx,
    )

    if args.span == "assistant_only":
        if args.text_file is not None:
            raise ValueError("--span assistant_only requires --dataset and --example-id.")
        tokens = tokens[prompt_token_count:]
        raw_scores = raw_scores[prompt_token_count:]

    log("Normalizing scores...")
    if args.normalize == "zscore":
        display_scores = zscore(raw_scores)
    elif args.normalize == "minmax":
        display_scores = minmax_scale(raw_scores)
    else:
        display_scores = raw_scores

    tokens, raw_scores, display_scores = maybe_filter_special_tokens(
        tokens=tokens,
        raw_scores=raw_scores,
        display_scores=display_scores,
        drop_special_tokens=args.drop_special_tokens,
    )

    log("Rendering HTML...")
    title = f"Vector activations | layer {layer_idx} | {position}"
    html_text = render_html(
        tokens=tokens,
        scores=display_scores,
        title=title,
        subtitle=subtitle,
    )

    out_html = Path(args.output_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_text, encoding="utf-8")

    out_json = out_html.with_suffix(".json")
    save_scores_json(
        output_json=out_json,
        tokens=tokens,
        raw_scores=raw_scores,
        display_scores=display_scores,
        metadata={
            "layer": layer_idx,
            "position": position,
            "model_name": model.model_name,
            "normalize": args.normalize,
            "span": args.span,
            "drop_special_tokens": args.drop_special_tokens,
            "text_source": args.text_file if args.text_file else args.dataset,
            "example_id": args.example_id,
            "label": args.label if args.dataset else None,
        },
    )

    log(f"Saved visualization to {out_html}")
    log(f"Saved token scores to {out_json}")
    log(f"Scored {len(tokens)} tokens.")


if __name__ == "__main__":
    main()