#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from typing import Any

import requests
import torch
from tqdm.auto import tqdm


API_BASE = "https://www.neuronpedia.org/api/v1"


def load_vector(path: str) -> list[float]:
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
    return obj.detach().float().cpu().tolist()


def zscore(xs: list[float]) -> list[float]:
    if not xs:
        return xs
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    std = math.sqrt(var)
    if std == 0:
        return [0.0 for _ in xs]
    return [(x - mean) / std for x in xs]


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


def render_html(tokens: list[str], scores: list[float], title: str) -> str:
    spans = []
    for tok, score in zip(tokens, scores):
        token_text = html.escape(tok).replace("\n", "↵\n")
        bg = score_to_rgba(score)
        spans.append(
            f'<span title="{score:.3f}" '
            f'style="background:{bg}; padding:2px 1px; border-radius:3px;">'
            f"{token_text}</span>"
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
<div class="legend">
  <span class="box" style="background:rgba(220,60,60,0.75)"></span> positive activation
  &nbsp;&nbsp;
  <span class="box" style="background:rgba(70,110,220,0.75)"></span> negative activation
</div>
<div class="tokens">
{''.join(spans)}
</div>
</body>
</html>"""


def build_activation_request_payload(
    *,
    model_id: str,
    prompt: str,
    vector_values: list[float],
    layer: int,
    position_hint: str | None = None,
) -> dict[str, Any]:
    return {
        "modelId": model_id,
        "prompt": prompt,
        "customVector": {
            "vector": vector_values,
            "layer": layer,
            **({"positionHint": position_hint} if position_hint else {}),
        },
    }


def call_activation_single(
    *,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
    progress: tqdm,
) -> dict[str, Any]:
    progress.set_description("Calling Neuronpedia API")
    progress.write("Sending request to Neuronpedia. This step can take a while...")
    resp = requests.post(
        f"{API_BASE}/activation/single",
        headers={
            "Content-Type": "application/json",
            "X-SECRET-KEY": api_key,
        },
        json=payload,
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(
            f"Neuronpedia API error {resp.status_code}:\n{resp.text}"
        )
    return resp.json()


def extract_tokens_and_scores(response_json: dict[str, Any]) -> tuple[list[str], list[float]]:
    if "activation" in response_json:
        activation = response_json["activation"]
        tokens = activation.get("tokens")
        values = activation.get("values") or activation.get("activations")
        if tokens is not None and values is not None:
            return list(tokens), [float(x) for x in values]

    if "tokens" in response_json and "activations" in response_json:
        return list(response_json["tokens"]), [float(x) for x in response_json["activations"]]

    if "results" in response_json and response_json["results"]:
        first = response_json["results"][0]
        tokens = first.get("tokens")
        values = first.get("values") or first.get("activations")
        if tokens is not None and values is not None:
            return list(tokens), [float(x) for x in values]

    raise ValueError(
        "Could not find token activations in API response. "
        "Print the raw JSON once and adjust extract_tokens_and_scores()."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector", required=True, help="Path to local .pt vector file")
    parser.add_argument("--metadata", required=True, help="Path to vector metadata .json")
    parser.add_argument("--prompt-file", required=True, help="Path to text file containing the text to visualize")
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--model-id", default="llama3.1-8b-it", help="Neuronpedia model id")
    parser.add_argument("--api-key", default=os.environ.get("NEURONPEDIA_API_KEY"))
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Pass --api-key or set NEURONPEDIA_API_KEY")

    progress = tqdm(total=7, desc="Starting", unit="step")

    try:
        progress.set_description("Loading vector")
        vector_values = load_vector(args.vector)
        progress.update(1)

        progress.set_description("Loading metadata")
        metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
        progress.update(1)

        progress.set_description("Reading input text")
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        progress.update(1)

        progress.set_description("Building request payload")
        payload = build_activation_request_payload(
            model_id=args.model_id,
            prompt=prompt,
            vector_values=vector_values,
            layer=int(metadata["layer"]),
            position_hint=metadata.get("position"),
        )
        progress.update(1)

        response_json = call_activation_single(
            api_key=args.api_key,
            payload=payload,
            timeout=args.timeout,
            progress=progress,
        )
        progress.update(1)

        progress.set_description("Parsing activations")
        tokens, raw_scores = extract_tokens_and_scores(response_json)
        scores = zscore(raw_scores)
        progress.update(1)

        progress.set_description("Writing HTML")
        title = f"Neuronpedia activation | {args.model_id} | layer {metadata['layer']} | {metadata.get('position', 'vector')}"
        html_text = render_html(tokens, scores, title)

        out = Path(args.output_html)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html_text, encoding="utf-8")

        raw_json_out = out.with_suffix(".response.json")
        raw_json_out.write_text(
            json.dumps(response_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        progress.update(1)

    finally:
        progress.close()

    print(f"Saved visualization to {out}")
    print(f"Saved raw API response to {raw_json_out}")
    print(f"Scored {len(tokens)} tokens.")


if __name__ == "__main__":
    main()