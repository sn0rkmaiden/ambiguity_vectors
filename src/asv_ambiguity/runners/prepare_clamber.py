from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


DEFAULT_CLAMBER_URL = "https://raw.githubusercontent.com/zt991211/CLAMBER/main/clamber_benchmark.jsonl"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_download_file(url: str, dest: Path) -> Path:
    ensure_parent(dest)
    if dest.exists():
        return dest
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    dest.write_bytes(data)
    return dest


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def build_prompt_text(question: str, context: str) -> str:
    if context:
        return f"Context:\n{context}\n\nUser query:\n{question}"
    return f"User query:\n{question}"


def prepare_rows(raw_rows: list[dict]) -> list[dict]:
    prepared: list[dict] = []
    for idx, row in enumerate(raw_rows):
        question = normalize_text(row.get("question"))
        context = normalize_text(row.get("context"))
        clarifying_question = normalize_text(row.get("clarifying_question"))
        require_clarification = int(row.get("require_clarification", 0))
        category = normalize_text(row.get("category"))
        subclass = normalize_text(row.get("subclass"))

        prepared.append(
            {
                "example_id": f"clamber_{idx:05d}",
                "source_dataset": "CLAMBER",
                "question": question,
                "context": context,
                "prompt_text": build_prompt_text(question=question, context=context),
                "clarifying_question": clarifying_question,
                "require_clarification": require_clarification,
                "label": "ambiguous" if require_clarification == 1 else "clear",
                "category": category,
                "subclass": subclass,
            }
        )
    return prepared



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default="")
    parser.add_argument("--download-url", default=DEFAULT_CLAMBER_URL)
    parser.add_argument("--cache-raw", default="outputs/external/raw/clamber_benchmark.jsonl")
    parser.add_argument("--output-jsonl", default="outputs/external/clamber_official_v1.jsonl")
    args = parser.parse_args()

    if args.input_jsonl:
        raw_path = Path(args.input_jsonl)
    else:
        raw_path = maybe_download_file(args.download_url, Path(args.cache_raw))

    raw_rows = read_jsonl(raw_path)
    prepared_rows = prepare_rows(raw_rows)
    output_path = Path(args.output_jsonl)
    write_jsonl(output_path, prepared_rows)

    num_ambiguous = sum(r["require_clarification"] for r in prepared_rows)
    summary = {
        "source": str(raw_path.resolve()),
        "output": str(output_path.resolve()),
        "num_examples": len(prepared_rows),
        "num_ambiguous": num_ambiguous,
        "num_clear": len(prepared_rows) - num_ambiguous,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
