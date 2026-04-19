from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from tqdm.auto import tqdm

from asv_ambiguity.config import load_yaml
from asv_ambiguity.models.hf import HFCausalModel
from asv_ambiguity.utils.io import write_jsonl
from asv_ambiguity.utils.seed import set_seed


class SplitAssigner:
    def __init__(self, train_ratio: float, val_ratio: float):
        if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
            raise ValueError("Invalid split ratios.")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def assign(self, idx: int, total: int) -> str:
        frac = (idx + 1) / max(total, 1)
        if frac <= self.train_ratio:
            return "train"
        if frac <= self.train_ratio + self.val_ratio:
            return "val"
        return "test"


def resolve_project_path(config_path: str | Path, maybe_relative_path: str | Path) -> Path:
    base = Path(config_path).resolve().parents[2]
    path = Path(maybe_relative_path)
    return path if path.is_absolute() else (base / path)


def slugify(text: str) -> str:
    text = text.strip().replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_")


def build_output_path(
    *,
    data_config_path: str,
    configured_output_path: str,
    model_name: str,
    num_concepts: int,
    generations_per_seed_per_concept: int,
    seed: int,
) -> Path:
    base_output_path = resolve_project_path(data_config_path, configured_output_path)
    output_dir = base_output_path.parent
    filename = (
        f"concept_corpus_v1"
        f"__{slugify(model_name)}"
        f"__c{num_concepts}"
        f"__g{generations_per_seed_per_concept}"
        f"__seed{seed}.jsonl"
    )
    return output_dir / filename


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def clean_generated_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*assistant\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*output\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def eligible_concepts_for_seed(seed_row: dict, concepts: list[str]) -> list[str]:
    ambiguity_type = seed_row["ambiguity_type"]
    out = []

    if "ask_question_vs_answer" in concepts:
        out.append("ask_question_vs_answer")

    if ambiguity_type == "object_identity" and "referent_clarification" in concepts:
        out.append("referent_clarification")
    if ambiguity_type == "preference" and "preference_clarification" in concepts:
        out.append("preference_clarification")
    if ambiguity_type == "destination" and "destination_clarification" in concepts:
        out.append("destination_clarification")

    return out


def build_concept_prompt(
    *,
    concept: str,
    seed_row: dict,
    min_words: int,
    max_words: int,
) -> str:
    candidate_values = seed_row.get("candidate_values", [])
    candidates_block = "\n".join(f"- {x}" for x in candidate_values)

    base = f"""Write a short natural text of about {min_words}-{max_words} words.
It should be plain text only, no bullet points, no JSON, no headings.

Use this scenario:

Topic: {seed_row['topic']}
Ambiguity type: {seed_row['ambiguity_type']}
Context: {seed_row['context']}
Instruction: {seed_row['instruction']}
Candidate values:
{candidates_block}
"""

    if concept == "ask_question_vs_answer":
        return base + """
The text should make it clear that the assistant chooses to ask a clarifying question instead of taking action immediately.
Make the question natural and useful.
The concept should be apparent throughout the text, not only at the end.
Output only the text.
""".strip()

    if concept == "referent_clarification":
        return base + """
The text should make it clear that the missing information is which object or referent is intended.
The assistant should ask a specific clarifying question that resolves that referent ambiguity.
The concept should be apparent throughout the text, not only at the end.
Output only the text.
""".strip()

    if concept == "preference_clarification":
        return base + """
The text should make it clear that the missing information is the user's preference or option choice.
The assistant should ask a specific clarifying question that resolves that preference ambiguity.
The concept should be apparent throughout the text, not only at the end.
Output only the text.
""".strip()

    if concept == "destination_clarification":
        return base + """
The text should make it clear that the missing information is the intended destination or placement.
The assistant should ask a specific clarifying question that resolves that destination ambiguity.
The concept should be apparent throughout the text, not only at the end.
Output only the text.
""".strip()

    raise ValueError(f"Unsupported concept: {concept}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-config", required=True)
    args = parser.parse_args()

    model_config = load_yaml(args.model_config)
    data_config = load_yaml(args.data_config)

    seed = int(data_config["sampling"].get("random_seed", 42))
    set_seed(seed)

    model = HFCausalModel(model_config)
    model_name = getattr(model, "model_name", model_config["model"]["name"])

    seed_path = resolve_project_path(args.data_config, data_config["input"]["seed_jsonl"])
    seed_rows = read_jsonl(seed_path)

    concepts = [str(x) for x in data_config["concepts"]]
    min_words = int(data_config["generation"]["min_words"])
    max_words = int(data_config["generation"]["max_words"])
    generations_per_seed_per_concept = int(data_config["sampling"]["generations_per_seed_per_concept"])
    max_attempts = int(data_config["sampling"].get("max_attempts_per_example", 5))

    rng = random.Random(seed)
    shuffled_seed_rows = seed_rows[:]
    rng.shuffle(shuffled_seed_rows)

    split_assigner = SplitAssigner(
        train_ratio=float(data_config["splits"]["train_ratio"]),
        val_ratio=float(data_config["splits"]["val_ratio"]),
    )

    rows = []
    total_seed_items = len(shuffled_seed_rows)
    record_counter = 0

    estimated_total = 0
    for seed_row in shuffled_seed_rows:
        estimated_total += len(eligible_concepts_for_seed(seed_row, concepts)) * generations_per_seed_per_concept

    progress = tqdm(total=estimated_total, desc="Generating concept corpus", unit="record")

    try:
        for seed_idx, seed_row in enumerate(shuffled_seed_rows):
            split = split_assigner.assign(seed_idx, total_seed_items)
            concepts_for_seed = eligible_concepts_for_seed(seed_row, concepts)

            for concept in concepts_for_seed:
                for gen_idx in range(generations_per_seed_per_concept):
                    prompt = build_concept_prompt(
                        concept=concept,
                        seed_row=seed_row,
                        min_words=min_words,
                        max_words=max_words,
                    )

                    last_error = None
                    for attempt in range(max_attempts):
                        try:
                            raw = model.generate_text(prompt)
                            text = clean_generated_text(raw)

                            wc = word_count(text)
                            if wc < min_words:
                                raise ValueError(f"Generated text too short: {wc} words")

                            row = {
                                "record_id": f"concept_{record_counter:05d}",
                                "seed_row_index": seed_idx,
                                "concept": concept,
                                "topic": seed_row["topic"],
                                "ambiguity_type": seed_row["ambiguity_type"],
                                "gold_missing_slot": seed_row["gold_missing_slot"],
                                "split": split,
                                "text": text,
                                "metadata": {
                                    "model_name": model_name,
                                    "generation_index": gen_idx,
                                    "word_count": wc,
                                    "candidate_values": seed_row.get("candidate_values", []),
                                    "instruction": seed_row["instruction"],
                                    "context": seed_row["context"],
                                    "raw_model_output": raw,
                                },
                            }
                            rows.append(row)
                            record_counter += 1
                            progress.update(1)
                            progress.set_postfix_str(f"concept={concept}")
                            break
                        except Exception as exc:
                            last_error = exc
                            progress.write(
                                f"[warn] seed={seed_idx:04d} concept={concept} "
                                f"attempt={attempt + 1}/{max_attempts} failed: {exc}"
                            )
                    else:
                        raise RuntimeError(
                            f"Failed to generate concept text for seed={seed_idx}, concept={concept}. "
                            f"Last error: {last_error}"
                        )
    finally:
        progress.close()

    output_path = build_output_path(
        data_config_path=args.data_config,
        configured_output_path=data_config["output"]["dataset_jsonl"],
        model_name=model_name,
        num_concepts=len(concepts),
        generations_per_seed_per_concept=generations_per_seed_per_concept,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} concept-text records to {output_path}")


if __name__ == "__main__":
    main()