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


def build_dataset_output_path(
    *,
    data_config_path: str,
    configured_output_path: str,
    concept_name: str,
    model_name: str,
    generations_per_seed: int,
    seed: int,
) -> Path:
    base_output_path = resolve_project_path(data_config_path, configured_output_path)
    output_dir = base_output_path.parent
    filename = (
        f"{slugify(concept_name)}"
        f"__{slugify(model_name)}"
        f"__gps{generations_per_seed}"
        f"__seed{seed}.jsonl"
    )
    return output_dir / filename


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_positive_question_prompt(
    *,
    context: str,
    instruction: str,
    candidate_values: list[str],
    ambiguity_type: str,
    gold_missing_slot: str,
) -> str:
    candidates_block = "\n".join(f"- {x}" for x in candidate_values)

    return f"""You are creating a high-quality clarifying question for an ambiguous instruction.

Your task:
Write exactly one short clarifying question that resolves the missing information.

Rules:
- Output exactly one question.
- Do not answer the instruction.
- Do not add explanation.
- Use only the ambiguity described in the context and candidate values.
- Do not invent new distinctions or new objects.
- The question should directly help choose the missing information.
- Keep it natural and specific.

Ambiguity type: {ambiguity_type}
Missing slot: {gold_missing_slot}

Context:
{context}

Instruction:
{instruction}

Candidate values:
{candidates_block}

Output one clarifying question only."""
    

def build_negative_direct_answer(seed_row: dict) -> str:
    first_value = seed_row["candidate_values"][0]
    ambiguity_type = seed_row["ambiguity_type"]

    if ambiguity_type == "destination":
        return f"I'll put it on the {first_value}."
    if ambiguity_type == "preference":
        return f"I'll use {first_value}."
    return f"I'll use the {first_value}."


def build_negative_wrong_question(seed_row: dict) -> str:
    custom = seed_row.get("negative_wrong_question")
    if custom is not None and str(custom).strip():
        return str(custom).strip()

    ambiguity_type = seed_row["ambiguity_type"]

    if ambiguity_type == "object_identity":
        return "Where should I put it?"
    if ambiguity_type == "preference":
        return "Should I serve it now?"
    if ambiguity_type == "destination":
        return "Should I put it away now?"
    return "Could you clarify?"


def maybe_get_input_split(seed_row: dict) -> str | None:
    split = seed_row.get("split")
    if split in {"train", "val", "test"}:
        return split
    return None


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
    model_name_for_filename = getattr(model, "model_name", model_config["model"]["name"])

    seed_path = resolve_project_path(args.data_config, data_config["input"]["seed_jsonl"])
    seed_rows = read_jsonl(seed_path)

    generations_per_seed = int(data_config["sampling"].get("generations_per_seed", 1))
    max_attempts = int(data_config["sampling"].get("max_attempts_per_example", 5))
    preserve_input_split = bool(data_config["splits"].get("preserve_input_split", False))

    rng = random.Random(seed)
    shuffled_seed_rows = seed_rows[:]
    rng.shuffle(shuffled_seed_rows)

    split_assigner = SplitAssigner(
        train_ratio=float(data_config["splits"]["train_ratio"]),
        val_ratio=float(data_config["splits"]["val_ratio"]),
    )

    rows = []
    total_items = len(shuffled_seed_rows) * generations_per_seed
    counter = 0

    progress = tqdm(total=total_items, desc="Generating policy pairs", unit="example")

    try:
        for seed_idx, seed_row in enumerate(shuffled_seed_rows):
            for gen_idx in range(generations_per_seed):
                if preserve_input_split:
                    split = maybe_get_input_split(seed_row)
                    if split is None:
                        raise ValueError(
                            "preserve_input_split=true but a seed row has no valid split field."
                        )
                else:
                    split = split_assigner.assign(counter, total_items)

                prompt = build_positive_question_prompt(
                    context=seed_row["context"],
                    instruction=seed_row["instruction"],
                    candidate_values=seed_row["candidate_values"],
                    ambiguity_type=seed_row["ambiguity_type"],
                    gold_missing_slot=seed_row["gold_missing_slot"],
                )

                last_error = None
                for attempt in range(max_attempts):
                    try:
                        raw = model.generate_text(prompt)
                        positive_response = model.extract_first_question(raw)

                        row = {
                            "example_id": f"seeded_{counter:05d}",
                            "concept_name": data_config["concept"]["name"],
                            "topic": seed_row["topic"],
                            "ambiguity_type": seed_row["ambiguity_type"],
                            "gold_missing_slot": seed_row["gold_missing_slot"],
                            "split": split,
                            "context": seed_row["context"],
                            "instruction": seed_row["instruction"],
                            "positive_response": positive_response,
                            "negative_direct_answer": build_negative_direct_answer(seed_row),
                            "negative_wrong_question": build_negative_wrong_question(seed_row),
                            "candidate_values": seed_row["candidate_values"],
                            # kept for backward compatibility with earlier utilities
                            "candidate_referents": seed_row["candidate_values"],
                            "metadata": {
                                "generator": "seeded_context_model_question",
                                "model_name": model_name_for_filename,
                                "seed_row_index": seed_idx,
                                "generation_index": gen_idx,
                                "raw_model_output": raw,
                            },
                        }

                        rows.append(row)
                        counter += 1
                        progress.update(1)
                        progress.set_postfix_str(
                            f"type={seed_row['ambiguity_type']} topic={seed_row['topic']}"
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                        progress.write(
                            f"[warn] topic={seed_row['topic']!r} example={counter:05d} "
                            f"attempt={attempt + 1}/{max_attempts} failed: {exc}"
                        )
                else:
                    raise RuntimeError(
                        f"Failed to build example for topic={seed_row['topic']!r}, "
                        f"example={counter:05d} after {max_attempts} attempts. "
                        f"Last error: {last_error}"
                    )
    finally:
        progress.close()

    output_path = build_dataset_output_path(
        data_config_path=args.data_config,
        configured_output_path=data_config["output"]["dataset_jsonl"],
        concept_name=data_config["concept"]["name"],
        model_name=model_name_for_filename,
        generations_per_seed=generations_per_seed,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} examples to {output_path}")


if __name__ == "__main__":
    main()