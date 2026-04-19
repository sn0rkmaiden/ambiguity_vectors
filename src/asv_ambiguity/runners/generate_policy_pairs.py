from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

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
- Ask directly about the missing slot.
- If the user answered this question, the original instruction should become executable without guessing.
- Do not ask about side details, timing, or background facts unless that is the missing slot.
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


def build_positive_validation_prompt(*, seed_row: dict, candidate_question: str) -> str:
    candidates_block = "\n".join(f"- {x}" for x in seed_row["candidate_values"])
    return f"""Judge whether the assistant question is a good clarifying question for the ambiguous instruction.

Return strict JSON only with these keys:
{{
  "resolves_ambiguity": true or false,
  "targets_missing_slot": true or false,
  "stays_within_context": true or false,
  "single_natural_question": true or false,
  "reason": "short explanation"
}}

Definitions:
- resolves_ambiguity: If the user answered the question, the original instruction could be executed without guessing.
- targets_missing_slot: The question asks about the actual missing slot, not a side detail.
- stays_within_context: The question does not invent new objects, distinctions, or facts beyond the provided context and candidates.
- single_natural_question: It is exactly one short, natural clarifying question.

Ambiguity type: {seed_row['ambiguity_type']}
Missing slot: {seed_row['gold_missing_slot']}

Context:
{seed_row['context']}

Instruction:
{seed_row['instruction']}

Candidate values:
{candidates_block}

Assistant question:
{candidate_question}
"""


def build_positive_repair_prompt(*, seed_row: dict, bad_question: str, feedback: str) -> str:
    candidates_block = "\n".join(f"- {x}" for x in seed_row["candidate_values"])
    return f"""Rewrite the assistant question so that it becomes a good clarifying question.

Requirements:
- Output exactly one question.
- Ask directly about the missing slot.
- If the user answered it, the original instruction should become executable without guessing.
- Use only the given context and candidate values.
- Do not ask about side details or background facts.
- Keep it short, natural, and specific.

Feedback on the bad question:
{feedback}

Bad question:
{bad_question}

Ambiguity type: {seed_row['ambiguity_type']}
Missing slot: {seed_row['gold_missing_slot']}

Context:
{seed_row['context']}

Instruction:
{seed_row['instruction']}

Candidate values:
{candidates_block}

Output one repaired clarifying question only."""


def build_negative_wrong_question_prompt(*, seed_row: dict) -> str:
    candidates_block = "\n".join(f"- {x}" for x in seed_row["candidate_values"])
    return f"""You are creating a hard negative question for an ambiguous instruction.

Your task:
Write exactly one short, natural question that sounds relevant to the same scene or task, but does NOT resolve the core ambiguity.

Rules:
- Output exactly one question.
- It should be a plausible question an assistant might ask.
- It should stay within the context and not invent new objects.
- It should NOT ask which candidate to choose.
- If the user answered this question, the original instruction should still NOT be executable without guessing.
- Keep it short and natural.

Ambiguity type: {seed_row['ambiguity_type']}
Missing slot: {seed_row['gold_missing_slot']}

Context:
{seed_row['context']}

Instruction:
{seed_row['instruction']}

Candidate values:
{candidates_block}

Output one hard-negative question only."""


def build_negative_wrong_validation_prompt(*, seed_row: dict, candidate_question: str) -> str:
    candidates_block = "\n".join(f"- {x}" for x in seed_row["candidate_values"])
    return f"""Judge whether the assistant question is a good hard negative for the ambiguous instruction.

Return strict JSON only with these keys:
{{
  "does_not_resolve_ambiguity": true or false,
  "stays_within_context": true or false,
  "single_natural_question": true or false,
  "still_relevant_to_task": true or false,
  "reason": "short explanation"
}}

Definitions:
- does_not_resolve_ambiguity: Even if the user answered the question, the original instruction would still not be executable without guessing the missing slot.
- stays_within_context: The question does not invent new objects, distinctions, or facts beyond the provided context and candidates.
- single_natural_question: It is exactly one short, natural question.
- still_relevant_to_task: It is not random filler; it sounds related to the same scene or task.

Ambiguity type: {seed_row['ambiguity_type']}
Missing slot: {seed_row['gold_missing_slot']}

Context:
{seed_row['context']}

Instruction:
{seed_row['instruction']}

Candidate values:
{candidates_block}

Assistant question:
{candidate_question}
"""


def build_negative_wrong_repair_prompt(*, seed_row: dict, bad_question: str, feedback: str) -> str:
    candidates_block = "\n".join(f"- {x}" for x in seed_row["candidate_values"])
    return f"""Rewrite the assistant question so that it becomes a good hard negative.

Requirements:
- Output exactly one question.
- Keep it relevant to the same scene or task.
- It must NOT resolve the missing slot.
- If the user answered it, the original instruction should still not be executable without guessing.
- Use only the given context and candidate values.
- Keep it short and natural.

Feedback on the bad question:
{feedback}

Bad question:
{bad_question}

Ambiguity type: {seed_row['ambiguity_type']}
Missing slot: {seed_row['gold_missing_slot']}

Context:
{seed_row['context']}

Instruction:
{seed_row['instruction']}

Candidate values:
{candidates_block}

Output one repaired hard-negative question only."""


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def question_word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def deterministic_question_checks(text: str, *, allow_generic: bool) -> list[str]:
    errors: list[str] = []
    normalized = normalize_space(text)
    lowered = normalized.lower()

    if not normalized:
        return ["empty_output"]
    if not normalized.endswith("?"):
        errors.append("does_not_end_with_question_mark")
    if normalized.count("?") != 1:
        errors.append("must_contain_exactly_one_question_mark")

    wc = question_word_count(normalized)
    if wc < 4:
        errors.append("too_short")
    if wc > 28:
        errors.append("too_long")

    generic_patterns = {
        "could you clarify?",
        "can you clarify?",
        "please clarify?",
        "which one?",
        "which is it?",
        "what do you mean?",
    }
    if not allow_generic and lowered in generic_patterns:
        errors.append("too_generic")

    return errors


def extract_first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError(f"Could not find JSON object in validator output: {text!r}")

    depth = 0
    end = None
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break

    if end is None:
        raise ValueError(f"Could not find complete JSON object in validator output: {text!r}")

    return json.loads(text[start:end])


def validate_positive_question(model: HFCausalModel, seed_row: dict, question: str) -> dict[str, Any]:
    deterministic_errors = deterministic_question_checks(question, allow_generic=False)
    if deterministic_errors:
        return {
            "passed": False,
            "resolves_ambiguity": False,
            "targets_missing_slot": False,
            "stays_within_context": False,
            "single_natural_question": False,
            "reason": "; ".join(deterministic_errors),
            "source": "deterministic",
        }

    raw = model.generate_text(
        build_positive_validation_prompt(seed_row=seed_row, candidate_question=question)
    )
    result = extract_first_json_object(raw)
    passed = all(
        bool(result.get(k, False))
        for k in [
            "resolves_ambiguity",
            "targets_missing_slot",
            "stays_within_context",
            "single_natural_question",
        ]
    )
    result["passed"] = passed
    result["source"] = "llm"
    result["raw_validator_output"] = raw
    return result


def validate_wrong_question(model: HFCausalModel, seed_row: dict, question: str) -> dict[str, Any]:
    deterministic_errors = deterministic_question_checks(question, allow_generic=True)
    if deterministic_errors:
        return {
            "passed": False,
            "does_not_resolve_ambiguity": False,
            "stays_within_context": False,
            "single_natural_question": False,
            "still_relevant_to_task": False,
            "reason": "; ".join(deterministic_errors),
            "source": "deterministic",
        }

    raw = model.generate_text(
        build_negative_wrong_validation_prompt(seed_row=seed_row, candidate_question=question)
    )
    result = extract_first_json_object(raw)
    passed = all(
        bool(result.get(k, False))
        for k in [
            "does_not_resolve_ambiguity",
            "stays_within_context",
            "single_natural_question",
            "still_relevant_to_task",
        ]
    )
    result["passed"] = passed
    result["source"] = "llm"
    result["raw_validator_output"] = raw
    return result


def build_negative_direct_answer(seed_row: dict, rng: random.Random) -> str:
    candidate_values = list(seed_row["candidate_values"])
    chosen_value = rng.choice(candidate_values)
    ambiguity_type = seed_row["ambiguity_type"]

    if ambiguity_type == "destination":
        return f"I'll put it on the {chosen_value}."
    if ambiguity_type == "preference":
        return f"I'll use {chosen_value}."
    return f"I'll use the {chosen_value}."


def build_fallback_negative_wrong_question(seed_row: dict) -> str:
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


def generate_validated_question(
    *,
    model: HFCausalModel,
    seed_row: dict,
    initial_prompt: str,
    validate_fn,
    repair_prompt_builder,
    max_attempts: int,
    max_repairs_per_attempt: int,
) -> tuple[str, dict[str, Any]]:
    last_error: Exception | None = None
    generation_attempts: list[dict[str, Any]] = []

    for attempt_idx in range(max_attempts):
        try:
            raw = model.generate_text(initial_prompt)
            candidate = model.extract_first_question(raw)
            validation = validate_fn(model, seed_row, candidate)
            repairs: list[dict[str, Any]] = []

            if validation.get("passed", False):
                generation_attempts.append(
                    {
                        "stage": "initial",
                        "raw_model_output": raw,
                        "candidate_question": candidate,
                        "validation": validation,
                        "repairs": repairs,
                    }
                )
                return candidate, {
                    "attempt_count": attempt_idx + 1,
                    "attempts": generation_attempts,
                }

            repaired_candidate = candidate
            repaired_validation = validation
            for repair_idx in range(max_repairs_per_attempt):
                repair_prompt = repair_prompt_builder(
                    seed_row=seed_row,
                    bad_question=repaired_candidate,
                    feedback=str(repaired_validation.get("reason", "failed validation")),
                )
                repair_raw = model.generate_text(repair_prompt)
                repaired_candidate = model.extract_first_question(repair_raw)
                repaired_validation = validate_fn(model, seed_row, repaired_candidate)
                repairs.append(
                    {
                        "repair_index": repair_idx,
                        "raw_model_output": repair_raw,
                        "candidate_question": repaired_candidate,
                        "validation": repaired_validation,
                    }
                )
                if repaired_validation.get("passed", False):
                    generation_attempts.append(
                        {
                            "stage": "initial",
                            "raw_model_output": raw,
                            "candidate_question": candidate,
                            "validation": validation,
                            "repairs": repairs,
                        }
                    )
                    return repaired_candidate, {
                        "attempt_count": attempt_idx + 1,
                        "attempts": generation_attempts,
                    }

            generation_attempts.append(
                {
                    "stage": "initial",
                    "raw_model_output": raw,
                    "candidate_question": candidate,
                    "validation": validation,
                    "repairs": repairs,
                }
            )
        except Exception as exc:
            last_error = exc
            generation_attempts.append(
                {
                    "stage": "exception",
                    "error": repr(exc),
                }
            )

    raise RuntimeError(
        f"Failed to generate a validated question after {max_attempts} attempts. "
        f"Last error: {last_error}. Attempt metadata: {generation_attempts}"
    )


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

    sampling_cfg = data_config.get("sampling", {})
    validation_cfg = data_config.get("validation", {})

    generations_per_seed = int(sampling_cfg.get("generations_per_seed", 1))
    max_attempts = int(sampling_cfg.get("max_attempts_per_example", 5))
    preserve_input_split = bool(data_config["splits"].get("preserve_input_split", False))
    max_repairs_per_attempt = int(validation_cfg.get("max_repairs_per_attempt", 1))
    use_model_generated_wrong_questions = bool(
        validation_cfg.get("use_model_generated_wrong_questions", True)
    )
    fallback_to_seed_wrong_question = bool(
        validation_cfg.get("fallback_to_seed_wrong_question", True)
    )

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

                positive_prompt = build_positive_question_prompt(
                    context=seed_row["context"],
                    instruction=seed_row["instruction"],
                    candidate_values=seed_row["candidate_values"],
                    ambiguity_type=seed_row["ambiguity_type"],
                    gold_missing_slot=seed_row["gold_missing_slot"],
                )

                try:
                    positive_response, positive_meta = generate_validated_question(
                        model=model,
                        seed_row=seed_row,
                        initial_prompt=positive_prompt,
                        validate_fn=validate_positive_question,
                        repair_prompt_builder=build_positive_repair_prompt,
                        max_attempts=max_attempts,
                        max_repairs_per_attempt=max_repairs_per_attempt,
                    )

                    wrong_question_meta: dict[str, Any] = {
                        "source": "fallback",
                        "attempt_count": 0,
                        "attempts": [],
                    }
                    if use_model_generated_wrong_questions:
                        try:
                            negative_wrong_question, wrong_question_meta = generate_validated_question(
                                model=model,
                                seed_row=seed_row,
                                initial_prompt=build_negative_wrong_question_prompt(seed_row=seed_row),
                                validate_fn=validate_wrong_question,
                                repair_prompt_builder=build_negative_wrong_repair_prompt,
                                max_attempts=max_attempts,
                                max_repairs_per_attempt=max_repairs_per_attempt,
                            )
                        except Exception as exc:
                            if fallback_to_seed_wrong_question:
                                negative_wrong_question = build_fallback_negative_wrong_question(seed_row)
                                wrong_question_meta = {
                                    "source": "fallback_after_generation_failure",
                                    "error": repr(exc),
                                    "attempt_count": 0,
                                    "attempts": [],
                                }
                            else:
                                raise
                    else:
                        negative_wrong_question = build_fallback_negative_wrong_question(seed_row)

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
                        "negative_direct_answer": build_negative_direct_answer(seed_row, rng),
                        "negative_wrong_question": negative_wrong_question,
                        "candidate_values": seed_row["candidate_values"],
                        # kept for backward compatibility with earlier utilities
                        "candidate_referents": seed_row["candidate_values"],
                        "metadata": {
                            "generator": "seeded_context_model_question",
                            "model_name": model_name_for_filename,
                            "seed_row_index": seed_idx,
                            "generation_index": gen_idx,
                            "positive_generation": positive_meta,
                            "negative_wrong_generation": wrong_question_meta,
                        },
                    }

                    rows.append(row)
                    counter += 1
                    progress.update(1)
                    progress.set_postfix_str(
                        f"type={seed_row['ambiguity_type']} topic={seed_row['topic']}"
                    )
                except Exception as exc:
                    progress.write(
                        f"[warn] topic={seed_row['topic']!r} example={counter:05d} failed: {exc}"
                    )
                    raise RuntimeError(
                        f"Failed to build example for topic={seed_row['topic']!r}, "
                        f"example={counter:05d}. Last error: {exc}"
                    ) from exc
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
